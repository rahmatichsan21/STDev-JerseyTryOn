"""
Real-time Football Jersey Filter Server - HYBRID APPROACH
==========================================================

Masalah dengan optimized version:
- ML model terlalu lambat bahkan dengan optimisasi
- Detection tidak reliable

Solusi HYBRID:
- Gunakan COLOR-BASED detection (jauh lebih cepat!)
- Fallback ke ML hanya untuk captured images
- Real-time: Speed first, ML for final result
"""

import asyncio
import websockets
import cv2
import json
import sys
from pathlib import Path
import numpy as np
import joblib
import time

# Add DatasetImplementation to path
DATASET_PATH = Path(__file__).parent / "DatasetImplementation"
sys.path.insert(0, str(DATASET_PATH))

from shirt_detector_full import extract_pixel_features

class HybridJerseyFilterServer:
    def __init__(self):
        self.cap = None
        self.is_streaming = True
        self.clients = set()
        self.classifier = None
        self.jersey = None
        self.jersey_path = None
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.frame_counter = 0
        
        self.load_model()
        self.load_jersey()
        
    def load_model(self):
        """Load trained Random Forest classifier"""
        model_path = DATASET_PATH / "shirt_detector_model.joblib"
        
        if not model_path.exists():
            print(f"⚠️  Model not found - will use color-based detection only")
            return False
        
        try:
            self.classifier = joblib.load(model_path)
            print(f"✓ ML Model loaded (for captured images only)")
            return True
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            return False
    
    def load_jersey(self):
        """Load Brighton Home jersey"""
        self.jersey_path = Path(__file__).parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home_NOBG" / "Brighton Home.png"
        
        if not self.jersey_path.exists():
            print(f"✗ Jersey not found at {self.jersey_path}")
            return False
        
        self.jersey = cv2.imread(str(self.jersey_path), cv2.IMREAD_UNCHANGED)
        if self.jersey is None:
            print(f"✗ Failed to load jersey")
            return False
        
        print(f"✓ Brighton Home jersey loaded: {self.jersey.shape}")
        return True
    
    def detect_shirt_color_based(self, image):
        """
        FAST color-based shirt detection for real-time
        Strategi: Detect skin dan hair, sisanya adalah shirt
        """
        h, w = image.shape[:2]
        
        # Remove background
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = (gray > 20).astype(np.uint8) * 255
        
        # === SKIN DETECTION ===
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skin_mask1 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Clean skin mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # === FIND NECK LEVEL ===
        upper_half = np.zeros_like(skin_mask)
        upper_half[:int(h * 0.5), :] = 255
        face_skin = cv2.bitwise_and(skin_mask, upper_half)
        
        face_contours, _ = cv2.findContours(face_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        neck_y = int(h * 0.35)
        if len(face_contours) > 0:
            face_contour = max(face_contours, key=cv2.contourArea)
            if cv2.contourArea(face_contour) > 1000:
                bottommost = tuple(face_contour[face_contour[:, :, 1].argmax()][0])
                neck_y = min(bottommost[1] + 20, int(h * 0.5))
        
        # === HAIR DETECTION ===
        hair_mask = ((gray < 50) & (foreground > 0)).astype(np.uint8) * 255
        above_neck = np.zeros_like(hair_mask)
        above_neck[:neck_y, :] = 255
        hair_mask = cv2.bitwise_and(hair_mask, above_neck)
        
        # === SHIRT = Below neck - Skin (arms) ===
        below_neck = np.zeros_like(foreground)
        below_neck[neck_y:, :] = 255
        
        potential_shirt = cv2.bitwise_and(foreground, below_neck)
        
        # Remove skin from shirt region (arms)
        shirt_mask = cv2.subtract(potential_shirt, skin_mask)
        
        # Clean up shirt mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel)
        shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate sedikit untuk coverage lebih baik
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shirt_mask = cv2.dilate(shirt_mask, kernel_dilate, iterations=1)
        
        return shirt_mask
    
    def predict_shirt_mask_ml(self, image):
        """
        ML-based detection - HANYA untuk captured images
        """
        if self.classifier is None:
            return self.detect_shirt_color_based(image)
        
        h, w = image.shape[:2]
        
        # Get foreground
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = gray > 20
        
        coords = np.where(foreground)
        
        if len(coords[0]) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Sample pixels (jangan semua untuk speed)
        sample_rate = 2
        sampled_indices = np.arange(0, len(coords[0]), sample_rate)
        
        sampled_y = coords[0][sampled_indices]
        sampled_x = coords[1][sampled_indices]
        
        pixels = image[sampled_y, sampled_x]
        features = np.array([extract_pixel_features(px) for px in pixels])
        predictions = self.classifier.predict(features)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[sampled_y, sampled_x] = predictions.astype(np.uint8)
        
        # Fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def apply_jersey_to_frame(self, frame, use_ml=False):
        """
        Apply jersey dengan mode selection
        use_ml=False: Color-based (FAST untuk real-time)
        use_ml=True: ML-based (ACCURATE untuk captured images)
        """
        if self.jersey is None:
            return frame
        
        try:
            # Choose detection method
            if use_ml and self.classifier is not None:
                shirt_mask = self.predict_shirt_mask_ml(frame)
            else:
                shirt_mask = self.detect_shirt_color_based(frame)
            
            # Debug logging
            mask_pixels = np.sum(shirt_mask > 0)
            if self.frame_counter % 60 == 0 and not use_ml:
                print(f"🔍 Color detection: {mask_pixels} shirt pixels")
            
            # Check detection
            if mask_pixels < 500:
                if self.frame_counter % 120 == 0 and not use_ml:
                    print("⚠️  Low shirt detection - check camera position")
                return frame
            
            # Find shirt bounding box
            contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return frame
            
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Resize jersey
            resized_jersey = cv2.resize(self.jersey, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Extract alpha
            if resized_jersey.shape[2] == 4:
                jersey_bgr = resized_jersey[:, :, :3]
                jersey_alpha = resized_jersey[:, :, 3] / 255.0
            else:
                jersey_bgr = resized_jersey
                jersey_alpha = np.ones((h, w))
            
            # Blend
            result = frame.copy()
            overlay_region = result[y:y+h, x:x+w]
            shirt_region_mask = shirt_mask[y:y+h, x:x+w].astype(float) / 255.0
            
            combined_alpha = jersey_alpha * shirt_region_mask
            combined_alpha_3d = combined_alpha[:, :, np.newaxis]
            
            blended = (combined_alpha_3d * jersey_bgr + 
                      (1 - combined_alpha_3d) * overlay_region).astype(np.uint8)
            
            result[y:y+h, x:x+w] = blended
            
            return result
            
        except Exception as e:
            if not use_ml:
                print(f"Error applying jersey: {e}")
            return frame
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        print(f"✓ Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                try:
                    command = json.loads(message)
                    msg_type = command.get("type", "")
                    
                    if msg_type == "save_capture":
                        self.is_streaming = False
                        await websocket.send(json.dumps({"type": "capture_ready"}))
                        print("📸 Capture requested")
                    
                    elif msg_type == "resume":
                        self.is_streaming = True
                        print("▶ Streaming resumed")
                    
                    elif msg_type == "process_image":
                        image_path = command.get("image_path", "")
                        output_path = command.get("output_path", "")
                        
                        success = await self.process_captured_image(image_path, output_path)
                        
                        response = {
                            "type": "processing_complete",
                            "success": success,
                            "output_path": output_path if success else ""
                        }
                        await websocket.send(json.dumps(response))
                
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {message}")
                except Exception as e:
                    print(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            self.clients.remove(websocket)
            print(f"✓ Client removed. Total clients: {len(self.clients)}")
    
    async def process_captured_image(self, image_path, output_path):
        """Process captured image dengan ML (high quality)"""
        try:
            print(f"\n{'='*60}")
            print(f"PROCESSING CAPTURED IMAGE (ML MODE)")
            print(f"{'='*60}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"✗ Failed to load: {image_path}")
                return False
            
            print(f"✓ Image loaded: {image.shape}")
            print("Applying jersey with ML detection...")
            
            # Use ML for captured image
            result = self.apply_jersey_to_frame(image, use_ml=True)
            
            cv2.imwrite(output_path, result)
            print(f"✓ Saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def update_fps(self):
        """Calculate FPS"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 2.0:
            self.current_fps = self.fps_counter / elapsed
            print(f"📊 Current FPS: {self.current_fps:.1f}")
            
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    async def video_stream(self):
        """Video streaming dengan color-based detection"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("✗ Failed to open webcam")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("✓ Webcam opened")
        print("📹 Starting video stream with COLOR-BASED detection...")
        print("   (ML detection will be used for captured images only)")
        print()
        
        while True:
            if not self.is_streaming:
                await asyncio.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # Apply jersey with color-based detection (FAST!)
            processed_frame = self.apply_jersey_to_frame(frame, use_ml=False)
            
            # Encode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, jpeg_data = cv2.imencode('.jpg', processed_frame, encode_param)
            
            # Send to clients
            if self.clients:
                await asyncio.gather(
                    *[client.send(jpeg_data.tobytes()) for client in self.clients],
                    return_exceptions=True
                )
            
            self.frame_counter += 1
            self.update_fps()
            
            # Minimal delay untuk max throughput
            await asyncio.sleep(0.001)
    
    async def start_server(self):
        """Start server"""
        print("=" * 80)
        print("HYBRID FOOTBALL JERSEY FILTER SERVER")
        print("=" * 80)
        print()
        
        if self.jersey is None:
            print("✗ Cannot start: Jersey not loaded")
            return
        
        print("✓ Server ready with HYBRID approach!")
        print()
        print("Strategy:")
        print("  • Real-time preview: COLOR-BASED detection (fast!)")
        print("  • Captured images: ML detection (accurate!)")
        print()
        print("Expected: 25-30 FPS with good filter accuracy")
        print()
        print("Starting WebSocket server on ws://localhost:8765")
        print("Waiting for Godot connection...")
        print()
        
        async with websockets.serve(self.handle_client, "localhost", 8765):
            await self.video_stream()
    
    def cleanup(self):
        """Cleanup"""
        if self.cap is not None:
            self.cap.release()
        print("\n✓ Server shutdown complete")


async def main():
    server = HybridJerseyFilterServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n\n⚠ Shutting down...")
    finally:
        server.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
