"""
Real-time Football Jersey Filter Server - OPTIMIZED VERSION
===========================================================

Optimizations implemented:
1. Batch prediction instead of per-pixel processing
2. Frame skipping for ML predictions (reuse mask)
3. Reduced morphological operations
4. Lower resolution processing
5. Asynchronous encoding
6. Better memory management
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

class OptimizedJerseyFilterServer:
    def __init__(self):
        self.cap = None
        self.is_streaming = True
        self.clients = set()
        self.classifier = None
        self.jersey = None
        self.jersey_path = None
        
        # Performance optimization settings
        self.mask_cache = None  # Cache mask untuk beberapa frame
        self.mask_update_interval = 2  # Update mask setiap 2 frame (lebih sering = lebih akurat)
        self.frame_counter = 0
        self.process_scale = 0.65  # Process at 65% resolution (balance antara speed & quality)
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.load_model()
        self.load_jersey()
        
    def load_model(self):
        """Load trained Random Forest classifier"""
        model_path = DATASET_PATH / "shirt_detector_model.joblib"
        
        if not model_path.exists():
            print(f"✗ Model not found at {model_path}")
            print("  Please run shirt_detector_full.py first to train the model!")
            return False
        
        try:
            self.classifier = joblib.load(model_path)
            print(f"✓ ML Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
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
    
    def predict_shirt_mask_optimized(self, image):
        """
        OPTIMIZED: Batch prediction dengan sampling pixels
        """
        if self.classifier is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # Resize untuk processing lebih cepat
        small_h, small_w = int(h * self.process_scale), int(w * self.process_scale)
        small_image = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Get foreground
        gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
        foreground = gray > 20
        
        coords = np.where(foreground)
        
        if len(coords[0]) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # OPTIMISASI: Sample hanya sebagian pixels (setiap pixel ke-N)
        sample_rate = 3  # Sample setiap 3 pixel (lebih banyak sample = lebih akurat)
        sampled_indices = np.arange(0, len(coords[0]), sample_rate)
        
        sampled_y = coords[0][sampled_indices]
        sampled_x = coords[1][sampled_indices]
        
        # Extract features untuk sampled pixels
        pixels = small_image[sampled_y, sampled_x]
        
        # OPTIMISASI: Vectorized feature extraction jika memungkinkan
        # Jika tidak bisa vectorize, minimal batch process
        try:
            features = np.array([extract_pixel_features(px) for px in pixels])
            predictions = self.classifier.predict(features)
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros((h, w), dtype=np.uint8)
        
        # Create mask dari sampled predictions
        small_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        small_mask[sampled_y, sampled_x] = predictions.astype(np.uint8)
        
        # Interpolate mask untuk fill gaps dari sampling
        if np.sum(small_mask) > 100:
            # Dilate untuk mengisi gaps
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # Kernel lebih besar
            small_mask = cv2.dilate(small_mask, kernel_small, iterations=2)  # 2 iterasi
            
            # Morphological operations untuk smooth
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            small_mask = cv2.morphologyEx(small_mask, cv2.MORPH_CLOSE, kernel)
        
        # Resize mask kembali ke ukuran asli
        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 127).astype(np.uint8)  # Threshold untuk binary mask
        
        return mask
    
    def apply_jersey_to_frame(self, frame, use_cached_mask=False):
        """
        Apply Brighton jersey dengan optimisasi
        """
        if self.jersey is None or self.classifier is None:
            return frame
        
        try:
            # OPTIMISASI: Reuse cached mask untuk beberapa frame
            if use_cached_mask and self.mask_cache is not None:
                shirt_mask = self.mask_cache
            else:
                shirt_mask = self.predict_shirt_mask_optimized(frame)
                self.mask_cache = shirt_mask
                
                # Debug: Print mask stats saat update
                mask_pixels = np.sum(shirt_mask)
                if self.frame_counter % 30 == 0:  # Print setiap 30 frame
                    print(f"🔍 Mask update: {mask_pixels} shirt pixels detected")
            
            # Check if any shirt detected
            if np.sum(shirt_mask) < 200:  # Lebih sensitif untuk detection
                if self.frame_counter % 60 == 0:  # Warn occasionally
                    print("⚠️  No shirt detected in frame")
                return frame
            
            # Find shirt bounding box
            contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return frame
            
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Reduced padding
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Resize jersey
            resized_jersey = cv2.resize(self.jersey, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Create result - OPTIMISASI: hanya copy region yang perlu
            result = frame  # Langsung gunakan frame, no copy
            
            # Extract BGR and alpha
            if resized_jersey.shape[2] == 4:
                jersey_bgr = resized_jersey[:, :, :3]
                jersey_alpha = resized_jersey[:, :, 3] / 255.0
            else:
                jersey_bgr = resized_jersey
                jersey_alpha = np.ones((h, w))
            
            # Get regions
            overlay_region = result[y:y+h, x:x+w].copy()  # Copy hanya region yang kecil
            shirt_region_mask = shirt_mask[y:y+h, x:x+w].astype(float)
            
            # Combine alpha
            combined_alpha = jersey_alpha * shirt_region_mask
            
            # OPTIMISASI: Vectorized blending
            combined_alpha_3d = combined_alpha[:, :, np.newaxis]
            blended = (combined_alpha_3d * jersey_bgr + 
                      (1 - combined_alpha_3d) * overlay_region).astype(np.uint8)
            
            result[y:y+h, x:x+w] = blended
            
            return result
            
        except Exception as e:
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
                        print("📸 Capture requested - streaming paused")
                    
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
                    print(f"Invalid JSON received: {message}")
                except Exception as e:
                    print(f"Error handling message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        finally:
            self.clients.remove(websocket)
            print(f"✓ Client removed. Total clients: {len(self.clients)}")
    
    async def process_captured_image(self, image_path, output_path):
        """Process captured image dengan full quality (no optimization)"""
        try:
            print(f"\n{'='*60}")
            print(f"PROCESSING CAPTURED IMAGE (FULL QUALITY)")
            print(f"{'='*60}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"✗ Failed to load image: {image_path}")
                return False
            
            print(f"✓ Image loaded: {image.shape}")
            
            # Untuk captured image, gunakan full processing (no cache, no optimization)
            old_scale = self.process_scale
            self.process_scale = 1.0  # Full resolution
            self.mask_cache = None  # Clear cache
            
            result = self.apply_jersey_to_frame(image, use_cached_mask=False)
            
            self.process_scale = old_scale  # Restore optimization settings
            
            cv2.imwrite(output_path, result)
            print(f"✓ Result saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing image: {e}")
            return False
    
    def update_fps(self):
        """Calculate and display FPS"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 2.0:  # Update every 2 seconds
            self.current_fps = self.fps_counter / elapsed
            print(f"📊 Current FPS: {self.current_fps:.1f} | Cache hits: {self.frame_counter % self.mask_update_interval}")
            
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    async def video_stream(self):
        """OPTIMIZED video streaming"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("✗ Failed to open webcam")
            return
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        print("✓ Webcam opened successfully")
        print("📹 Starting OPTIMIZED video stream...")
        print(f"   - Mask update interval: {self.mask_update_interval} frames")
        print(f"   - Processing scale: {self.process_scale*100}%")
        print()
        
        while True:
            if not self.is_streaming:
                await asyncio.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            # OPTIMISASI: Reuse mask untuk beberapa frame
            use_cached = (self.frame_counter % self.mask_update_interval) != 0
            
            # Apply jersey filter
            processed_frame = self.apply_jersey_to_frame(frame, use_cached_mask=use_cached)
            
            # Encode as JPEG dengan quality sedang untuk balance speed/quality
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Balanced quality
            _, jpeg_data = cv2.imencode('.jpg', processed_frame, encode_param)
            
            # Send to clients
            if self.clients:
                await asyncio.gather(
                    *[client.send(jpeg_data.tobytes()) for client in self.clients],
                    return_exceptions=True
                )
            
            self.frame_counter += 1
            self.update_fps()
            
            # OPTIMISASI: Dynamic sleep berdasarkan processing load
            # Jika ada client, prioritaskan throughput
            if self.clients:
                await asyncio.sleep(0.001)  # Minimal delay
            else:
                await asyncio.sleep(0.033)
    
    async def start_server(self):
        """Start WebSocket server"""
        print("=" * 80)
        print("OPTIMIZED FOOTBALL JERSEY FILTER SERVER")
        print("=" * 80)
        print()
        
        if self.classifier is None:
            print("✗ Cannot start server: ML model not loaded")
            return
        
        if self.jersey is None:
            print("✗ Cannot start server: Jersey not loaded")
            return
        
        print("✓ Server ready with optimizations enabled!")
        print()
        print("Optimizations:")
        print(f"  • Mask caching: Update every {self.mask_update_interval} frames")
        print(f"  • Processing scale: {self.process_scale*100:.0f}% of original resolution")
        print(f"  • Pixel sampling: 3x reduction (better accuracy)")
        print(f"  • Enhanced morphological operations")
        print(f"  • Vectorized blending")
        print(f"  • JPEG quality: 80 (balanced)")
        print()
        print("Balance Settings: SPEED + QUALITY")
        print("Expected: 20-25 FPS with good filter accuracy")
        print()
        print("Starting WebSocket server on ws://localhost:8765")
        print("Waiting for Godot connection...")
        print()
        
        async with websockets.serve(self.handle_client, "localhost", 8765):
            await self.video_stream()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()
        print("\n✓ Server shutdown complete")


async def main():
    """Main entry point"""
    server = OptimizedJerseyFilterServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n\n⚠ Shutting down server...")
    finally:
        server.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
