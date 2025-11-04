"""
Real-time Football Jersey Filter Server for Godot
==================================================

This server applies Brighton Home jersey using trained ML model.
Optimized for real-time performance with traditional machine learning.
"""

import asyncio
import websockets
import cv2
import json
import sys
from pathlib import Path
import numpy as np
import joblib

# Add DatasetImplementation to path
DATASET_PATH = Path(__file__).parent / "DatasetImplementation"
sys.path.insert(0, str(DATASET_PATH))

# Import our shirt detection functions
from shirt_detector_full import extract_pixel_features

class JerseyFilterServer:
    def __init__(self):
        self.cap = None
        self.is_streaming = True
        self.clients = set()
        self.classifier = None
        self.jersey = None
        self.jersey_path = None
        
        # Load trained model
        self.load_model()
        
        # Load Brighton jersey
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
    
    def predict_shirt_mask_fast(self, image):
        """
        Fast shirt detection using trained classifier
        """
        if self.classifier is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        h, w = image.shape[:2]
        
        # Get foreground (non-background pixels)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = gray > 20
        
        coords = np.where(foreground)
        
        if len(coords[0]) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # Extract features for all foreground pixels
        pixels = image[coords]
        features = np.array([extract_pixel_features(px) for px in pixels])
        
        # Predict
        predictions = self.classifier.predict(features)
        
        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[coords] = predictions.astype(np.uint8)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def apply_jersey_to_frame(self, frame):
        """
        Apply Brighton jersey to detected shirt region
        """
        if self.jersey is None or self.classifier is None:
            return frame
        
        try:
            # Detect shirt mask
            shirt_mask = self.predict_shirt_mask_fast(frame)
            
            # Check if any shirt detected
            if np.sum(shirt_mask) < 1000:  # Too few shirt pixels
                return frame
            
            # Find shirt bounding box
            contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return frame
            
            main_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Resize jersey to match shirt dimensions
            resized_jersey = cv2.resize(self.jersey, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Create result
            result = frame.copy()
            
            # Extract BGR and alpha
            if resized_jersey.shape[2] == 4:
                jersey_bgr = resized_jersey[:, :, :3]
                jersey_alpha = resized_jersey[:, :, 3] / 255.0
            else:
                jersey_bgr = resized_jersey
                jersey_alpha = np.ones((h, w))
            
            # Create region to overlay
            overlay_region = result[y:y+h, x:x+w]
            shirt_region_mask = shirt_mask[y:y+h, x:x+w].astype(float)
            
            # Combine alpha with shirt mask
            combined_alpha = jersey_alpha * shirt_region_mask
            
            # Blend jersey onto result
            for c in range(3):
                overlay_region[:, :, c] = (
                    combined_alpha * jersey_bgr[:, :, c] +
                    (1 - combined_alpha) * overlay_region[:, :, c]
                )
            
            result[y:y+h, x:x+w] = overlay_region
            
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
                    # Parse JSON command
                    command = json.loads(message)
                    msg_type = command.get("type", "")
                    
                    if msg_type == "save_capture":
                        # Pause streaming for capture
                        self.is_streaming = False
                        await websocket.send(json.dumps({"type": "capture_ready"}))
                        print("📸 Capture requested - streaming paused")
                    
                    elif msg_type == "resume":
                        # Resume streaming
                        self.is_streaming = True
                        print("▶ Streaming resumed")
                    
                    elif msg_type == "process_image":
                        # Process captured image with jersey
                        image_path = command.get("image_path", "")
                        output_path = command.get("output_path", "")
                        
                        success = await self.process_captured_image(image_path, output_path)
                        
                        response = {
                            "type": "processing_complete",
                            "success": success,
                            "output_path": output_path if success else ""
                        }
                        await websocket.send(json.dumps(response))
                    
                    elif msg_type == "set_jersey":
                        # Change jersey (future feature)
                        jersey_name = command.get("jersey", "Brighton Home.png")
                        print(f"Jersey change requested: {jersey_name}")
                        # TODO: Implement dynamic jersey switching
                
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
        """Process a captured image and save with jersey applied"""
        try:
            print(f"\n{'='*60}")
            print(f"PROCESSING CAPTURED IMAGE")
            print(f"{'='*60}")
            print(f"Input: {image_path}")
            print(f"Output: {output_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"✗ Failed to load image: {image_path}")
                return False
            
            print(f"✓ Image loaded: {image.shape}")
            
            # Apply jersey
            print("Applying Brighton jersey...")
            result = self.apply_jersey_to_frame(image)
            
            # Save result
            cv2.imwrite(output_path, result)
            print(f"✓ Result saved to: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing image: {e}")
            return False
    
    async def video_stream(self):
        """Capture webcam and stream with jersey filter"""
        # Open webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("✗ Failed to open webcam")
            return
        
        # Set webcam properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Webcam opened successfully")
        print("📹 Starting video stream with jersey filter...")
        
        frame_count = 0
        
        while True:
            if not self.is_streaming:
                await asyncio.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print("✗ Failed to read frame")
                await asyncio.sleep(0.1)
                continue
            
            # Apply jersey filter
            processed_frame = self.apply_jersey_to_frame(frame)
            
            # Encode as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, jpeg_data = cv2.imencode('.jpg', processed_frame, encode_param)
            
            # Send to all connected clients
            if self.clients:
                await asyncio.gather(
                    *[client.send(jpeg_data.tobytes()) for client in self.clients],
                    return_exceptions=True
                )
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"📊 Frames processed: {frame_count}")
            
            # Small delay to prevent overwhelming the clients
            await asyncio.sleep(0.033)  # ~30 FPS
    
    async def start_server(self):
        """Start WebSocket server"""
        print("=" * 80)
        print("FOOTBALL JERSEY FILTER SERVER")
        print("=" * 80)
        print()
        
        if self.classifier is None:
            print("✗ Cannot start server: ML model not loaded")
            return
        
        if self.jersey is None:
            print("✗ Cannot start server: Jersey not loaded")
            return
        
        print("✓ Server ready!")
        print()
        print("Starting WebSocket server on ws://localhost:8765")
        print("Waiting for Godot connection...")
        print()
        
        # Start WebSocket server
        async with websockets.serve(self.handle_client, "localhost", 8765):
            # Run video streaming
            await self.video_stream()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()
        print("\n✓ Server shutdown complete")


async def main():
    """Main entry point"""
    server = JerseyFilterServer()
    
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
