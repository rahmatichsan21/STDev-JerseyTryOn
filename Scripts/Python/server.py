import asyncio
import websockets
import cv2
import json
import os
import sys
from pathlib import Path

# Add DatasetImplementation to path for imports
DATASET_PATH = Path(__file__).parent / "DatasetImplementation"
sys.path.insert(0, str(DATASET_PATH))

import numpy as np

# Import jersey application modules
try:
    from pose_detection import detect_pose_landmarks
    from detect_shirt_region import get_shirt_keypoints
    from detect_actual_shirt import detect_skin, create_shirt_only_mask, refine_with_grabcut
    print("✓ Jersey processing modules loaded successfully")
except ImportError as e:
    print(f"✗ Failed to import jersey modules: {e}")
    print(f"   Make sure DatasetImplementation folder exists at: {DATASET_PATH}")

def load_jersey(jersey_name):
    """Load processed jersey image"""
    JERSEY_DIR = Path(__file__).parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home" / "Processed"
    base_name = jersey_name.replace('.jpg', '').replace('.png', '')
    processed_name = f"{base_name}_processed.png"
    jersey_path = JERSEY_DIR / processed_name
    
    jersey = cv2.imread(str(jersey_path))
    if jersey is None:
        print(f"✗ Failed to load jersey: {jersey_path}")
        return None
    
    print(f"✓ Loaded jersey: {jersey.shape}")
    return jersey

def apply_jersey_realtime(frame, jersey_name, jersey_cache, pose_detector):
    """
    Apply jersey to a single frame in real-time (HIGHLY optimized for speed)
    
    Args:
        frame: Current video frame (numpy array)
        jersey_name: Name of jersey to apply
        jersey_cache: Dictionary to cache loaded jerseys and shirt masks
        pose_detector: Pre-initialized MediaPipe Pose detector
        
    Returns:
        Processed frame with jersey applied
    """
    try:
        # Load jersey from cache or disk
        if jersey_name not in jersey_cache:
            jersey = load_jersey(jersey_name)
            if jersey is None:
                return frame  # Return original if jersey fails to load
            jersey_cache[jersey_name] = {'jersey': jersey, 'last_mask': None}
        
        jersey = jersey_cache[jersey_name]['jersey']
        
        # Fast pose detection (using tracking mode for speed)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(rgb_frame)
        
        if not results.pose_landmarks:
            return frame  # No person detected, return original
        
        # Get shirt keypoints from pose
        h, w = frame.shape[:2]
        from detect_shirt_region import get_shirt_keypoints, create_shirt_mask
        shirt_keypoints = get_shirt_keypoints(results.pose_landmarks, w, h)
        
        if not shirt_keypoints:
            return frame
        
        # FAST: Use only bounding box from keypoints (skip skin detection entirely for speed)
        shirt_mask = create_shirt_mask(frame, shirt_keypoints)
        
        # Extract shirt region
        shirt_extracted = cv2.bitwise_and(frame, frame, mask=shirt_mask)
        
        # Fit jersey to shirt shape
        from apply_jersey import fit_jersey_to_shirt_shape
        jersey_fitted = fit_jersey_to_shirt_shape(frame, jersey, shirt_extracted)
        
        # Ensure uint8
        if jersey_fitted.dtype != np.uint8:
            jersey_fitted = jersey_fitted.astype(np.uint8)
        
        # Quick overlay with minimal blending
        gray_jersey = cv2.cvtColor(jersey_fitted, cv2.COLOR_BGR2GRAY)
        _, jersey_mask = cv2.threshold(gray_jersey, 10, 255, cv2.THRESH_BINARY)
        
        # Direct blend without Gaussian blur (much faster)
        mask_3ch = cv2.cvtColor(jersey_mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask_3ch == 255, jersey_fitted, frame).astype(np.uint8)
        
        return result
        
    except Exception as e:
        # Silently return original frame on any error (don't spam logs)
        return frame

def apply_jersey_to_image(image_path, jersey_name, output_path):
    """
    Apply jersey to captured image
    
    Args:
        image_path: Path to captured image (JPEG from Godot)
        jersey_name: Name of jersey to apply
        output_path: Where to save result
        
    Returns:
        bool: Success status
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # Convert JPEG to PNG using PIL for reliable format conversion
    if image_path.lower().endswith(('.jpg', '.jpeg')):
        print("Converting JPEG to PNG using PIL...")
        from PIL import Image as PILImage
        
        temp_png_path = image_path.replace('.jpg', '_converted.png').replace('.jpeg', '_converted.png')
        try:
            pil_img = PILImage.open(image_path)
            # Convert to RGB if needed (removes any alpha channel)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            # Save as PNG with proper format
            pil_img.save(temp_png_path, 'PNG')
            print(f"✓ Converted to PNG: {temp_png_path}")
            # Use the converted PNG for processing
            image_path = temp_png_path
        except Exception as e:
            print(f"✗ Failed to convert JPEG to PNG: {e}")
            # Try to continue with original file
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"✗ Failed to load image: {image_path}")
        return False
    
    # Verify image format
    if image.dtype != np.uint8:
        print(f"⚠ Warning: Image dtype is {image.dtype}, converting to uint8...")
        if image.max() > 255:
            image = (image / image.max() * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure 3 channels (BGR)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h}, dtype={image.dtype}, channels={image.shape[2] if len(image.shape) > 2 else 1}")
    
    # Detect pose
    print("Detecting pose...")
    results = detect_pose_landmarks(image)
    if not results.pose_landmarks:
        print("✗ No person detected")
        return False
    print("✓ Person detected")
    
    # Get shirt keypoints
    print("Extracting shirt region...")
    shirt_keypoints = get_shirt_keypoints(results.pose_landmarks, w, h)
    if not shirt_keypoints:
        print("✗ Failed to extract keypoints")
        return False
    
    # Create shirt mask
    from detect_shirt_region import create_shirt_mask
    shirt_region_mask = create_shirt_mask(image, shirt_keypoints)
    
    # Detect and remove skin
    print("Detecting skin...")
    skin_mask = detect_skin(image, mask=shirt_region_mask)
    shirt_only_mask = create_shirt_only_mask(shirt_region_mask, skin_mask)
    
    # Refine with GrabCut
    print("Refining shirt boundaries...")
    refined_mask = refine_with_grabcut(image, shirt_only_mask)
    
    # Extract shirt
    shirt_extracted = cv2.bitwise_and(image, image, mask=refined_mask)
    
    # Load jersey
    print(f"Loading jersey: {jersey_name}")
    jersey = load_jersey(jersey_name)
    if jersey is None:
        return False
    
    # Fit jersey to shirt shape
    print("Fitting jersey...")
    from apply_jersey import fit_jersey_to_shirt_shape
    jersey_fitted = fit_jersey_to_shirt_shape(image, jersey, shirt_extracted)
    
    # DEBUG: Check dtype
    print(f"DEBUG: jersey_fitted dtype={jersey_fitted.dtype}, shape={jersey_fitted.shape}")
    
    # Ensure uint8 format (np.where can change dtype!)
    if jersey_fitted.dtype != np.uint8:
        print(f"⚠ Converting from {jersey_fitted.dtype} to uint8")
        jersey_fitted = jersey_fitted.astype(np.uint8)
    
    # Overlay on original
    print("Creating final overlay...")
    gray_jersey = cv2.cvtColor(jersey_fitted, cv2.COLOR_BGR2GRAY)
    _, jersey_mask = cv2.threshold(gray_jersey, 10, 255, cv2.THRESH_BINARY)
    jersey_mask_smooth = cv2.GaussianBlur(jersey_mask, (15, 15), 0)
    
    # Alpha blend
    alpha = jersey_mask_smooth.astype(float) / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    
    result = (jersey_fitted.astype(float) * alpha + 
              image.astype(float) * (1 - alpha))
    result = result.astype(np.uint8)
    
    # Save result
    cv2.imwrite(output_path, result)
    print(f"✓ Result saved: {output_path}")
    print(f"{'='*60}\n")
    
    # Clean up temporary converted PNG file if it exists
    if '_converted.png' in image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
            print(f"✓ Cleaned up temporary file: {image_path}")
        except Exception as e:
            print(f"⚠ Could not remove temp file: {e}")
    
    return True

async def handle_commands(websocket, message):
    """Handle commands from Godot."""
    try:
        data = json.loads(message)
        cmd_type = data.get("type")
        
        if cmd_type == "change_jersey":
            jersey_name = data.get("jersey", "none")
            print(f"Received change_jersey -> '{jersey_name}'")
            return {"type": "jersey_changed", "jersey": jersey_name, "success": True}
        
        elif cmd_type == "process_image":
            image_path = data.get("image_path")
            jersey_name = data.get("jersey", "Brighton Home.jpg")
            output_path = data.get("output_path")
            
            print(f"Processing image: {image_path} with jersey: {jersey_name}")
            
            success = apply_jersey_to_image(image_path, jersey_name, output_path)
            
            return {
                "type": "processing_complete",
                "success": success,
                "output_path": output_path if success else None
            }
            
    except json.JSONDecodeError:
        print("Invalid JSON message received")
    except Exception as e:
        print(f"Error handling command: {e}")
        import traceback
        traceback.print_exc()
        return {"type": "error", "message": str(e)}
    
    return None

async def video_stream_handler(websocket):
    print("Klien Godot terhubung (streaming mode).")
    cap = cv2.VideoCapture(0)  # Angka 0 berarti menggunakan webcam default
    # Set resolusi lebih rendah for reasonable size
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    streaming = True
    realtime_jersey_enabled = True  # ALWAYS ENABLED for real-time jersey application
    current_jersey_name = "Arsenal Home.jpg"
    
    # Pre-load jersey and MediaPipe models for faster processing
    jersey_cache = {}
    from pose_detection import mp_pose
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,  # Fastest model for real-time
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("🎽 Real-time jersey mode: ENABLED")
    
    try:
        async def message_listener():
            nonlocal streaming
            async for message in websocket:
                # Handle text messages as JSON commands
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        print("Invalid JSON message received")
                        continue

                    cmd = data.get("type")
                    if cmd == "save_capture":
                        # Stop streaming and notify Godot to save
                        streaming = False
                        print("Capture requested - streaming paused")
                        response = {"type": "capture_ready", "message": "Frame ready to save"}
                        await websocket.send(json.dumps(response))
                    elif cmd == "resume":
                        # Resume streaming
                        streaming = True
                        print("Streaming resumed")
                    elif cmd == "toggle_realtime_jersey":
                        # Toggle real-time jersey application
                        realtime_jersey_enabled = not realtime_jersey_enabled
                        print(f"Real-time jersey: {'ENABLED' if realtime_jersey_enabled else 'DISABLED'}")
                        response = {"type": "realtime_jersey_status", "enabled": realtime_jersey_enabled}
                        await websocket.send(json.dumps(response))
                    elif cmd == "set_jersey":
                        # Change the jersey being used
                        current_jersey_name = data.get("jersey_name", "Arsenal Home.jpg")
                        print(f"Jersey changed to: {current_jersey_name}")
                        # Clear cache to reload
                        jersey_cache.clear()
                        response = {"type": "jersey_changed", "jersey": current_jersey_name}
                        await websocket.send(json.dumps(response))
                    else:
                        # Fallback to existing command handler
                        response = await handle_commands(websocket, message)
                        if response:
                            await websocket.send(json.dumps(response))

        # Start message listener in background
        listener_task = asyncio.create_task(message_listener())
        
        # Continuous video streaming loop
        while True:
            if streaming:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.033)  # ~30 FPS
                    continue

                frame = cv2.flip(frame, 1)
                
                # Apply real-time jersey if enabled
                if realtime_jersey_enabled:
                    try:
                        frame = apply_jersey_realtime(frame, current_jersey_name, jersey_cache, pose_detector)
                    except Exception as e:
                        # If processing fails, just show original frame
                        print(f"⚠ Real-time processing error: {e}")
                        pass

                # Compress and send JPEG bytes
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, 60]
                success, buffer = cv2.imencode('.jpg', frame, encode_param)
                if success:
                    await websocket.send(buffer.tobytes())
                
                await asyncio.sleep(0.033)  # ~30 FPS
            else:
                # When paused, just wait
                await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosed:
        print("Koneksi klien Godot ditutup.")
    finally:
        cap.release()
        print("Kamera dilepaskan.")

async def main():
    # Mulai server di komputer lokal (localhost) pada port 8765
    server = await websockets.serve(video_stream_handler, "localhost", 8765)
    print("Server Python dimulai di ws://localhost:8765. Menunggu koneksi dari Godot...")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())