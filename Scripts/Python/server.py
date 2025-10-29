import asyncio
import websockets
import cv2
import json

async def handle_commands(websocket, message):
    """Handle commands from Godot."""
    try:
        data = json.loads(message)
        if data.get("type") == "change_jersey":
            jersey_name = data.get("jersey", "none")
            print(f"Received change_jersey -> '{jersey_name}'")
            return {"type": "jersey_changed", "jersey": jersey_name, "success": True}
    except json.JSONDecodeError:
        print("Invalid JSON message received")
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