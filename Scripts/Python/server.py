import asyncio
import websockets
import cv2
import json
from jersey_filter import JerseyOverlay

# Create jersey overlay instance
jersey_overlay = JerseyOverlay()

async def handle_commands(websocket, message):
    """Handle commands from Godot"""
    try:
        data = json.loads(message)
        if data.get("type") == "change_jersey":
            jersey_name = data.get("jersey", "none")
            success = jersey_overlay.set_jersey(jersey_name)
            print(f"Jersey changed to: {jersey_name}")
            return {"type": "jersey_changed", "jersey": jersey_name, "success": success}
    except json.JSONDecodeError:
        print("Invalid JSON message received")
    return None

async def video_stream_handler(websocket):
    print("Klien Godot terhubung.")
    cap = cv2.VideoCapture(0) # Angka 0 berarti menggunakan webcam default
    # Set resolusi lebih rendah untuk FPS lebih baik
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    except Exception:
        pass

    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    try:
        # Create a task to listen for incoming messages
        async def message_listener():
            async for message in websocket:
                if isinstance(message, str):  # Text message (commands)
                    response = await handle_commands(websocket, message)
                    if response:
                        await websocket.send(json.dumps(response))
        
        # Start message listener in background
        listener_task = asyncio.create_task(message_listener())
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror the frame horizontally (flip kiri-kanan)
            frame = cv2.flip(frame, 1)
            
            # Apply jersey overlay
            frame = jersey_overlay.apply_jersey_overlay(frame)

            # Kompres gambar ke format JPEG untuk pengiriman yang lebih cepat
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 60]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)

            # Kirim data gambar melalui WebSocket
            await websocket.send(buffer.tobytes())

            # Beri jeda singkat untuk menjaga frame rate tetap stabil (~30 FPS)
            await asyncio.sleep(0.03)

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