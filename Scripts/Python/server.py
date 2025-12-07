import asyncio
import websockets
import json
import cv2
import os
from camera_manager import CameraManager
from jersey_processor import process_video_frame # Import fungsi baru

# --- Konfigurasi ---
HOST = "localhost"
PORT = 9001
JERSEY_ASSET_PATH = "../../Assets/Jerseys/PremierLeague/Home_NOBG/"
DEFAULT_JERSEY = "Arsenal Home.png" # Ganti default jersey di sini

# --- Global ---
current_jersey_img = None
camera = CameraManager(device_id=0)

def load_jersey(jersey_name):
    global current_jersey_img
    path = os.path.join(JERSEY_ASSET_PATH, jersey_name)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Gagal memuat jersey: {jersey_name}")
    else:
        print(f"Jersey dimuat: {jersey_name}")
        current_jersey_img = img

async def godot_handler(websocket):
    print(f"Godot Terhubung!")
    
    # Load jersey awal
    if current_jersey_img is None:
        load_jersey(DEFAULT_JERSEY)

    try:
        while True:
            # 1. Cek jika ada perintah ganti jersey dari Godot (Non-blocking)
            try:
                # Timeout sangat kecil agar tidak menghambat streaming
                message = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                data = json.loads(message)
                
                if data.get("type") == "change_jersey":
                    new_jersey = data.get("jersey_name")
                    load_jersey(new_jersey)
                    
            except asyncio.TimeoutError:
                pass # Tidak ada perintah, lanjut streaming
            except Exception:
                pass

            # 2. Ambil Frame Kamera (Raw)
            # Kita akses properti internal camera_manager agar dapat raw frame
            # karena kita perlu memprosesnya SEBELUM di-encode ke JPG
            if camera.cap and camera.cap.isOpened():
                ret, frame = camera.cap.read()
                if ret:
                    # 3. PROSES AI (REAL TIME FILTER) DI SINI
                    if current_jersey_img is not None:
                        # Frame diedit langsung
                        final_frame = process_video_frame(frame, current_jersey_img)
                    else:
                        final_frame = frame

                    # 4. Encode ke JPG untuk dikirim
                    _, jpeg = cv2.imencode('.jpg', final_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]) # Quality 70 biar ngebut
                    
                    # 5. Kirim ke Godot
                    await websocket.send(jpeg.tobytes())
                
                # Kontrol FPS (0 = secepat mungkin, 0.01 = ~100fps max)
                await asyncio.sleep(0) 
            else:
                await asyncio.sleep(0.1)

    except websockets.exceptions.ConnectionClosed:
        print("Godot terputus.")
    finally:
        pass

async def main():
    print(f"Server Real-Time berjalan di ws://{HOST}:{PORT}")
    # Load jersey pertama kali
    load_jersey(DEFAULT_JERSEY)
    async with websockets.serve(godot_handler, HOST, PORT, max_size=1_000_000 * 20):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        camera.stop()