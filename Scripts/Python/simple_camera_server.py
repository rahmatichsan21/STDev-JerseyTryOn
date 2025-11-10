import asyncio
import websockets
import cv2

async def camera_stream(websocket):
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            await websocket.send(jpeg.tobytes())
            await asyncio.sleep(0.03)  # ~30 FPS
    finally:
        cap.release()

async def main():
    async with websockets.serve(camera_stream, "localhost", 9001):
        print("Camera server running on ws://localhost:9001")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())