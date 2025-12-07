import cv2

class CameraManager:
    """
    Kelas untuk mengelola pengambilan frame dari webcam.
    """
    def __init__(self, device_id=0, quality=80):
        self.device_id = device_id
        self.cap = None
        self.quality = quality
        self.jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        self.start()

    def start(self):
        """Mulai mengambil gambar dari kamera."""
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                print(f"Error: Tidak dapat membuka kamera device {self.device_id}")
                self.cap = None
        except Exception as e:
            print(f"Exception saat membuka kamera: {e}")
            self.cap = None

    def stop(self):
        """Melepaskan kamera."""
        if self.cap:
            self.cap.release()
            self.cap = None
            print("Kamera dilepaskan.")

    def get_jpeg_frame(self):
        """Mengambil satu frame, meng-encode-nya ke JPG, dan mengembalikannya sebagai bytes."""
        if not self.cap or not self.cap.isOpened():
            print("Mencoba menghubungkan kembali kamera...")
            self.start()
            if not self.cap:
                return None # Gagal terhubung

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Gagal mengambil frame.")
            return None
        
        # Encode ke JPG
        ret, jpeg = cv2.imencode('.jpg', frame, self.jpeg_params)
        if not ret:
            print("Error: Gagal encode frame ke JPG.")
            return None
            
        return jpeg.tobytes()