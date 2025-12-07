import cv2
import mediapipe as mp
import numpy as np

# OPTIMISASI 1: Gunakan model_complexity=0 (Paling Cepat)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False, # False karena ini video stream
    model_complexity=0,      # 0 = Cepat, 1 = Standar, 2 = Akurat
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_video_frame(user_frame, jersey_img):
    """
    Memproses satu frame video secara langsung di memori.
    """
    try:
        # OPTIMISASI 2: Resize frame jika terlalu besar (opsional, misal ke 640x480)
        # h, w = user_frame.shape[:2]
        # if w > 640:
        #     user_frame = cv2.resize(user_frame, (640, 480))
        
        h, w, _ = user_frame.shape
        
        # Jika jersey belum dimuat atau invalid
        if jersey_img is None:
            return user_frame

        jh, jw, _ = jersey_img.shape

        # 1. Deteksi Pose
        # Konversi ke RGB untuk MediaPipe (pass by reference agar cepat)
        user_img_rgb = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
        user_img_rgb.flags.writeable = False
        results = pose.process(user_img_rgb)
        user_img_rgb.flags.writeable = True

        if not results.pose_landmarks:
            # Jika tidak ada orang, kembalikan frame asli
            return user_frame

        landmarks = results.pose_landmarks.landmark

        # 2. Ambil Koordinat Kunci
        # Menggunakan visibilitas untuk memastikan titik terdeteksi dengan baik
        pts = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        ]
        
        # Cek confidence score (opsional, agar tidak glitchy)
        if any(p.visibility < 0.5 for p in pts):
            return user_frame

        # Konversi ke piksel
        left_shoulder = (int(pts[0].x * w), int(pts[0].y * h))
        right_shoulder = (int(pts[1].x * w), int(pts[1].y * h))
        right_hip = (int(pts[2].x * w), int(pts[2].y * h))
        left_hip = (int(pts[3].x * w), int(pts[3].y * h))

        # 3. Warping (Inti Filter)
        src_pts = np.float32([[0, 0], [jw, 0], [jw, jh], [0, jh]])
        dst_pts = np.float32([left_shoulder, right_shoulder, right_hip, left_hip])
        
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_jersey = cv2.warpPerspective(jersey_img, matrix, (w, h))

        # 4. Overlay Cepat
        # Buat mask
        alpha_channel = warped_jersey[:, :, 3] / 255.0
        
        # Optimisasi Overlay menggunakan NumPy Broadcasting
        # Area user background
        user_frame = user_frame.astype(float)
        
        # Inverse alpha
        inverse_alpha = 1.0 - alpha_channel
        
        # Kombinasi (Blending)
        for c in range(0, 3):
            user_frame[:, :, c] = (alpha_channel * warped_jersey[:, :, c] + 
                                   inverse_alpha * user_frame[:, :, c])

        return user_frame.astype(np.uint8)

    except Exception as e:
        # Jika error (misal matriks singular), kembalikan frame asli agar tidak crash
        # print(f"Frame error: {e}")
        return user_frame