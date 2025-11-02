# pose_detection.py
# Reusable module for MediaPipe pose detection

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def detect_pose_landmarks(image):
    """
    Detect pose landmarks using MediaPipe.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        MediaPipe pose results containing landmarks
    """
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = pose.process(image_rgb)
        
        return results

def get_landmark_pixel_coords(landmarks, image_width, image_height):
    """
    Convert normalized landmarks to pixel coordinates.
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        dict: Dictionary mapping landmark index to (x, y, visibility)
    """
    if not landmarks:
        return None
    
    coords = {}
    for idx, landmark in enumerate(landmarks.landmark):
        x = int(landmark.x * image_width)
        y = int(landmark.y * image_height)
        visibility = landmark.visibility
        
        coords[idx] = {
            'x': x,
            'y': y,
            'visibility': visibility
        }
    
    return coords

# Landmark indices mapping
LANDMARK_INDICES = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_pinky': 17,
    'right_pinky': 18,
    'left_index': 19,
    'right_index': 20,
    'left_thumb': 21,
    'right_thumb': 22,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_heel': 29,
    'right_heel': 30,
    'left_foot_index': 31,
    'right_foot_index': 32
}
