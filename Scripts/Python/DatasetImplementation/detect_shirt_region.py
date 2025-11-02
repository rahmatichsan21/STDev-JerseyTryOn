# Detect Shirt Region using MediaPipe Pose Landmarks

import cv2
import numpy as np
from pose_detection import detect_pose_landmarks, get_landmark_pixel_coords, LANDMARK_INDICES

# Configuration
INPUT_IMAGE_PATH = r"Dataset_Raw\Dataset 10.jpg"
OUTPUT_IMAGE_PATH = r"Output\shirt_region.jpg"

# Adjustment offsets for shirt region (in pixels)
# Make it WIDE to capture full shirt width
SHIRT_ADJUSTMENTS = {
    'left_shoulder': {'x': 60, 'y': -30},   # More left - wider at shoulders
    'right_shoulder': {'x': -60, 'y': -30}, # More right - wider at shoulders
    'left_elbow': {'x': 80, 'y': 0},        # More left - wider at sides
    'right_elbow': {'x': -80, 'y': 0},      # More right - wider at sides
    'left_hip': {'x': 60, 'y': 0},        # More left - wider at bottom
    'right_hip': {'x': -60, 'y': 0},      # More right - wider at bottom
}

def get_shirt_keypoints(landmarks, image_width, image_height):
    """
    Extract only the keypoints relevant to shirt region with adjustments.
    Returns a dictionary with shoulder, elbow, and hip positions.
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        
    Returns:
        dict: Dictionary containing shirt-relevant keypoints (adjusted)
    """
    if not landmarks:
        return None
    
    # Get all landmark coordinates
    all_coords = get_landmark_pixel_coords(landmarks, image_width, image_height)
    
    if not all_coords:
        return None
    
    # Shirt-relevant landmark names
    SHIRT_LANDMARK_NAMES = [
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_hip', 'right_hip'
    ]
    
    shirt_keypoints = {}
    
    for name in SHIRT_LANDMARK_NAMES:
        idx = LANDMARK_INDICES[name]
        point = all_coords[idx]
        
        # Apply adjustments to expand shirt region
        adjustment = SHIRT_ADJUSTMENTS.get(name, {'x': 0, 'y': 0})
        adjusted_x = point['x'] + adjustment['x']
        adjusted_y = point['y'] + adjustment['y']
        
        # Ensure points stay within image bounds
        adjusted_x = max(0, min(image_width - 1, adjusted_x))
        adjusted_y = max(0, min(image_height - 1, adjusted_y))
        
        shirt_keypoints[name] = {
            'x': adjusted_x,
            'y': adjusted_y,
            'visibility': point['visibility'],
            'original_x': point['x'],
            'original_y': point['y']
        }
    
    return shirt_keypoints

def draw_shirt_keypoints(image, shirt_keypoints, show_original=True):
    """
    Draw only the shirt-relevant keypoints on the image.
    
    Args:
        image: Input image
        shirt_keypoints: Dictionary of shirt keypoints
        show_original: If True, also show original detected points
        
    Returns:
        Image with shirt keypoints drawn
    """
    output = image.copy()
    
    if not shirt_keypoints:
        return output
    
    # Draw original keypoints if requested (smaller, semi-transparent)
    if show_original:
        for name, point in shirt_keypoints.items():
            orig_x = point.get('original_x')
            orig_y = point.get('original_y')
            if orig_x is not None and orig_y is not None:
                cv2.circle(output, (orig_x, orig_y), 4, (128, 128, 128), -1)
                # Draw line from original to adjusted
                cv2.line(output, (orig_x, orig_y), (point['x'], point['y']), (200, 200, 200), 1)
    
    # Define connections for shirt region
    connections = [
        ('left_shoulder', 'right_shoulder'),   # Shoulder line
        ('left_shoulder', 'left_elbow'),       # Left upper arm
        ('right_shoulder', 'right_elbow'),     # Right upper arm
        ('left_shoulder', 'left_hip'),         # Left torso
        ('right_shoulder', 'right_hip'),       # Right torso
        ('left_hip', 'right_hip'),             # Hip line
    ]
    
    # Draw connections (shirt outline)
    for start_name, end_name in connections:
        if start_name in shirt_keypoints and end_name in shirt_keypoints:
            start_point = (shirt_keypoints[start_name]['x'], shirt_keypoints[start_name]['y'])
            end_point = (shirt_keypoints[end_name]['x'], shirt_keypoints[end_name]['y'])
            cv2.line(output, start_point, end_point, (0, 255, 255), 3)
    
    # Draw adjusted keypoints
    for name, point in shirt_keypoints.items():
        x, y = point['x'], point['y']
        
        # Color code by body part
        if 'shoulder' in name:
            color = (0, 0, 255)  # Red for shoulders
            radius = 10
        elif 'elbow' in name:
            color = (255, 0, 0)  # Blue for elbows
            radius = 8
        elif 'hip' in name:
            color = (0, 255, 0)  # Green for hips
            radius = 10
        else:
            color = (255, 255, 255)
            radius = 6
        
        # Draw circle
        cv2.circle(output, (x, y), radius, color, -1)
        cv2.circle(output, (x, y), radius + 1, (255, 255, 255), 2)
        
        # Draw label
        cv2.putText(output, name, (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output

def create_shirt_mask(image, shirt_keypoints):
    """
    Create a binary mask for the shirt region based on keypoints.
    
    Args:
        image: Input image
        shirt_keypoints: Dictionary of shirt keypoints
        
    Returns:
        Binary mask of shirt region
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not shirt_keypoints:
        return mask
    
    # Define shirt polygon points in proper order to capture FULL shirt
    # Go around the outer edge: shoulders (top) → elbows (sides) → hips (bottom)
    # This creates a shape that follows the shirt outline
    polygon_points = [
        (shirt_keypoints['left_shoulder']['x'], shirt_keypoints['left_shoulder']['y']),    # Top left
        (shirt_keypoints['right_shoulder']['x'], shirt_keypoints['right_shoulder']['y']),  # Top right
        (shirt_keypoints['right_elbow']['x'], shirt_keypoints['right_elbow']['y']),        # Right side (sleeve)
        (shirt_keypoints['right_hip']['x'], shirt_keypoints['right_hip']['y']),            # Bottom right
        (shirt_keypoints['left_hip']['x'], shirt_keypoints['left_hip']['y']),              # Bottom left
        (shirt_keypoints['left_elbow']['x'], shirt_keypoints['left_elbow']['y']),          # Left side (sleeve)
    ]
    
    # Convert to numpy array
    polygon = np.array(polygon_points, dtype=np.int32)
    
    # Fill polygon
    cv2.fillPoly(mask, [polygon], 255)
    
    return mask

def main():
    """Main execution function"""
    print("Loading image...")
    image = cv2.imread(INPUT_IMAGE_PATH)
    
    if image is None:
        print(f"Error: Cannot load {INPUT_IMAGE_PATH}")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    print("\nDetecting pose landmarks...")
    results = detect_pose_landmarks(image)
    
    if not results.pose_landmarks:
        print("✗ No person detected in the image")
        return
    
    print("✓ Person detected")
    
    # Get shirt-relevant keypoints
    print("\nExtracting shirt keypoints...")
    shirt_keypoints = get_shirt_keypoints(results.pose_landmarks, w, h)
    
    if shirt_keypoints:
        print(f"✓ Extracted {len(shirt_keypoints)} shirt-relevant keypoints\n")
        
        # Print keypoint positions
        print("Shirt keypoint positions (adjusted):")
        for name, point in shirt_keypoints.items():
            orig_x = point.get('original_x', 'N/A')
            orig_y = point.get('original_y', 'N/A')
            print(f"  {name}:")
            print(f"    Original: ({orig_x}, {orig_y})")
            print(f"    Adjusted: ({point['x']}, {point['y']}) - visibility: {point['visibility']:.2f}")
        
        # Draw shirt keypoints
        output = draw_shirt_keypoints(image, shirt_keypoints)
        
        # Create shirt mask
        mask = create_shirt_mask(image, shirt_keypoints)
        
        # Apply mask to show shirt region
        shirt_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Create colored mask overlay
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 0] = 0  # Remove blue channel
        mask_colored[:, :, 2] = 0  # Remove red channel
        overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        # Show results
        cv2.namedWindow("Shirt Keypoints", cv2.WINDOW_NORMAL)
        cv2.imshow("Shirt Keypoints", output)
        
        cv2.namedWindow("Shirt Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Shirt Mask", mask)
        
        cv2.namedWindow("Shirt Region", cv2.WINDOW_NORMAL)
        cv2.imshow("Shirt Region", shirt_region)
        
        cv2.namedWindow("Shirt Overlay", cv2.WINDOW_NORMAL)
        cv2.imshow("Shirt Overlay", overlay)
        
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("✗ Could not extract shirt keypoints")

if __name__ == "__main__":
    main()
