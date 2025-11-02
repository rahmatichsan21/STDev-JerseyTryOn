"""
Simple jersey replacement - directly map jersey texture to shirt region
"""

import cv2
import numpy as np
from pathlib import Path
from pose_detection import detect_pose_landmarks
from detect_shirt_region import get_shirt_keypoints

# Paths
JERSEY_DIR = Path(__file__).parent.parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home" / "Processed"
INPUT_IMAGE = "Dataset_NoBG/Dataset_1-removebg-preview.png"  # Original image for pose detection
OUTPUT_DIR = "Output"

def load_jersey(jersey_name):
    """Load processed jersey (already has background removed)"""
    # Remove extension and add _processed.png
    base_name = jersey_name.replace('.jpg', '').replace('.png', '')
    processed_name = f"{base_name}_processed.png"
    jersey_path = JERSEY_DIR / processed_name
    
    jersey = cv2.imread(str(jersey_path))
    
    if jersey is None:
        print(f"✗ Failed to load: {jersey_path}")
        return None
    
    print(f"✓ Loaded processed jersey: {jersey.shape}")
    return jersey

def fit_jersey_to_shirt_shape(image, jersey, shirt_image):
    """
    Fit jersey to EXACTLY match the extracted shirt shape
    
    Args:
        image: Original person image (for canvas size)
        jersey: Jersey image with background removed
        shirt_image: The extracted shirt image (shows exact shape we want)
        
    Returns:
        Jersey fitted to exact shirt shape
    """
    h, w = image.shape[:2]
    print(f"\nImage canvas: {w}x{h}")
    
    # Get shirt shape from the extracted shirt image
    gray_shirt = cv2.cvtColor(shirt_image, cv2.COLOR_BGR2GRAY)
    _, shirt_mask = cv2.threshold(gray_shirt, 10, 255, cv2.THRESH_BINARY)
    
    # Find shirt contours
    contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("✗ No shirt contour found")
        return np.zeros_like(image)
    
    # Get the largest contour (the shirt)
    shirt_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x_s, y_s, w_s, h_s = cv2.boundingRect(shirt_contour)
    
    print(f"Shirt shape: {w_s}x{h_s} at position ({x_s}, {y_s})")
    
    # Resize jersey to match shirt dimensions
    jersey_resized = cv2.resize(jersey, (w_s, h_s), interpolation=cv2.INTER_LANCZOS4)
    
    # Create output canvas
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Place resized jersey at shirt position
    result[y_s:y_s+h_s, x_s:x_s+w_s] = jersey_resized
    
    # Apply shirt mask to match EXACT shape (not just bounding box)
    mask_3ch = cv2.cvtColor(shirt_mask, cv2.COLOR_GRAY2BGR)
    result = np.where(mask_3ch == 255, result, [0, 0, 0])
    
    jersey_pixels = np.count_nonzero(result.sum(axis=2))
    print(f"✓ Jersey fitted to exact shirt shape: {jersey_pixels} pixels")
    
    return result

def apply_jersey(image, jersey, shirt_image):
    """
    Apply jersey to exactly match the extracted shirt shape
    
    Args:
        image: Original person image
        jersey: Jersey image
        shirt_image: The extracted shirt image showing exact shape
        
    Returns:
        Jersey fitted to shirt shape
    """
    # Fit jersey to the EXACT extracted shirt shape
    result = fit_jersey_to_shirt_shape(image, jersey, shirt_image)
    
    return result

def main():
    print("=" * 60)
    print("JERSEY REPLACEMENT - BODY-FITTED MODE")
    print("=" * 60)
    
    # Load original image
    print(f"\n[1/5] Loading image: {INPUT_IMAGE}")
    image = cv2.imread(INPUT_IMAGE)
    
    if image is None:
        print(f"✗ FAILED TO LOAD IMAGE: {INPUT_IMAGE}")
        return
    
    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h}")
    
    # Detect pose landmarks
    print(f"\n[2/5] Detecting pose landmarks...")
    results = detect_pose_landmarks(image)
    
    if not results.pose_landmarks:
        print("✗ No person detected in image")
        return
    
    print("✓ Person detected")
    
    # Get shirt keypoints
    print(f"\n[3/5] Extracting shirt keypoints...")
    shirt_keypoints = get_shirt_keypoints(results.pose_landmarks, w, h)
    
    if not shirt_keypoints:
        print("✗ Failed to extract shirt keypoints")
        return
    
    print("✓ Shirt keypoints extracted")
    for name, coords in shirt_keypoints.items():
        print(f"  {name}: ({coords['x']}, {coords['y']})")
    
    # Load extracted shirt image (the EXACT shape we want to match)
    print(f"\n[4/5] Loading extracted shirt shape...")
    shirt_extracted_path = f"{OUTPUT_DIR}/7_final_extracted.png"
    shirt_image = cv2.imread(shirt_extracted_path)
    
    if shirt_image is None:
        # Try alternative path
        shirt_extracted_path = f"{OUTPUT_DIR}/6_final_extracted.png"
        shirt_image = cv2.imread(shirt_extracted_path)
    
    if shirt_image is None:
        print(f"✗ Extracted shirt image not found!")
        print(f"   Please run detect_actual_shirt.py first")
        return
    
    print(f"✓ Loaded extracted shirt: {shirt_image.shape}")
    
    # Load Arsenal jersey
    print(f"\n[5/5] Loading and applying jersey...")
    jersey = load_jersey("Arsenal Home.jpg")
    if jersey is None:
        return
    
    print(f"  Jersey loaded: {jersey.shape}")
    
    # Apply jersey to match exact shirt shape
    result = apply_jersey(image, jersey, shirt_image)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    # Save result
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    output_path = f"{OUTPUT_DIR}/jersey_result.png"
    cv2.imwrite(output_path, result)
    print(f"\n✓ Saved: {output_path}")
    
    # Create visualization with keypoints
    debug_image = image.copy()
    for name, coords in shirt_keypoints.items():
        x, y = coords['x'], coords['y']
        cv2.circle(debug_image, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(debug_image, name[:8], (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imwrite(f"{OUTPUT_DIR}/debug_keypoints.png", debug_image)
    print(f"✓ Saved keypoints debug: {OUTPUT_DIR}/debug_keypoints.png")
    
    # Save comparison
    comparison = np.hstack([image, result])
    cv2.imwrite(f"{OUTPUT_DIR}/comparison.png", comparison)
    print(f"✓ Saved comparison: {OUTPUT_DIR}/comparison.png")
    
    print(f"\n{'=' * 60}")
    print("Run overlay_jersey.py next to combine with original image!")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
