"""
Annotation Visualization with Background Removal
=================================================

This script visualizes annotations for images WITH backgrounds (IcanDataset).
It removes the background first, then shows the annotations.

Shows:
- Detected neck level (yellow line)
- Skin regions (green)
- Hair regions (red)
- Shirt regions (blue)
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from shirt_detector_full import detect_shirt_region_automatic


def remove_background_simple(image):
    """
    Simple background removal using color threshold
    Works for images with solid/light backgrounds
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for non-background pixels (not white/light gray/black)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert to get foreground
    foreground = cv2.bitwise_not(white_mask)
    
    # Clean up with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    
    return foreground


def visualize_annotations_with_bg(image_path, output_path):
    """
    Visualize annotations for images WITH background
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return None
    
    h, w = image.shape[:2]
    
    # Remove background
    print(f"  Removing background...")
    foreground = remove_background_simple(image)
    
    # Create image with transparent background (only foreground)
    image_no_bg = image.copy()
    image_no_bg[foreground == 0] = 255  # Make background white for visualization
    
    # Get automatic annotations on the no-background image
    print(f"  Detecting annotations...")
    shirt_mask, skin_mask, hair_mask = detect_shirt_region_automatic(image_no_bg)
    
    # Create visualization
    vis = image.copy()
    
    # Overlay masks with transparency
    overlay = vis.copy()
    
    # Background = BLACK (removed areas)
    overlay[foreground == 0] = [0, 0, 0]
    
    # Hair = RED
    overlay[hair_mask > 0] = [0, 0, 255]
    
    # Skin = GREEN
    overlay[skin_mask > 0] = [0, 255, 0]
    
    # Shirt = BLUE
    overlay[shirt_mask > 0] = [255, 0, 0]
    
    # Blend
    vis = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    # Find neck level for visualization
    upper_half = np.zeros_like(skin_mask)
    upper_half[:int(h * 0.5), :] = 255
    face_skin = cv2.bitwise_and(skin_mask, upper_half)
    
    face_contours, _ = cv2.findContours(face_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    neck_y = int(h * 0.35)  # Default
    if len(face_contours) > 0:
        face_contour = max(face_contours, key=cv2.contourArea)
        if cv2.contourArea(face_contour) > 1000:
            bottommost = tuple(face_contour[face_contour[:, :, 1].argmax()][0])
            neck_y = bottommost[1] + 20
    
    # Draw neck level line
    cv2.line(vis, (0, neck_y), (w, neck_y), (0, 255, 255), 3)
    cv2.putText(vis, f"NECK LEVEL (y={neck_y})", (10, neck_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw legend
    legend_y = 30
    cv2.putText(vis, "BLACK = BACKGROUND", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(vis, "RED = HAIR", (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "GREEN = SKIN", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, "BLUE = SHIRT", (10, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(vis, "YELLOW = NECK LINE", (10, legend_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Count pixels
    n_hair = np.sum(hair_mask > 0)
    n_skin = np.sum(skin_mask > 0)
    n_shirt = np.sum(shirt_mask > 0)
    n_bg = np.sum(foreground == 0)
    
    cv2.putText(vis, f"Background: {n_bg} px", (10, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Hair: {n_hair} px", (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Skin: {n_skin} px", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Shirt: {n_shirt} px", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), vis)
    
    return {
        'neck_y': neck_y,
        'n_hair': n_hair,
        'n_skin': n_skin,
        'n_shirt': n_shirt
    }


def main():
    """Visualize annotations for images WITH background"""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "IcanDataset"  # WITH background
    output_dir = base_dir / "annotation_check_with_bg"
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ANNOTATION VERIFICATION - Images WITH Background")
    print("=" * 80)
    print()
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_dir}")
        return
    
    # Get sample images (first 10)
    image_files = sorted(list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png")))[:10]
    
    if len(image_files) == 0:
        print(f"❌ No images found in {dataset_dir}")
        return
    
    print(f"✓ Found {len(image_files)} images")
    print(f"✓ Processing first 10 samples...")
    print()
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        
        output_path = output_dir / f"annotated_with_bg_{img_path.stem}.png"
        
        result = visualize_annotations_with_bg(img_path, output_path)
        
        if result:
            print(f"  ✓ Neck level: y={result['neck_y']}")
            print(f"  ✓ Hair pixels: {result['n_hair']}")
            print(f"  ✓ Skin pixels: {result['n_skin']}")
            print(f"  ✓ Shirt pixels: {result['n_shirt']}")
            print(f"  ✓ Saved: {output_path.name}")
        print()
    
    print("=" * 80)
    print("✓ VERIFICATION COMPLETE!")
    print(f"✓ Check the images in: {output_dir}")
    print()
    print("Legend:")
    print("  RED = Hair (should be ABOVE neck line)")
    print("  GREEN = Skin (face, neck, arms)")
    print("  BLUE = Shirt (should be BELOW neck line)")
    print("  YELLOW LINE = Detected neck level")
    print("=" * 80)


if __name__ == "__main__":
    main()
