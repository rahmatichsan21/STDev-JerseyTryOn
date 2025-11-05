"""
Annotation Visualization - Dual Dataset Model
==============================================

This script shows annotations using the DUAL-DATASET trained model.
Visualizes performance on BOTH NOBG and WITH-BG images.

Shows:
- Detected neck level (yellow line)
- Skin regions (green)
- Hair regions (red)
- Shirt regions (blue) - detected by dual model
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from shirt_detector_full import extract_pixel_features


def remove_background_simple(image):
    """Simple background removal using color threshold"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    foreground = cv2.bitwise_not(white_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    
    return foreground


def predict_shirt_mask_dual(image, classifier):
    """
    Predict shirt mask using dual-dataset trained model
    """
    h, w = image.shape[:2]
    
    # Get foreground
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    coords = np.where(foreground)
    
    if len(coords[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Extract features for all foreground pixels
    pixels = image[coords]
    features = np.array([extract_pixel_features(px) for px in pixels])
    
    # Predict
    predictions = classifier.predict(features)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[coords] = predictions.astype(np.uint8)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def visualize_annotations_mixed(image_path, classifier, output_path, is_nobg=False):
    """
    Visualize annotations using dual-dataset model
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return None
    
    h, w = image.shape[:2]
    
    # If WITH-BG, remove background first
    if not is_nobg:
        foreground = remove_background_simple(image)
        image_processed = image.copy()
        image_processed[foreground == 0] = 255  # Make background white
    else:
        image_processed = image.copy()
    
    # Predict shirt mask using dual model
    shirt_mask = predict_shirt_mask_dual(image_processed, classifier)
    
    # Get foreground for visualization
    gray = cv2.cvtColor(image_processed, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    # Detect skin (simplified for visualization)
    ycrcb = cv2.cvtColor(image_processed, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    hsv = cv2.cvtColor(image_processed, cv2.COLOR_BGR2HSV)
    skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
    
    # Hair detection
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    upper_half = np.zeros_like(gray_blurred)
    upper_half[:int(h * 0.4), :] = 1
    hair_mask = ((gray_blurred < 80) & (upper_half > 0)).astype(np.uint8) * 255
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    # Create visualization
    vis = image.copy()
    overlay = vis.copy()
    
    # Background = BLACK (default where nothing else is)
    overlay[foreground == 0] = [0, 0, 0]
    
    # Hair = RED
    overlay[hair_mask > 0] = [0, 0, 255]
    
    # Skin = GREEN
    overlay[skin_mask > 0] = [0, 255, 0]
    
    # Shirt = BLUE (from dual model)
    overlay[shirt_mask > 0] = [255, 0, 0]
    
    # Blend
    vis = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    
    # Find neck level
    upper_half_skin = np.zeros_like(skin_mask)
    upper_half_skin[:int(h * 0.5), :] = 255
    face_skin = cv2.bitwise_and(skin_mask, upper_half_skin)
    
    face_contours, _ = cv2.findContours(face_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    neck_y = int(h * 0.35)
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
    cv2.putText(vis, "BLUE = SHIRT (DUAL MODEL)", (10, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
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
        'n_bg': n_bg,
        'n_hair': n_hair,
        'n_skin': n_skin,
        'n_shirt': n_shirt
    }


def main():
    """Visualize annotations with dual-dataset model on BOTH datasets"""
    base_dir = Path(__file__).parent
    nobg_dir = base_dir / "IcanDataset_NOBG"
    with_bg_dir = base_dir / "IcanDataset"
    output_dir = base_dir / "annotation_check_mixed"
    model_path = base_dir / "shirt_detector_model_dual.joblib"
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ANNOTATION VERIFICATION - Dual Dataset Model")
    print("=" * 80)
    print()
    
    # Load dual model
    if not model_path.exists():
        print(f"❌ Dual model not found: {model_path}")
        print("   Please run shirt_detector_dual_dataset.py first!")
        return
    
    print(f"Loading dual-dataset model...")
    classifier = joblib.load(model_path)
    print(f"✓ Model loaded\n")
    
    # Test on NOBG images
    print("=" * 80)
    print("Testing on NO-BACKGROUND images (original dataset)")
    print("=" * 80)
    
    nobg_files = sorted(list(nobg_dir.glob("*.png")))[:5]
    
    print(f"\nProcessing {len(nobg_files)} NOBG samples...\n")
    
    for idx, img_path in enumerate(nobg_files, 1):
        print(f"[{idx}/{len(nobg_files)}] {img_path.name}")
        
        output_path = output_dir / f"nobg_{idx:02d}_{img_path.stem}.png"
        
        result = visualize_annotations_mixed(img_path, classifier, output_path, is_nobg=True)
        
        if result:
            print(f"  ✓ Shirt: {result['n_shirt']} px")
            print(f"  ✓ Saved: {output_path.name}")
        print()
    
    # Test on WITH-BG images
    print("=" * 80)
    print("Testing on WITH-BACKGROUND images (context dataset)")
    print("=" * 80)
    
    with_bg_files = sorted(list(with_bg_dir.glob("*.jpg")) + list(with_bg_dir.glob("*.png")))[:5]
    
    print(f"\nProcessing {len(with_bg_files)} WITH-BG samples...\n")
    
    for idx, img_path in enumerate(with_bg_files, 1):
        print(f"[{idx}/{len(with_bg_files)}] {img_path.name}")
        
        output_path = output_dir / f"withbg_{idx:02d}_{img_path.stem}.png"
        
        result = visualize_annotations_mixed(img_path, classifier, output_path, is_nobg=False)
        
        if result:
            print(f"  ✓ Shirt: {result['n_shirt']} px")
            print(f"  ✓ Saved: {output_path.name}")
        print()
    
    print("=" * 80)
    print("✓ VERIFICATION COMPLETE!")
    print(f"✓ Results saved to: {output_dir}")
    print()
    print("Comparison:")
    print("  - NOBG images: Show clean shirt detection in isolation")
    print("  - WITH-BG images: Show shirt detection with context/background")
    print()
    print("The dual model should:")
    print("  ✓ Correctly identify ONLY the shirt (blue)")
    print("  ✓ NOT misclassify arms as shirt")
    print("  ✓ NOT include background artifacts")
    print("=" * 80)


if __name__ == "__main__":
    main()
