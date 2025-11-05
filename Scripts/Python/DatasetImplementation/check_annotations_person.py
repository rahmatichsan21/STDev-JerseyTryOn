"""
Person Detection Visualization
===============================

Visualizes the person detector model performance.
Shows:
- Original image
- Person mask (cyan) vs Background mask (black)
- Shirt detection ONLY within detected person region
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from shirt_detector_full import extract_pixel_features


def predict_person_mask(image, person_classifier):
    """
    Predict person mask using trained person detector
    """
    h, w = image.shape[:2]
    
    # Get foreground (non-black pixels)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    coords = np.where(foreground)
    
    if len(coords[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Extract features for all foreground pixels
    pixels = image[coords]
    features = np.array([extract_pixel_features(px) for px in pixels])
    
    # Predict
    predictions = person_classifier.predict(features)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[coords] = predictions.astype(np.uint8)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def predict_shirt_mask(image, shirt_classifier, person_mask):
    """
    Predict shirt mask using trained shirt detector
    Only consider pixels within person region
    """
    h, w = image.shape[:2]
    
    # Get foreground
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    # Only consider pixels that are both foreground AND person
    person_region = (person_mask > 0) & foreground
    coords = np.where(person_region)
    
    if len(coords[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Extract features for person pixels only
    pixels = image[coords]
    features = np.array([extract_pixel_features(px) for px in pixels])
    
    # Predict
    predictions = shirt_classifier.predict(features)
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[coords] = predictions.astype(np.uint8)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def visualize_person_detection(image_path, person_classifier, shirt_classifier, output_path):
    """
    Visualize person detection and shirt detection
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return None
    
    h, w = image.shape[:2]
    
    # Predict person mask
    person_mask = predict_person_mask(image, person_classifier)
    
    # Predict shirt mask (only within person region)
    shirt_mask = predict_shirt_mask(image, shirt_classifier, person_mask)
    
    # Create visualization
    vis = image.copy()
    overlay = vis.copy()
    
    # Background = BLACK (where no person detected)
    overlay[person_mask == 0] = [0, 0, 0]
    
    # Person (non-shirt) = CYAN
    overlay[(person_mask > 0) & (shirt_mask == 0)] = [255, 255, 0]
    
    # Shirt = BLUE (detected shirt within person region)
    overlay[shirt_mask > 0] = [255, 0, 0]
    
    # Blend
    vis = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    
    # Detect neck level (bottom of face/top of shirt)
    # Find skin pixels in upper half
    upper_half_person = np.zeros_like(person_mask)
    upper_half_person[:int(h * 0.5), :] = person_mask[:int(h * 0.5), :]
    
    # Detect skin using color ranges
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
    
    # Find face region (skin in upper half)
    face_skin = cv2.bitwise_and(skin_mask, upper_half_person)
    
    # Find contours of face to get bottom point
    face_contours, _ = cv2.findContours(face_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    neck_y = int(h * 0.35)  # Default neck position
    if len(face_contours) > 0:
        face_contour = max(face_contours, key=cv2.contourArea)
        if cv2.contourArea(face_contour) > 1000:
            # Get bottom of face contour
            bottommost = tuple(face_contour[face_contour[:, :, 1].argmax()][0])
            neck_y = bottommost[1] + 20  # Add offset for actual neck
    
    # Draw neck line
    cv2.line(vis, (0, neck_y), (w, neck_y), (0, 255, 255), 3)
    cv2.putText(vis, f"NECK LEVEL (y={neck_y})", (10, neck_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Draw legend
    legend_y = 30
    cv2.putText(vis, "BLACK = BACKGROUND", 
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "CYAN = PERSON (non-shirt)", 
                (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "BLUE = SHIRT", 
                (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "YELLOW = NECK LINE", 
                (10, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Count pixels
    n_background = np.sum(person_mask == 0)
    n_person = np.sum((person_mask > 0) & (shirt_mask == 0))
    n_shirt = np.sum(shirt_mask > 0)
    
    cv2.putText(vis, f"Background: {n_background} px", (10, h - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Person (non-shirt): {n_person} px", (10, h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Shirt: {n_shirt} px", (10, h), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save
    cv2.imwrite(str(output_path), vis)
    
    return {
        'n_background': n_background,
        'n_person': n_person,
        'n_shirt': n_shirt
    }


def main():
    """Visualize person detection on both datasets"""
    base_dir = Path(__file__).parent
    nobg_dir = base_dir / "IcanDataset_NOBG"
    with_bg_dir = base_dir / "IcanDataset"
    output_dir = base_dir / "person_detection_results"
    person_model_path = base_dir / "person_detector_model.joblib"
    shirt_model_path = base_dir / "shirt_detector_model_dual.joblib"
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("PERSON DETECTION VISUALIZATION")
    print("=" * 80)
    print()
    
    # Load models
    if not person_model_path.exists():
        print(f"❌ Person detector model not found: {person_model_path}")
        print("   Please run person_detector_trainer.py first!")
        return
    
    if not shirt_model_path.exists():
        print(f"❌ Shirt detector model not found: {shirt_model_path}")
        print("   Please run shirt_detector_dual_dataset.py first!")
        return
    
    print(f"Loading person detector model...")
    person_classifier = joblib.load(person_model_path)
    print(f"✓ Person detector loaded\n")
    
    print(f"Loading shirt detector model...")
    shirt_classifier = joblib.load(shirt_model_path)
    print(f"✓ Shirt detector loaded\n")
    
    # Test on NOBG images
    print("=" * 80)
    print("Testing on NO-BACKGROUND images")
    print("=" * 80)
    print()
    
    nobg_files = sorted(list(nobg_dir.glob("*.png")))[:5]
    
    print(f"Processing {len(nobg_files)} NOBG samples...\n")
    
    for idx, img_path in enumerate(nobg_files, 1):
        print(f"[{idx}/{len(nobg_files)}] {img_path.name}")
        
        output_path = output_dir / f"nobg_{idx:02d}_{img_path.stem}.png"
        
        result = visualize_person_detection(img_path, person_classifier, shirt_classifier, output_path)
        
        if result:
            print(f"  Background: {result['n_background']:8d} px")
            print(f"  Person:     {result['n_person']:8d} px")
            print(f"  Shirt:      {result['n_shirt']:8d} px")
            print(f"  ✓ Saved: {output_path.name}")
        print()
    
    # Test on WITH-BG images
    print("=" * 80)
    print("Testing on WITH-BACKGROUND images")
    print("=" * 80)
    print()
    
    with_bg_files = sorted(list(with_bg_dir.glob("*.jpg")) + list(with_bg_dir.glob("*.png")))[:5]
    
    print(f"Processing {len(with_bg_files)} WITH-BG samples...\n")
    
    for idx, img_path in enumerate(with_bg_files, 1):
        print(f"[{idx}/{len(with_bg_files)}] {img_path.name}")
        
        output_path = output_dir / f"withbg_{idx:02d}_{img_path.stem}.png"
        
        result = visualize_person_detection(img_path, person_classifier, shirt_classifier, output_path)
        
        if result:
            print(f"  Background: {result['n_background']:8d} px")
            print(f"  Person:     {result['n_person']:8d} px")
            print(f"  Shirt:      {result['n_shirt']:8d} px")
            print(f"  ✓ Saved: {output_path.name}")
        print()
    
    print("=" * 80)
    print("✓ VISUALIZATION COMPLETE!")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
