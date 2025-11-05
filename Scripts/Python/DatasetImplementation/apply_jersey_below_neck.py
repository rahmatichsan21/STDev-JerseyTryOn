"""
Jersey Application - Below Neck Level
======================================

Applies Brighton Home jersey to detected shirt pixels ONLY below neck level.
Processes all images in IcanDataset folder and saves results to OUTPUT_JERSEY.

Process:
1. Detect person region
2. Detect neck level
3. Detect shirt region
4. Filter shirt to only pixels BELOW neck level
5. Apply jersey overlay
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from shirt_detector_full import extract_pixel_features


def predict_person_mask(image, person_classifier):
    """Predict person mask using trained person detector"""
    if person_classifier is None:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (gray > 20).astype(np.uint8)
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    coords = np.where(foreground)
    
    if len(coords[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    pixels = image[coords]
    features = np.array([extract_pixel_features(px) for px in pixels])
    predictions = person_classifier.predict(features)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[coords] = predictions.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def predict_shirt_mask(image, shirt_classifier, person_mask):
    """Predict shirt mask using trained shirt detector"""
    if shirt_classifier is None:
        return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    if person_mask is not None:
        region_of_interest = foreground & (person_mask > 0)
    else:
        region_of_interest = foreground
    
    coords = np.where(region_of_interest)
    
    if len(coords[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    pixels = image[coords]
    features = np.array([extract_pixel_features(px) for px in pixels])
    predictions = shirt_classifier.predict(features)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[coords] = predictions.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def detect_neck_level(image, person_mask):
    """Detect neck level from face region"""
    h, w = image.shape[:2]
    
    upper_half_person = np.zeros_like(person_mask)
    upper_half_person[:int(h * 0.5), :] = person_mask[:int(h * 0.5), :]
    
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
    
    face_skin = cv2.bitwise_and(skin_mask, upper_half_person)
    face_contours, _ = cv2.findContours(face_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    neck_y = int(h * 0.35)  # Default
    if len(face_contours) > 0:
        face_contour = max(face_contours, key=cv2.contourArea)
        if cv2.contourArea(face_contour) > 1000:
            bottommost = tuple(face_contour[face_contour[:, :, 1].argmax()][0])
            neck_y = bottommost[1] + 20
    
    return neck_y


def apply_jersey_to_image(image_path, person_classifier, shirt_classifier, jersey, output_path):
    """
    Apply jersey to shirt region ONLY BELOW NECK LEVEL
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return False
    
    h, w = image.shape[:2]
    result = image.copy()
    
    try:
        # Step 1: Detect person
        person_mask = predict_person_mask(image, person_classifier)
        
        # Step 2: Detect neck level
        neck_y = detect_neck_level(image, person_mask)
        
        # Step 3: Detect shirt
        shirt_mask = predict_shirt_mask(image, shirt_classifier, person_mask)
        
        # Step 4: Filter shirt to ONLY pixels BELOW neck level
        shirt_mask[:neck_y, :] = 0  # Zero out everything above neck
        
        # Step 5: Check if any shirt detected
        if np.sum(shirt_mask) < 1000:  # Too few shirt pixels
            cv2.imwrite(str(output_path), result)
            return True
        
        # Step 6: Find bounding box and apply jersey
        contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            cv2.imwrite(str(output_path), result)
            return True
        
        main_contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(main_contour)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(neck_y, y - padding)  # Ensure starts at or below neck
        bw = min(w - x, bw + 2 * padding)
        bh = min(h - y, bh + 2 * padding)
        
        # Resize jersey to match shirt dimensions
        if jersey is None or bw <= 0 or bh <= 0:
            cv2.imwrite(str(output_path), result)
            return True
        
        resized_jersey = cv2.resize(jersey, (bw, bh), interpolation=cv2.INTER_LINEAR)
        
        # Extract BGR and alpha
        if resized_jersey.shape[2] == 4:
            jersey_bgr = resized_jersey[:, :, :3]
            jersey_alpha = resized_jersey[:, :, 3] / 255.0
        else:
            jersey_bgr = resized_jersey
            jersey_alpha = np.ones((bh, bw))
        
        # Get shirt region mask in the bounding box
        shirt_region_mask = shirt_mask[y:y+bh, x:x+bw].astype(float)
        
        # Combine alpha with shirt mask
        combined_alpha = jersey_alpha * shirt_region_mask
        
        # Blend jersey onto result
        overlay_region = result[y:y+bh, x:x+bw]
        for c in range(3):
            overlay_region[:, :, c] = (
                combined_alpha * jersey_bgr[:, :, c] +
                (1 - combined_alpha) * overlay_region[:, :, c]
            )
        
        result[y:y+bh, x:x+bw] = overlay_region
        
        # Save result
        cv2.imwrite(str(output_path), result)
        return True
        
    except Exception as e:
        print(f"ERROR processing {image_path}: {e}")
        # Save original if error
        cv2.imwrite(str(output_path), result)
        return False


def main():
    """Apply jersey to all images in IcanDataset"""
    base_dir = Path(__file__).parent
    input_dir = base_dir / "IcanDataset"
    output_dir = base_dir / "OUTPUT_JERSEY"
    person_model_path = base_dir / "person_detector_model.joblib"
    shirt_model_path = base_dir / "shirt_detector_model_dual.joblib"
    jersey_path = base_dir.parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home_NOBG" / "Brighton Home.png"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("JERSEY APPLICATION - Below Neck Level")
    print("=" * 80)
    print()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check input directory
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Load models
    if not person_model_path.exists():
        print(f"❌ Person detector model not found: {person_model_path}")
        return
    
    if not shirt_model_path.exists():
        print(f"❌ Shirt detector model not found: {shirt_model_path}")
        return
    
    print("Loading models...")
    person_classifier = joblib.load(person_model_path)
    print(f"✓ Person detector loaded")
    
    shirt_classifier = joblib.load(shirt_model_path)
    print(f"✓ Shirt detector loaded")
    
    # Load jersey
    jersey = None
    if jersey_path.exists():
        jersey = cv2.imread(str(jersey_path), cv2.IMREAD_UNCHANGED)
        if jersey is not None:
            print(f"✓ Jersey loaded: {jersey.shape}")
    else:
        print(f"⚠ Jersey not found: {jersey_path}")
        print(f"  Will process images without jersey overlay")
    
    print()
    
    # Get all image files
    image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    
    print(f"Found {len(image_files)} images in {input_dir.name}")
    print()
    
    # Process each image
    successful = 0
    failed = 0
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        
        output_path = output_dir / f"{img_path.stem}_jersey{img_path.suffix}"
        
        if apply_jersey_to_image(img_path, person_classifier, shirt_classifier, jersey, output_path):
            print(f"  ✓ Saved: {output_path.name}")
            successful += 1
        else:
            print(f"  ✗ Failed")
            failed += 1
        print()
    
    # Summary
    print("=" * 80)
    print("✓ JERSEY APPLICATION COMPLETE!")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
