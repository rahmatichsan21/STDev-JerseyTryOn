"""
Apply Brighton Jersey to Detected Shirt Regions
================================================

Uses the retrained model to detect shirt (blue regions) and overlay Brighton jersey.
"""

import cv2
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.insert(0, str(Path(__file__).parent))
from shirt_detector_full import detect_shirt_region_automatic, extract_pixel_features


def predict_shirt_mask_fast(image, classifier):
    """
    Predict shirt mask using the trained classifier
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


def overlay_jersey_on_shirt(original, shirt_mask, jersey_path):
    """
    Overlay Brighton jersey on the detected shirt region
    """
    # Load jersey
    jersey = cv2.imread(str(jersey_path), cv2.IMREAD_UNCHANGED)
    if jersey is None:
        print(f"ERROR: Could not load jersey from {jersey_path}")
        return original
    
    # Find shirt bounding box
    contours, _ = cv2.findContours(shirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return original
    
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Add padding
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(original.shape[1] - x, w + 2 * padding)
    h = min(original.shape[0] - y, h + 2 * padding)
    
    # Resize jersey to match shirt dimensions
    resized_jersey = cv2.resize(jersey, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create output
    result = original.copy()
    
    # Extract BGR and alpha
    if resized_jersey.shape[2] == 4:
        jersey_bgr = resized_jersey[:, :, :3]
        jersey_alpha = resized_jersey[:, :, 3] / 255.0
    else:
        jersey_bgr = resized_jersey
        jersey_alpha = np.ones((h, w))
    
    # Create region to overlay
    overlay_region = result[y:y+h, x:x+w]
    shirt_region_mask = shirt_mask[y:y+h, x:x+w].astype(float)
    
    # Combine alpha with shirt mask
    combined_alpha = jersey_alpha * shirt_region_mask
    
    # Blend jersey onto result
    for c in range(3):
        overlay_region[:, :, c] = (
            combined_alpha * jersey_bgr[:, :, c] +
            (1 - combined_alpha) * overlay_region[:, :, c]
        )
    
    result[y:y+h, x:x+w] = overlay_region
    
    return result


def main():
    """Apply Brighton jersey to all images"""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "IcanDataset_NOBG"
    output_dir = base_dir / "final_jersey_output"
    model_path = base_dir / "shirt_detector_model.joblib"
    
    jersey_path = base_dir.parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home_NOBG" / "Brighton Home.png"
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("APPLYING BRIGHTON JERSEY TO DETECTED SHIRT REGIONS")
    print("=" * 80)
    print()
    
    # Load model
    print("Loading trained model...")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run shirt_detector_full.py first to train the model!")
        return
    
    classifier = joblib.load(model_path)
    print(f"✓ Model loaded from {model_path}")
    
    # Load jersey
    print(f"Loading jersey: {jersey_path.name}")
    if not jersey_path.exists():
        print(f"ERROR: Jersey not found at {jersey_path}")
        return
    
    jersey = cv2.imread(str(jersey_path), cv2.IMREAD_UNCHANGED)
    print(f"✓ Jersey loaded: {jersey.shape}")
    print()
    
    # Process all images
    print("=" * 80)
    print("PROCESSING ALL IMAGES")
    print("=" * 80)
    
    image_files = list(dataset_dir.glob("*.png"))
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("  ERROR: Could not load image")
            continue
        
        # Predict shirt mask
        print("  Detecting shirt region...")
        shirt_mask = predict_shirt_mask_fast(image, classifier)
        
        n_shirt = np.sum(shirt_mask == 1)
        n_total = np.sum(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 20)
        shirt_percent = (n_shirt / n_total * 100) if n_total > 0 else 0
        print(f"  Shirt detected: {n_shirt}/{n_total} pixels ({shirt_percent:.1f}%)")
        
        # Overlay jersey
        print("  Applying Brighton jersey...")
        result = overlay_jersey_on_shirt(image, shirt_mask, jersey_path)
        
        # Save results
        cv2.imwrite(str(output_dir / f"result_{img_path.name}"), result)
        cv2.imwrite(str(output_dir / f"mask_{img_path.name}"), shirt_mask * 255)
        
        # Create comparison
        h_resize = 300
        w_resize = int(300 * image.shape[1] / image.shape[0])
        comparison = np.hstack([
            cv2.resize(image, (w_resize, h_resize)),
            cv2.resize(result, (w_resize, h_resize))
        ])
        cv2.imwrite(str(output_dir / f"compare_{img_path.name}"), comparison)
        
        print(f"  ✓ Saved results")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print(f"✓ Processed {len(image_files)} images")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
