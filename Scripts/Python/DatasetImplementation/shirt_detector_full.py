"""
SHIRT DETECTOR - Traditional ML
================================

Goal: Detect SHIRT pixels ONLY. Not skin, not hair, just SHIRT.

Uses ALL images in the dataset.
Uses Traditional ML: Random Forest on color features.
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path


def detect_shirt_region_automatic(image):
    """
    Automatically detect shirt region using color-based heuristics
    
    Strategy:
    1. Detect SKIN (face, neck, arms) using YCrCb
    2. Detect HAIR (dark regions at top of image)
    3. Find NECK position (lowest point of face)
    4. SHIRT = Everything BELOW neck level (that's not skin/arms)
    """
    h, w = image.shape[:2]
    
    # Remove black background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    # ===== SKIN DETECTION =====
    # Method 1: YCrCb color space (best for skin)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask1 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    # Method 2: HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    
    # Method 3: RGB rules
    b, g, r = cv2.split(image)
    skin_mask3 = ((r > 95) & (g > 40) & (b > 20) & 
                  (r > g) & (r > b) & 
                  ((r - g) > 15)).astype(np.uint8) * 255
    
    # Combine all skin detection methods
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask3)
    
    # Clean up skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # ===== FIND NECK LEVEL =====
    # Find skin contours in upper half of image (face region)
    upper_half = np.zeros_like(skin_mask)
    upper_half[:int(h * 0.5), :] = 255
    face_skin = cv2.bitwise_and(skin_mask, upper_half)
    
    # Find contours of face
    face_contours, _ = cv2.findContours(face_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Determine neck level (lowest point of face + small margin)
    neck_y = int(h * 0.35)  # Default to 35% from top if no face found
    if len(face_contours) > 0:
        # Get largest contour (main face)
        face_contour = max(face_contours, key=cv2.contourArea)
        if cv2.contourArea(face_contour) > 1000:  # Valid face detected
            # Find lowest point of face
            bottommost = tuple(face_contour[face_contour[:, :, 1].argmax()][0])
            neck_y = bottommost[1] + 20  # Add small margin below chin
    
    print(f"  Detected neck level at y={neck_y} (image height={h})")
    
    # ===== HAIR DETECTION =====
    # Hair is typically dark and ABOVE neck level
    hair_mask = (gray < 50) & foreground
    
    # Only consider regions ABOVE neck level
    above_neck = np.zeros_like(hair_mask)
    above_neck[:neck_y, :] = True
    hair_mask = hair_mask & above_neck
    
    # Clean up hair mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hair_mask = cv2.morphologyEx(hair_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel_small)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel_small)
    
    # ===== SHIRT DETECTION =====
    # Shirt = Foreground BELOW neck level - Skin (arms)
    
    # Create mask for region below neck
    below_neck = np.zeros_like(foreground)
    below_neck[neck_y:, :] = True
    
    # Shirt is foreground below neck, excluding skin/arms
    shirt_mask = foreground.astype(np.uint8) * 255
    shirt_mask = cv2.bitwise_and(shirt_mask, below_neck.astype(np.uint8) * 255)
    shirt_mask = cv2.bitwise_and(shirt_mask, cv2.bitwise_not(skin_mask))
    
    # Clean up shirt mask
    shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_CLOSE, kernel)
    shirt_mask = cv2.morphologyEx(shirt_mask, cv2.MORPH_OPEN, kernel_small)
    
    return shirt_mask, skin_mask, hair_mask


def extract_pixel_features(pixel_bgr):
    """
    Extract features from a single pixel
    
    Features (9 total):
    - BGR (3)
    - HSV (3)
    - YCrCb (3)
    """
    b, g, r = float(pixel_bgr[0]), float(pixel_bgr[1]), float(pixel_bgr[2])
    
    # Convert to HSV
    bgr_pixel = np.uint8([[pixel_bgr]])
    hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])
    
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2YCrCb)[0][0]
    y, cr, cb = float(ycrcb[0]), float(ycrcb[1]), float(ycrcb[2])
    
    return np.array([b, g, r, h, s, v, y, cr, cb], dtype=np.float32)


def collect_training_data_from_all_images(dataset_dir, pixels_per_image=3000):
    """
    Collect training data from ALL images in the dataset
    
    Returns:
    - X: Feature matrix
    - y: Labels (0 = NOT shirt, 1 = shirt)
    """
    dataset_dir = Path(dataset_dir)
    image_files = list(dataset_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} images in dataset")
    print(f"Processing ALL images...")
    print("=" * 80)
    
    all_shirt_pixels = []
    all_non_shirt_pixels = []
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  ERROR: Could not load image")
            continue
        
        # Detect regions automatically
        shirt_mask, skin_mask, hair_mask = detect_shirt_region_automatic(image)
        
        # Get shirt pixels
        shirt_coords = np.where(shirt_mask > 0)
        n_shirt = len(shirt_coords[0])
        
        # Get non-shirt pixels (skin + hair)
        non_shirt_mask = cv2.bitwise_or(skin_mask, hair_mask)
        non_shirt_coords = np.where(non_shirt_mask > 0)
        n_non_shirt = len(non_shirt_coords[0])
        
        # Sample pixels
        if n_shirt > 0:
            n_sample = min(pixels_per_image, n_shirt)
            indices = np.random.choice(n_shirt, n_sample, replace=False)
            for i in indices:
                y, x = shirt_coords[0][i], shirt_coords[1][i]
                all_shirt_pixels.append(image[y, x])
        
        if n_non_shirt > 0:
            n_sample = min(pixels_per_image, n_non_shirt)
            indices = np.random.choice(n_non_shirt, n_sample, replace=False)
            for i in indices:
                y, x = non_shirt_coords[0][i], non_shirt_coords[1][i]
                all_non_shirt_pixels.append(image[y, x])
        
        print(f"  Shirt pixels: {n_shirt}, Non-shirt pixels: {n_non_shirt}")
    
    # Convert to numpy arrays
    print("\n" + "=" * 80)
    print("Converting to feature matrix...")
    
    all_shirt_pixels = np.array(all_shirt_pixels) if len(all_shirt_pixels) > 0 else np.array([]).reshape(0, 3)
    all_non_shirt_pixels = np.array(all_non_shirt_pixels) if len(all_non_shirt_pixels) > 0 else np.array([]).reshape(0, 3)
    
    print(f"Total SHIRT pixels collected: {len(all_shirt_pixels)}")
    print(f"Total NON-SHIRT pixels collected: {len(all_non_shirt_pixels)}")
    
    if len(all_shirt_pixels) == 0 or len(all_non_shirt_pixels) == 0:
        raise ValueError("Need both shirt and non-shirt pixels!")
    
    # Extract features
    print("Extracting features from pixels...")
    X_shirt = np.array([extract_pixel_features(px) for px in all_shirt_pixels])
    X_non_shirt = np.array([extract_pixel_features(px) for px in all_non_shirt_pixels])
    
    # Create labels
    y_shirt = np.ones(len(X_shirt))      # 1 = SHIRT
    y_non_shirt = np.zeros(len(X_non_shirt))  # 0 = NOT SHIRT
    
    # Combine
    X = np.vstack([X_shirt, X_non_shirt])
    y = np.concatenate([y_shirt, y_non_shirt])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Class distribution: SHIRT={np.sum(y==1)}, NON-SHIRT={np.sum(y==0)}")
    
    return X, y


def train_shirt_detector(X, y):
    """
    Train Random Forest to detect shirt pixels
    """
    print("\n" + "=" * 80)
    print("TRAINING SHIRT DETECTOR")
    print("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['NON-SHIRT', 'SHIRT']))
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = ['B', 'G', 'R', 'H', 'S', 'V', 'Y', 'Cr', 'Cb']
    importances = clf.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {imp:.4f}")
    
    return clf


def predict_shirt_mask(image, classifier):
    """
    Predict shirt mask for an entire image
    
    Returns binary mask where 1 = SHIRT, 0 = NOT SHIRT
    """
    h, w = image.shape[:2]
    
    # Get foreground mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    # Get all foreground pixel coordinates
    coords = np.where(foreground)
    
    if len(coords[0]) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Extract features for all foreground pixels
    print(f"  Extracting features for {len(coords[0])} pixels...")
    pixels = image[coords]
    features = np.array([extract_pixel_features(px) for px in pixels])
    
    # Predict
    print(f"  Classifying pixels...")
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
    Overlay jersey ONLY on shirt region
    """
    # Load jersey
    jersey = cv2.imread(str(jersey_path), cv2.IMREAD_UNCHANGED)
    if jersey is None:
        print(f"ERROR: Could not load jersey from {jersey_path}")
        return original
    
    # Resize jersey to match image
    jersey = cv2.resize(jersey, (original.shape[1], original.shape[0]))
    
    # Extract BGR and alpha
    if jersey.shape[2] == 4:
        jersey_bgr = jersey[:, :, :3]
        alpha = jersey[:, :, 3] / 255.0
    else:
        jersey_bgr = jersey
        alpha = np.ones((jersey.shape[0], jersey.shape[1]))
    
    # Create result
    result = original.copy()
    
    # Apply jersey ONLY where shirt_mask == 1
    shirt_regions = (shirt_mask == 1)
    
    for c in range(3):
        result[:, :, c] = np.where(
            shirt_regions,
            alpha * jersey_bgr[:, :, c] + (1 - alpha) * original[:, :, c],
            original[:, :, c]
        )
    
    return result.astype(np.uint8)


def main():
    """Main execution"""
    base_dir = Path(__file__).parent
    dataset_dir = base_dir / "IcanDataset_NOBG"
    output_dir = base_dir / "shirt_detector_output"
    model_path = base_dir / "shirt_detector_model.joblib"
    
    jersey_path = base_dir.parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home_NOBG" / "Brighton Home.png"
    
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("SHIRT DETECTOR - USING ALL DATASET")
    print("=" * 80)
    print()
    
    # Collect training data from ALL images
    X, y = collect_training_data_from_all_images(dataset_dir, pixels_per_image=2000)
    
    # Train classifier
    classifier = train_shirt_detector(X, y)
    
    # Save model
    joblib.dump(classifier, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Test on ALL images
    print("\n" + "=" * 80)
    print("TESTING ON ALL IMAGES")
    print("=" * 80)
    
    test_images = list(dataset_dir.glob("*.png"))
    
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Processing: {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print("  ERROR: Could not load image")
            continue
        
        # Predict shirt mask
        shirt_mask = predict_shirt_mask(image, classifier)
        
        # Count shirt pixels
        n_shirt = np.sum(shirt_mask == 1)
        n_total = np.sum(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > 20)
        shirt_percentage = (n_shirt / n_total * 100) if n_total > 0 else 0
        print(f"  Shirt pixels: {n_shirt}/{n_total} ({shirt_percentage:.1f}%)")
        
        # Overlay jersey
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
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print(f"✓ Processed {len(test_images)} images")
    print(f"✓ Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
