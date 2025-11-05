"""
Dual-Dataset Shirt Detector
============================

Training on BOTH IcanDataset (WITH background) and IcanDataset_NOBG (NO background)

Strategy:
1. Load pair: image WITH background + SAME image WITHOUT background
2. Use NOBG as ground truth for what is "shirt"
3. Compare both to learn context-aware shirt detection
4. Train Random Forest on pixels from BOTH versions
5. Result: Model understands shirt in ANY context (background or not)

This prevents false positives like:
- Arms being classified as shirt
- Background artifacts being classified as shirt
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from tqdm import tqdm


def extract_pixel_features(pixel_bgr):
    """Extract 9D feature vector from single pixel"""
    b, g, r = float(pixel_bgr[0]), float(pixel_bgr[1]), float(pixel_bgr[2])
    
    # BGR (3)
    bgr_pixel = np.uint8([[pixel_bgr]])
    hsv = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
    ycrcb = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2YCrCb)[0][0]
    
    # Feature vector: BGR + HSV + YCrCb
    features = [b, g, r,  # BGR
                float(hsv[0]), float(hsv[1]), float(hsv[2]),  # HSV
                float(ycrcb[0]), float(ycrcb[1]), float(ycrcb[2])]  # YCrCb
    
    return np.array(features)


def detect_shirt_region_automatic(image):
    """
    Detect shirt region from no-background image (ground truth)
    Same logic as before
    """
    h, w = image.shape[:2]
    
    # Remove black background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = gray > 20
    
    # SKIN DETECTION
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin_mask1 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skin_mask2 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    
    b, g, r = cv2.split(image)
    skin_mask3 = ((r > 95) & (g > 40) & (b > 20) & 
                  (r > g) & (r > b) & 
                  ((r - g) > 15)).astype(np.uint8) * 255
    
    skin_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask1, skin_mask2), skin_mask3)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    # HAIR DETECTION - dark pixels at top
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    upper_half = np.zeros_like(gray_blurred)
    upper_half[:int(h * 0.4), :] = 1
    
    hair_mask = ((gray_blurred < 80) & (upper_half > 0)).astype(np.uint8) * 255
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    
    # NECK DETECTION
    skin_upper = skin_mask.copy()
    skin_upper[int(h * 0.5):, :] = 0
    
    contours, _ = cv2.findContours(skin_upper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    neck_y = int(h * 0.35)
    
    if len(contours) > 0:
        face_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(face_contour) > 1000:
            bottommost = tuple(face_contour[face_contour[:, :, 1].argmax()][0])
            neck_y = min(bottommost[1] + 20, h - 1)
    
    # SHIRT = Everything below neck (that's not skin/hair/background)
    shirt_mask = np.zeros((h, w), dtype=np.uint8)
    shirt_mask[neck_y:, :] = 255
    shirt_mask[skin_mask > 0] = 0
    shirt_mask[hair_mask > 0] = 0
    shirt_mask[foreground == 0] = 0
    
    return shirt_mask, skin_mask, hair_mask


def collect_training_data_dual_dataset(nobg_dir, with_bg_dir, samples_per_image=3000):
    """
    Collect training data from BOTH datasets
    
    For each image:
    - Use NOBG version as ground truth for shirt pixels
    - Use WITH-BG version to learn context
    - Sample pixels from both and label them
    """
    print("\n" + "=" * 80)
    print("COLLECTING TRAINING DATA FROM DUAL DATASETS")
    print("=" * 80)
    
    nobg_files = sorted(list(nobg_dir.glob("*.png")))
    with_bg_files = sorted(list(with_bg_dir.glob("*.png")) + list(with_bg_dir.glob("*.jpg")))
    
    print(f"NOBG images: {len(nobg_files)}")
    print(f"WITH-BG images: {len(with_bg_files)}")
    print()
    
    X_train = []
    y_train = []
    
    # Process NOBG images (clean ground truth)
    print("📊 Processing NOBG images (ground truth)...")
    for idx, nobg_path in enumerate(tqdm(nobg_files), 1):
        nobg_image = cv2.imread(str(nobg_path))
        if nobg_image is None:
            continue
        
        h, w = nobg_image.shape[:2]
        
        # Get ground truth shirt region
        shirt_mask, skin_mask, hair_mask = detect_shirt_region_automatic(nobg_image)
        
        # Get foreground
        gray = cv2.cvtColor(nobg_image, cv2.COLOR_BGR2GRAY)
        foreground = gray > 20
        
        # Sample SHIRT pixels
        shirt_pixels = np.where((shirt_mask > 0) & foreground)
        if len(shirt_pixels[0]) > 0:
            sample_indices = np.random.choice(len(shirt_pixels[0]), min(samples_per_image // 2, len(shirt_pixels[0])), replace=False)
            for idx_sample in sample_indices:
                y, x = shirt_pixels[0][idx_sample], shirt_pixels[1][idx_sample]
                pixel = nobg_image[y, x]
                features = extract_pixel_features(pixel)
                X_train.append(features)
                y_train.append(1)  # SHIRT
        
        # Sample NON-SHIRT pixels (skin, hair, background)
        non_shirt_pixels = np.where(((skin_mask > 0) | (hair_mask > 0) | (foreground == 0)) & (shirt_mask == 0))
        if len(non_shirt_pixels[0]) > 0:
            sample_indices = np.random.choice(len(non_shirt_pixels[0]), min(samples_per_image // 2, len(non_shirt_pixels[0])), replace=False)
            for idx_sample in sample_indices:
                y, x = non_shirt_pixels[0][idx_sample], non_shirt_pixels[1][idx_sample]
                pixel = nobg_image[y, x]
                features = extract_pixel_features(pixel)
                X_train.append(features)
                y_train.append(0)  # NOT SHIRT
    
    print(f"✓ Collected from NOBG: {len(X_train)} samples")
    
    # Process WITH-BG images (context learning)
    print("\n📊 Processing WITH-BG images (context learning)...")
    initial_samples = len(X_train)
    
    for with_bg_path in tqdm(with_bg_files[:len(nobg_files)]):  # Match number of images
        with_bg_image = cv2.imread(str(with_bg_path))
        if with_bg_image is None:
            continue
        
        h, w = with_bg_image.shape[:2]
        
        # Find corresponding NOBG image for ground truth
        stem = with_bg_path.stem
        nobg_match = nobg_dir / f"{stem}.png"
        
        if nobg_match.exists():
            nobg_image = cv2.imread(str(nobg_match))
            if nobg_image is None:
                continue
            
            # Get ground truth shirt region from NOBG
            shirt_mask_truth, _, _ = detect_shirt_region_automatic(nobg_image)
            
            # Sample from WITH-BG image using NOBG labels
            gray = cv2.cvtColor(with_bg_image, cv2.COLOR_BGR2GRAY)
            foreground = gray > 20
            
            # Sample SHIRT pixels
            shirt_pixels = np.where((shirt_mask_truth > 0) & foreground)
            if len(shirt_pixels[0]) > 0:
                sample_indices = np.random.choice(len(shirt_pixels[0]), min(samples_per_image // 2, len(shirt_pixels[0])), replace=False)
                for idx_sample in sample_indices:
                    y, x = shirt_pixels[0][idx_sample], shirt_pixels[1][idx_sample]
                    pixel = with_bg_image[y, x]
                    features = extract_pixel_features(pixel)
                    X_train.append(features)
                    y_train.append(1)  # SHIRT
            
            # Sample NON-SHIRT pixels
            non_shirt_pixels = np.where((shirt_mask_truth == 0) & foreground)
            if len(non_shirt_pixels[0]) > 0:
                sample_indices = np.random.choice(len(non_shirt_pixels[0]), min(samples_per_image // 2, len(non_shirt_pixels[0])), replace=False)
                for idx_sample in sample_indices:
                    y, x = non_shirt_pixels[0][idx_sample], non_shirt_pixels[1][idx_sample]
                    pixel = with_bg_image[y, x]
                    features = extract_pixel_features(pixel)
                    X_train.append(features)
                    y_train.append(0)  # NOT SHIRT
    
    print(f"✓ Collected from WITH-BG: {len(X_train) - initial_samples} samples")
    print(f"✓ Total training samples: {len(X_train)}")
    print(f"✓ Shirt (1): {sum(y_train)}, Non-Shirt (0): {len(y_train) - sum(y_train)}")
    
    return np.array(X_train), np.array(y_train)


def train_shirt_detector_dual(X, y):
    """Train Random Forest on dual dataset"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST (DUAL DATASET)")
    print("=" * 80)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\nTraining Random Forest...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Test Accuracy: {accuracy * 100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Shirt', 'Shirt']))
    
    return classifier, accuracy


def main():
    """Main training pipeline"""
    base_dir = Path(__file__).parent
    nobg_dir = base_dir / "IcanDataset_NOBG"
    with_bg_dir = base_dir / "IcanDataset"
    model_path = base_dir / "shirt_detector_model_dual.joblib"
    
    print("\n" + "=" * 80)
    print("DUAL-DATASET SHIRT DETECTION TRAINING")
    print("=" * 80)
    print("\nUsing both NOBG and WITH-BACKGROUND datasets for better generalization")
    print()
    
    # Check directories
    if not nobg_dir.exists():
        print(f"❌ NOBG dataset not found: {nobg_dir}")
        return
    
    if not with_bg_dir.exists():
        print(f"❌ WITH-BG dataset not found: {with_bg_dir}")
        return
    
    # Collect data
    X, y = collect_training_data_dual_dataset(nobg_dir, with_bg_dir, samples_per_image=3000)
    
    # Train
    classifier, accuracy = train_shirt_detector_dual(X, y)
    
    # Save model
    print(f"\n\nSaving model to {model_path}...")
    joblib.dump(classifier, model_path)
    print(f"✓ Model saved!")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print(f"✓ Accuracy: {accuracy * 100:.1f}%")
    print(f"✓ Model: {model_path}")
    print("\nYou can now use this model for jersey overlay with better accuracy!")
    print("=" * 80)


if __name__ == "__main__":
    main()
