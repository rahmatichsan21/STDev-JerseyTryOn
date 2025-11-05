"""
Person Detector - Traditional ML
=================================

Trains a Random Forest classifier to detect PERSON vs BACKGROUND
using NOBG images as ground truth and WITH-BG images as context.

Strategy:
1. NOBG images: ALL pixels = PERSON (foreground)
2. WITH-BG images: Extract background pixels = NOT PERSON
3. Train classifier on mixed features to learn person vs background distinction
4. Result: Model understands what is "human" vs "environment"
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from shirt_detector_full import extract_pixel_features


def collect_person_training_data(nobg_dir, with_bg_dir, samples_per_image=5000):
    """
    Collect training data for person vs background detection
    
    PERSON samples (label=1):
    - All non-black pixels from NOBG images (these are human pixels)
    
    BACKGROUND samples (label=0):
    - Non-white pixels from WITH-BG images (these are background/environment)
    """
    print("\n" + "=" * 80)
    print("COLLECTING TRAINING DATA - PERSON vs BACKGROUND")
    print("=" * 80)
    
    nobg_files = sorted(list(nobg_dir.glob("*.png")))
    with_bg_files = sorted(list(with_bg_dir.glob("*.png")) + list(with_bg_dir.glob("*.jpg")))
    
    print(f"NOBG images: {len(nobg_files)}")
    print(f"WITH-BG images: {len(with_bg_files)}")
    print()
    
    X_train = []
    y_train = []
    
    # ============================================================================
    # PERSON SAMPLES from NOBG images
    # ============================================================================
    print("📊 Processing NOBG images → PERSON samples...")
    person_count = 0
    
    for nobg_path in tqdm(nobg_files, desc="NOBG"):
        nobg_image = cv2.imread(str(nobg_path))
        if nobg_image is None:
            continue
        
        h, w = nobg_image.shape[:2]
        
        # In NOBG: any pixel that's not black/empty = person
        gray = cv2.cvtColor(nobg_image, cv2.COLOR_BGR2GRAY)
        person_pixels = np.where(gray > 20)  # Non-black = person
        
        if len(person_pixels[0]) > 0:
            # Sample person pixels
            sample_indices = np.random.choice(
                len(person_pixels[0]),
                min(samples_per_image, len(person_pixels[0])),
                replace=False
            )
            
            for idx_sample in sample_indices:
                y, x = person_pixels[0][idx_sample], person_pixels[1][idx_sample]
                pixel = nobg_image[y, x]
                features = extract_pixel_features(pixel)
                X_train.append(features)
                y_train.append(1)  # PERSON
                person_count += 1
    
    print(f"✓ Collected PERSON samples: {person_count}")
    
    # ============================================================================
    # BACKGROUND SAMPLES from WITH-BG images
    # ============================================================================
    print("\n📊 Processing WITH-BG images → BACKGROUND samples...")
    bg_count = 0
    
    for with_bg_path in tqdm(with_bg_files, desc="WITH-BG"):
        with_bg_image = cv2.imread(str(with_bg_path))
        if with_bg_image is None:
            continue
        
        h, w = with_bg_image.shape[:2]
        
        # Strategy: Extract background by detecting person first, then invert
        # Simple approach: use color thresholding to find background
        
        # Convert to HSV for better background detection
        hsv = cv2.cvtColor(with_bg_image, cv2.COLOR_BGR2HSV)
        
        # Skin-like colors (person)
        skin_mask1 = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        
        ycrcb = cv2.cvtColor(with_bg_image, cv2.COLOR_BGR2YCrCb)
        skin_mask2 = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
        
        b, g, r = cv2.split(with_bg_image)
        skin_mask3 = ((r > 95) & (g > 40) & (b > 20) & 
                      (r > g) & (r > b) & 
                      ((r - g) > 15)).astype(np.uint8) * 255
        
        person_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask1, skin_mask2), skin_mask3)
        
        # Dark pixels (likely hair)
        gray = cv2.cvtColor(with_bg_image, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < 80).astype(np.uint8) * 255
        
        # Combine: person = skin or dark (hair)
        person_detected = cv2.bitwise_or(person_mask, dark_mask)
        person_detected = cv2.morphologyEx(person_detected, cv2.MORPH_CLOSE, 
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        
        # Background = NOT person
        background_mask = cv2.bitwise_not(person_detected)
        
        # Sample background pixels
        bg_pixels = np.where(background_mask > 0)
        
        if len(bg_pixels[0]) > 0:
            sample_indices = np.random.choice(
                len(bg_pixels[0]),
                min(samples_per_image, len(bg_pixels[0])),
                replace=False
            )
            
            for idx_sample in sample_indices:
                y, x = bg_pixels[0][idx_sample], bg_pixels[1][idx_sample]
                pixel = with_bg_image[y, x]
                features = extract_pixel_features(pixel)
                X_train.append(features)
                y_train.append(0)  # BACKGROUND
                bg_count += 1
    
    print(f"✓ Collected BACKGROUND samples: {bg_count}")
    
    return np.array(X_train), np.array(y_train)


def train_person_detector(X_train, y_train):
    """
    Train Random Forest for person vs background detection
    """
    print("\n" + "=" * 80)
    print("TRAINING PERSON DETECTOR - Random Forest")
    print("=" * 80)
    print()
    
    # Split data
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"Training set: {len(X_t)} samples")
    print(f"Validation set: {len(X_v)} samples")
    print(f"Class distribution (train): {np.bincount(y_t)}")
    print(f"Class distribution (val): {np.bincount(y_v)}")
    print()
    
    # Train Random Forest
    print("Training Random Forest (100 trees)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'  # Handle potential class imbalance
    )
    
    clf.fit(X_t, y_t)
    print("✓ Training complete!")
    print()
    
    # Evaluate
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    y_pred = clf.predict(X_v)
    accuracy = accuracy_score(y_v, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("Classification Report:")
    print(classification_report(y_v, y_pred, 
                                target_names=['Background', 'Person'],
                                digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_v, y_pred)
    print(f"  Predicted:     Background   Person")
    print(f"Actual Background:    {cm[0,0]:6d}      {cm[0,1]:6d}")
    print(f"Actual Person:        {cm[1,0]:6d}      {cm[1,1]:6d}")
    
    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (Top 10)")
    print("=" * 80)
    feature_names = ['B', 'G', 'R', 'H', 'S', 'V', 'Y', 'Cr', 'Cb']
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1:2d}. {feature_names[idx]:3s}: {importances[idx]:.6f}")
    
    return clf, accuracy


def main():
    """Train person detector model"""
    base_dir = Path(__file__).parent
    nobg_dir = base_dir / "IcanDataset_NOBG"
    with_bg_dir = base_dir / "IcanDataset"
    output_path = base_dir / "person_detector_model.joblib"
    
    # Check datasets exist
    if not nobg_dir.exists():
        print(f"❌ NOBG dataset not found: {nobg_dir}")
        return
    
    if not with_bg_dir.exists():
        print(f"❌ WITH-BG dataset not found: {with_bg_dir}")
        return
    
    print("=" * 80)
    print("PERSON DETECTOR - Traditional ML Training")
    print("=" * 80)
    print(f"\nNOBG directory: {nobg_dir}")
    print(f"WITH-BG directory: {with_bg_dir}")
    print(f"Output model: {output_path.name}")
    
    # Collect training data
    X_train, y_train = collect_person_training_data(nobg_dir, with_bg_dir, samples_per_image=5000)
    
    print(f"\n✓ Total training samples: {len(X_train)}")
    print(f"  Person samples: {np.sum(y_train == 1)}")
    print(f"  Background samples: {np.sum(y_train == 0)}")
    
    # Train model
    clf, accuracy = train_person_detector(X_train, y_train)
    
    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    joblib.dump(clf, output_path)
    print(f"✓ Model saved: {output_path}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Run: check_annotations_person.py")
    print(f"   Visualizes person detection on images")
    print(f"\n2. Update jersey_filter_server.py to use person_detector first")
    print(f"   Then apply shirt detection only within person region")
    print("=" * 80)


if __name__ == "__main__":
    main()
