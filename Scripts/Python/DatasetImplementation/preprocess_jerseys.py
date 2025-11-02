"""
Preprocess jersey images - remove background and crop to content
"""

import cv2
import numpy as np
from pathlib import Path

# Jersey directory
JERSEY_DIR = Path(__file__).parent.parent.parent.parent / "Assets" / "Jerseys" / "PremierLeague" / "Home"
OUTPUT_DIR = JERSEY_DIR / "Processed"

def preprocess_jersey(jersey_path, output_path):
    """
    Remove background and crop jersey to just the shirt content
    
    Args:
        jersey_path: Path to original jersey image
        output_path: Path to save processed jersey
    """
    print(f"\nProcessing: {jersey_path.name}")
    
    # Load jersey
    jersey = cv2.imread(str(jersey_path))
    if jersey is None:
        print(f"  ✗ Failed to load")
        return False
    
    h, w = jersey.shape[:2]
    print(f"  Original size: {w}x{h}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(jersey, cv2.COLOR_BGR2GRAY)
    
    # Remove white background (threshold at 240 to catch near-white)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find the largest contour (the jersey)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  ✗ No jersey found")
        return False
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest)
    print(f"  Jersey bounds: {w}x{h} at ({x}, {y})")
    
    # Crop to bounding box
    jersey_cropped = jersey[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    
    # Set background to black (where mask is 0)
    result = jersey_cropped.copy()
    result[mask_cropped == 0] = [0, 0, 0]
    
    # Save processed jersey
    cv2.imwrite(str(output_path), result)
    print(f"  ✓ Saved: {w}x{h} → {output_path.name}")
    
    return True

def main():
    print("=" * 60)
    print("JERSEY PREPROCESSING")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Get all jersey images
    jersey_files = list(JERSEY_DIR.glob("*.jpg")) + list(JERSEY_DIR.glob("*.png"))
    jersey_files = [f for f in jersey_files if f.parent.name != "Processed"]
    
    print(f"Found {len(jersey_files)} jersey images")
    
    # Process each jersey
    success = 0
    failed = 0
    
    for jersey_path in jersey_files:
        output_path = OUTPUT_DIR / f"{jersey_path.stem}_processed.png"
        
        if preprocess_jersey(jersey_path, output_path):
            success += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✓ Success: {success}")
    print(f"✗ Failed: {failed}")
    print(f"\nProcessed jerseys saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
