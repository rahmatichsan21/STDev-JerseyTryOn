"""
Overlay jersey result onto original image to make it look like person is wearing it
"""

import cv2
import numpy as np
from pathlib import Path

# Paths
INPUT_IMAGE = "Dataset_NoBG/Dataset_1-removebg-preview.png"  # Original image
JERSEY_RESULT = "Output/jersey_result.png"  # Jersey-fitted result
OUTPUT_DIR = "Output"

def overlay_jersey_on_person(original_image, jersey_result):
    """
    Overlay the jersey result onto the original image.
    
    Args:
        original_image: Original person image
        jersey_result: Jersey fitted to shirt shape
        
    Returns:
        Combined image with jersey overlaid on person
    """
    h, w = original_image.shape[:2]
    
    # Create mask from jersey result (non-black pixels are the jersey)
    gray_jersey = cv2.cvtColor(jersey_result, cv2.COLOR_BGR2GRAY)
    _, jersey_mask = cv2.threshold(gray_jersey, 10, 255, cv2.THRESH_BINARY)
    
    # Convert mask to 3 channels
    jersey_mask_3ch = cv2.cvtColor(jersey_mask, cv2.COLOR_GRAY2BGR)
    
    # Create inverse mask for the person (areas without jersey)
    inverse_mask = cv2.bitwise_not(jersey_mask_3ch)
    
    # Extract person without jersey area
    person_without_jersey = cv2.bitwise_and(original_image, inverse_mask)
    
    # Extract jersey area
    jersey_only = cv2.bitwise_and(jersey_result, jersey_mask_3ch)
    
    # Combine: person + jersey
    result = cv2.add(person_without_jersey, jersey_only)
    
    jersey_pixels = np.count_nonzero(jersey_mask)
    print(f"✓ Overlaid {jersey_pixels} jersey pixels onto original image")
    
    return result, jersey_mask

def create_smooth_overlay(original_image, jersey_result):
    """
    Create a smooth overlay with edge blending for more realistic appearance.
    
    Args:
        original_image: Original person image
        jersey_result: Jersey fitted to shirt shape
        
    Returns:
        Combined image with smooth jersey overlay
    """
    h, w = original_image.shape[:2]
    
    # Create mask from jersey result
    gray_jersey = cv2.cvtColor(jersey_result, cv2.COLOR_BGR2GRAY)
    _, jersey_mask = cv2.threshold(gray_jersey, 10, 255, cv2.THRESH_BINARY)
    
    # Apply Gaussian blur to mask edges for smooth transition
    jersey_mask_smooth = cv2.GaussianBlur(jersey_mask, (15, 15), 0)
    
    # Normalize mask to 0-1 range for alpha blending
    alpha = jersey_mask_smooth.astype(float) / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=2)
    
    # Alpha blend: result = jersey * alpha + original * (1 - alpha)
    result = (jersey_result.astype(float) * alpha + 
              original_image.astype(float) * (1 - alpha))
    result = result.astype(np.uint8)
    
    print(f"✓ Created smooth overlay with edge blending")
    
    return result, jersey_mask_smooth

def main():
    print("=" * 60)
    print("JERSEY OVERLAY - Combine Jersey with Original Person")
    print("=" * 60)
    print(f"DEBUG: Script started, Python is working!")
    print(f"DEBUG: Current working directory check...")
    import os
    print(f"DEBUG: CWD = {os.getcwd()}")
    
    # Load original image
    print(f"\n[1/4] Loading original image: {INPUT_IMAGE}")
    original = cv2.imread(INPUT_IMAGE)
    
    if original is None:
        print(f"✗ Failed to load: {INPUT_IMAGE}")
        return
    
    h, w = original.shape[:2]
    print(f"✓ Original image loaded: {w}x{h}")
    
    # Load jersey result
    print(f"\n[2/4] Loading jersey result: {JERSEY_RESULT}")
    jersey_result = cv2.imread(JERSEY_RESULT)
    
    if jersey_result is None:
        print(f"✗ Failed to load: {JERSEY_RESULT}")
        return
    
    print(f"✓ Jersey result loaded: {jersey_result.shape}")
    
    # Ensure both images are same size
    if original.shape != jersey_result.shape:
        print(f"\nResizing jersey result to match original...")
        jersey_result = cv2.resize(jersey_result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        print(f"✓ Jersey result resized to {w}x{h}")
    
    # Create basic overlay
    print(f"\n[3/4] Creating basic overlay...")
    basic_result, basic_mask = overlay_jersey_on_person(original, jersey_result)
    
    # Create smooth overlay
    print(f"\n[4/4] Creating smooth overlay with edge blending...")
    smooth_result, smooth_mask = create_smooth_overlay(original, jersey_result)
    
    print("\n" + "=" * 60)
    print("OVERLAY COMPLETE")
    print("=" * 60)
    
    # Save results
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save basic overlay
    basic_output = f"{OUTPUT_DIR}/final_basic_overlay.png"
    cv2.imwrite(basic_output, basic_result)
    print(f"\n✓ Saved basic overlay: {basic_output}")
    
    # Save smooth overlay
    smooth_output = f"{OUTPUT_DIR}/final_smooth_overlay.png"
    cv2.imwrite(smooth_output, smooth_result)
    print(f"✓ Saved smooth overlay: {smooth_output}")
    
    # Create comparison visualization
    print(f"\nCreating comparison visualization...")
    
    # Resize for comparison (to fit side by side)
    scale = 0.5
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    orig_small = cv2.resize(original, (new_w, new_h))
    basic_small = cv2.resize(basic_result, (new_w, new_h))
    smooth_small = cv2.resize(smooth_result, (new_w, new_h))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_small, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(basic_small, "Basic Overlay", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(smooth_small, "Smooth Overlay", (10, 30), font, 1, (255, 255, 255), 2)
    
    # Stack horizontally
    comparison = np.hstack([orig_small, basic_small, smooth_small])
    comparison_output = f"{OUTPUT_DIR}/overlay_comparison.png"
    cv2.imwrite(comparison_output, comparison)
    print(f"✓ Saved comparison: {comparison_output}")
    
    print(f"\n{'=' * 60}")
    print("All results saved successfully!")
    print(f"{'=' * 60}")
    print(f"\nBest result: {smooth_output}")
    print("This shows the person wearing the jersey with smooth edges.")

if __name__ == "__main__":
    main()
