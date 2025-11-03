# Detect Actual Shirt using Skin Detection + GrabCut Refinement

import cv2
import numpy as np
from pose_detection import detect_pose_landmarks
from detect_shirt_region import get_shirt_keypoints, create_shirt_mask

# Configuration
INPUT_IMAGE_PATH = r"Dataset_NoBG\Dataset_1-removebg-preview.png"
OUTPUT_DIR = r"Output"

def detect_skin(image, mask=None):
    """
    Detect skin regions in the image using HSV color space.
    
    Args:
        image: Input image (BGR)
        mask: Optional mask to limit detection area
        
    Returns:
        Binary mask of detected skin regions
    """
    # Convert to HSV and YCrCb color spaces for better skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color range in HSV
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    
    # Define skin color range in YCrCb
    lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
    
    # Create masks for both color spaces
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combine masks (intersection for better accuracy)
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    # Apply optional area mask
    if mask is not None:
        skin_mask = cv2.bitwise_and(skin_mask, mask)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel_small)
    
    return skin_mask

def create_shirt_only_mask(shirt_region_mask, skin_mask):
    """
    Remove skin areas from shirt region to get shirt-only mask.
    
    Args:
        shirt_region_mask: Mask of the general shirt region (from keypoints)
        skin_mask: Mask of detected skin areas
        
    Returns:
        Binary mask with skin removed from shirt region
    """
    # Subtract skin from shirt region
    shirt_only_mask = cv2.subtract(shirt_region_mask, skin_mask)
    
    # Clean up the result
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shirt_only_mask = cv2.morphologyEx(shirt_only_mask, cv2.MORPH_CLOSE, kernel)
    
    return shirt_only_mask

def refine_with_grabcut(image, initial_mask):
    """
    Refine shirt mask using GrabCut algorithm.
    
    Args:
        image: Input image (BGR)
        initial_mask: Initial shirt mask (binary)
        
    Returns:
        Refined shirt mask
    """
    # Convert initial mask to GrabCut format
    # GrabCut mask values: 0=definite background, 1=probable background, 
    #                      2=definite foreground, 3=probable foreground
    grabcut_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Set definite foreground where we have shirt
    grabcut_mask[initial_mask == 255] = cv2.GC_PR_FGD  # Probable foreground
    
    # Erode the mask to get definite shirt area
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    definite_shirt = cv2.erode(initial_mask, kernel, iterations=1)
    grabcut_mask[definite_shirt == 255] = cv2.GC_FGD  # Definite foreground
    
    # Set definite background (inverse of initial mask with margin)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    expanded_mask = cv2.dilate(initial_mask, kernel_large, iterations=1)
    grabcut_mask[expanded_mask == 0] = cv2.GC_BGD  # Definite background
    
    # Initialize background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Run GrabCut
    try:
        cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 
                    5, cv2.GC_INIT_WITH_MASK)
        
        # Create final mask (foreground only)
        refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | 
                               (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # Clean up the refined mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        
        return refined_mask
    except Exception as e:
        print(f"GrabCut failed: {e}")
        print("Returning initial mask without refinement")
        return initial_mask

def visualize_results(image, shirt_region_mask, skin_mask, shirt_only_mask, refined_mask):
    """
    Create visualization of all processing steps.
    
    Args:
        image: Original image
        shirt_region_mask: Initial shirt region from keypoints
        skin_mask: Detected skin mask
        shirt_only_mask: Shirt mask after skin removal
        refined_mask: Final refined shirt mask
        
    Returns:
        Combined visualization image
    """
    h, w = image.shape[:2]
    
    # Create overlays
    shirt_region_overlay = image.copy()
    shirt_region_colored = cv2.cvtColor(shirt_region_mask, cv2.COLOR_GRAY2BGR)
    shirt_region_colored[:, :, 0] = 0
    shirt_region_colored[:, :, 2] = 0
    shirt_region_overlay = cv2.addWeighted(image, 0.7, shirt_region_colored, 0.3, 0)
    
    skin_overlay = image.copy()
    skin_colored = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    skin_colored[:, :, 1] = 0
    skin_colored[:, :, 2] = 0
    skin_overlay = cv2.addWeighted(image, 0.7, skin_colored, 0.3, 0)
    
    shirt_only_overlay = image.copy()
    shirt_only_colored = cv2.cvtColor(shirt_only_mask, cv2.COLOR_GRAY2BGR)
    shirt_only_colored[:, :, 0] = 0
    shirt_only_colored[:, :, 1] = 0
    shirt_only_overlay = cv2.addWeighted(image, 0.7, shirt_only_colored, 0.3, 0)
    
    refined_overlay = image.copy()
    refined_colored = cv2.cvtColor(refined_mask, cv2.COLOR_GRAY2BGR)
    refined_colored[:, :, 0] = 0
    refined_colored[:, :, 1] = 0
    refined_overlay = cv2.addWeighted(image, 0.7, refined_colored, 0.3, 0)
    
    # Extract final shirt
    final_shirt = cv2.bitwise_and(image, image, mask=refined_mask)
    
    return {
        'shirt_region': shirt_region_overlay,
        'skin_detected': skin_overlay,
        'shirt_only': shirt_only_overlay,
        'refined_shirt': refined_overlay,
        'extracted_shirt': final_shirt
    }

def main():
    """Main execution function"""
    print("=" * 60)
    print("SHIRT DETECTION PIPELINE")
    print("=" * 60)
    
    print("\n[1/5] Loading image...")
    image = cv2.imread(INPUT_IMAGE_PATH)
    
    if image is None:
        print(f"✗ Error: Cannot load {INPUT_IMAGE_PATH}")
        return
    
    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h}")
    
    print("\n[2/5] Detecting pose landmarks...")
    results = detect_pose_landmarks(image)
    
    if not results.pose_landmarks:
        print("✗ No person detected")
        return
    
    print("✓ Person detected")
    
    print("\n[3/5] Creating shirt region mask from keypoints...")
    shirt_keypoints = get_shirt_keypoints(results.pose_landmarks, w, h)
    shirt_region_mask = create_shirt_mask(image, shirt_keypoints)
    
    shirt_pixels = np.count_nonzero(shirt_region_mask)
    print(f"✓ Shirt region created: {shirt_pixels} pixels")
    
    print("\n[4/5] Detecting skin regions...")
    skin_mask = detect_skin(image, mask=shirt_region_mask)
    
    skin_pixels = np.count_nonzero(skin_mask)
    print(f"✓ Skin detected: {skin_pixels} pixels")
    
    print("\n[5/5] Removing skin from shirt region...")
    shirt_only_mask = create_shirt_only_mask(shirt_region_mask, skin_mask)
    
    shirt_only_pixels = np.count_nonzero(shirt_only_mask)
    print(f"✓ Initial shirt mask: {shirt_only_pixels} pixels")
    
    print("\n[6/6] Refining shirt boundaries with GrabCut...")
    refined_mask = refine_with_grabcut(image, shirt_only_mask)
    
    refined_pixels = np.count_nonzero(refined_mask)
    print(f"✓ Refined shirt mask: {refined_pixels} pixels")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizations = visualize_results(image, shirt_region_mask, skin_mask, 
                                       shirt_only_mask, refined_mask)
    
    # Save all results instead of displaying (GUI not available)
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cv2.imwrite(f"{OUTPUT_DIR}/1_original.png", image)
    print(f"✓ Saved: {OUTPUT_DIR}/1_original.png")
    
    cv2.imwrite(f"{OUTPUT_DIR}/2_shirt_region.png", visualizations['shirt_region'])
    print(f"✓ Saved: {OUTPUT_DIR}/2_shirt_region.png")
    
    cv2.imwrite(f"{OUTPUT_DIR}/3_skin_detection.png", visualizations['skin_detected'])
    print(f"✓ Saved: {OUTPUT_DIR}/3_skin_detection.png")
    
    cv2.imwrite(f"{OUTPUT_DIR}/4_shirt_only.png", visualizations['shirt_only'])
    print(f"✓ Saved: {OUTPUT_DIR}/4_shirt_only.png")
    
    cv2.imwrite(f"{OUTPUT_DIR}/5_refined_shirt.png", visualizations['refined_shirt'])
    print(f"✓ Saved: {OUTPUT_DIR}/5_refined_shirt.png")
    
    cv2.imwrite(f"{OUTPUT_DIR}/6_final_extracted.png", visualizations['extracted_shirt'])
    print(f"✓ Saved: {OUTPUT_DIR}/6_final_extracted.png")
    
    print(f"\nAll visualizations saved to '{OUTPUT_DIR}' folder!")

def detect_shirt_mask_fast(image, pose_landmarks, w, h):
    """
    Fast shirt detection for real-time processing (skips GrabCut refinement)
    
    Args:
        image: Input frame (BGR)
        pose_landmarks: MediaPipe pose landmarks
        w, h: Frame dimensions
        
    Returns:
        Binary mask of shirt region
    """
    # Get shirt keypoints
    shirt_keypoints = get_shirt_keypoints(pose_landmarks, w, h)
    if not shirt_keypoints:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Create shirt region mask
    shirt_region_mask = create_shirt_mask(image, shirt_keypoints)
    
    # Quick skin detection
    skin_mask = detect_skin(image, shirt_region_mask)
    
    # Subtract skin from shirt region (no GrabCut refinement for speed)
    shirt_mask = create_shirt_only_mask(shirt_region_mask, skin_mask)
    
    return shirt_mask

if __name__ == "__main__":
    main()
