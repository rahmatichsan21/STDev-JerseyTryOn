"""
Debug what's in the files
"""

import cv2
import numpy as np

# Check the extracted shirt
shirt_img = cv2.imread("Output/5_final_extracted.png")
shirt_mask = cv2.imread("Output/4_shirt_only.png", cv2.IMREAD_GRAYSCALE)

print("=== FILE INSPECTION ===")

if shirt_img is not None:
    print(f"\n5_final_extracted.png:")
    print(f"  Shape: {shirt_img.shape}")
    print(f"  Min: {shirt_img.min()}, Max: {shirt_img.max()}, Mean: {shirt_img.mean():.1f}")
    non_black = np.count_nonzero(shirt_img.sum(axis=2))
    print(f"  Non-black pixels: {non_black}")
else:
    print("\n✗ Could not load 5_final_extracted.png")

if shirt_mask is not None:
    print(f"\n4_shirt_only.png:")
    print(f"  Shape: {shirt_mask.shape}")
    print(f"  White pixels (shirt): {np.count_nonzero(shirt_mask)}")
else:
    print("\n✗ Could not load 4_shirt_only.png")

# Load jersey
jersey = cv2.imread("../../../Assets/Jerseys/PremierLeague/Home/Arsenal Home.jpg")
if jersey is not None:
    print(f"\nArsenal Home.jpg:")
    print(f"  Shape: {jersey.shape}")
    print(f"  Min: {jersey.min()}, Max: {jersey.max()}")
else:
    print("\n✗ Could not load Arsenal jersey")
