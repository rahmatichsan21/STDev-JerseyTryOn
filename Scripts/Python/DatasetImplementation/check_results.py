"""
Compare before/after shirt detection results.
Shows what changed with the improved polygon.
"""

import cv2
import os

output_dir = "Output"

print("=" * 60)
print("SHIRT DETECTION RESULTS")
print("=" * 60)

# Check if files exist
files = [
    "3_shirt_region.png",
    "5_shirt_only.png", 
    "7_final_extracted.png"
]

for f in files:
    path = os.path.join(output_dir, f)
    if os.path.exists(path):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        size = os.path.getsize(path)
        print(f"\n✓ {f}")
        print(f"  Size: {w}x{h}")
        print(f"  File: {size:,} bytes")
    else:
        print(f"\n✗ {f} - NOT FOUND")

print("\n" + "=" * 60)
print("Check the 'Output' folder to see:")
print("  3_shirt_region.png  - Initial shirt area (GREEN)")
print("  5_shirt_only.png    - After skin removal (RED)")
print("  7_final_extracted.png - Final shirt ONLY")
print("=" * 60)
