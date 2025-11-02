from rembg import remove
from PIL import Image
import os

# Configuration - change this to the image you want to process
INPUT_IMAGE_PATH = r"Dataset_Raw\Dataset 1.jpg"
OUTPUT_DIR = r"Dataset_NoBG"

def remove_background_single(input_path, output_dir):
    """
    Remove background from a single image.
    
    Args:
        input_path: Path to input image
        output_dir: Directory to save output image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Output as PNG for transparency
    output_filename = f"{name_without_ext}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        print(f"Processing: {filename}...")
        
        # Open and process the image
        with Image.open(input_path) as img:
            print("  Removing background...")
            result = remove(img)
            
            # Save the result
            result.save(output_path)
            print(f"✓ Saved: {output_path}")
            
            # Display result info
            print(f"  Input size: {img.size}")
            print(f"  Output size: {result.size}")
            print(f"  Output mode: {result.mode}")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("BACKGROUND REMOVAL (Single Image)")
    print("=" * 60)
    print(f"\nInput: {INPUT_IMAGE_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    success = remove_background_single(INPUT_IMAGE_PATH, OUTPUT_DIR)
    
    if success:
        print("\n" + "=" * 60)
        print("BACKGROUND REMOVAL COMPLETE")
        print("=" * 60)
    else:
        print("\nFailed to remove background.")