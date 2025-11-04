from rembg import remove
from PIL import Image
import os
from pathlib import Path

# Define input and output directories
script_dir = Path(__file__).parent
input_dir = script_dir / "dataset_png"
output_dir = script_dir / "dataset_rembg"

# Create output directory if it doesn't exist
output_dir.mkdir(exist_ok=True)

# Check if input directory exists
if not input_dir.exists():
    print(f"❌ Error: Input directory not found: {input_dir}")
    print(f"   Please create the folder and add images to process.")
    exit(1)

print(f"📁 Input:  {input_dir}")
print(f"📁 Output: {output_dir}")
print()

# Supported image extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

# Process all images in the dataset folder
image_files = list(input_dir.glob('*'))
total_files = len([f for f in image_files if f.suffix.lower() in image_extensions])

if total_files == 0:
    print(f"❌ No images found in {input_dir}")
    exit(1)

print(f"🔍 Found {total_files} images to process\n")

for idx, file_path in enumerate(image_files, 1):
    filename = file_path.name
    
    # Check if it's a file and has an image extension
    if file_path.is_file():
        file_ext = file_path.suffix.lower()
        
        if file_ext in image_extensions:
            # Create output filename - save as PNG for transparency
            output_filename = f"{file_path.stem}.png"
            output_path = output_dir / output_filename
            
            try:
                # Open the input image
                with Image.open(file_path) as img:
                    print(f"[{idx}/{total_files}] Processing: {filename}...")
                    
                    # Remove the background
                    result = remove(img)
                    
                    # Save the output image as PNG (supports transparency)
                    result.save(output_path, 'PNG')
                    print(f"           ✓ Saved: {output_filename}")
                    
            except Exception as e:
                print(f"           ✗ Error: {str(e)}")

print("\n" + "="*60)
print("✓ All images processed!")
print(f"✓ Output saved to: {output_dir}")
print("="*60)