"""
Script untuk mengonversi berbagai jenis gambar (jpg, jpeg, bmp, gif, webp, dll) ke format PNG secara batch
"""

from PIL import Image
import os
from pathlib import Path


def convert_to_png(input_path, output_folder=None, keep_original=True):
    """
    Mengonversi gambar ke format PNG
    
    Args:
        input_path: Path ke file gambar atau folder yang berisi gambar
        output_folder: Folder output untuk menyimpan file PNG (opsional)
        keep_original: Jika True, file asli tidak akan dihapus
    
    Returns:
        List dari file yang berhasil dikonversi
    """
    # Format gambar yang didukung
    supported_formats = {'.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif', '.ico'}
    
    input_path = Path(input_path)
    converted_files = []
    
    # Jika input adalah file tunggal
    if input_path.is_file():
        files_to_convert = [input_path]
        # Jika output folder tidak ditentukan, gunakan folder yang sama dengan input
        if output_folder is None:
            output_folder = input_path.parent
    # Jika input adalah folder
    elif input_path.is_dir():
        # Cari semua file gambar di folder
        files_to_convert = [f for f in input_path.iterdir() 
                           if f.is_file() and f.suffix.lower() in supported_formats]
        # Jika output folder tidak ditentukan, gunakan folder input
        if output_folder is None:
            output_folder = input_path
    else:
        print(f"Error: Path '{input_path}' tidak ditemukan!")
        return converted_files
    
    # Buat output folder jika belum ada
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Ditemukan {len(files_to_convert)} file untuk dikonversi")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    # Konversi setiap file
    for file_path in files_to_convert:
        try:
            # Buka gambar
            img = Image.open(file_path)
            
            # Konversi ke RGB jika mode gambar adalah RGBA atau P (palette)
            # Ini penting untuk format seperti GIF dengan transparency
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                # Pertahankan alpha channel
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Buat nama file output
            output_filename = file_path.stem + '.png'
            output_path = output_folder / output_filename
            
            # Simpan sebagai PNG
            img.save(output_path, 'PNG', optimize=True)
            
            converted_files.append(output_path)
            print(f"✓ Berhasil: {file_path.name} → {output_filename}")
            
            # Hapus file asli jika diminta
            if not keep_original and file_path != output_path:
                file_path.unlink()
                print(f"  (File asli dihapus)")
            
        except Exception as e:
            print(f"✗ Gagal konversi {file_path.name}: {str(e)}")
    
    print("-" * 50)
    print(f"Konversi selesai! Total: {len(converted_files)} file berhasil dikonversi")
    
    return converted_files


def main():
    """
    Fungsi utama untuk menjalankan script
    """
    print("=" * 50)
    print("KONVERSI GAMBAR KE PNG - BATCH PROCESSING")
    print("=" * 50)
    print()
    
    # Contoh penggunaan 1: Konversi semua gambar di folder 'dataset'
    input_folder = "dataset"
    output_folder = "dataset_png"
    
    if os.path.exists(input_folder):
        print(f"Memproses folder: {input_folder}")
        convert_to_png(input_folder, output_folder, keep_original=True)
    else:
        print(f"Folder '{input_folder}' tidak ditemukan!")
        print()
        print("Gunakan script ini dengan cara:")
        print("1. Letakkan gambar-gambar di folder 'dataset'")
        print("2. Jalankan script ini")
        print("3. Hasil konversi akan tersimpan di folder 'dataset_png'")
        print()
        print("Atau edit fungsi main() untuk menyesuaikan folder input/output")
    
    print()
    
    # Contoh penggunaan 2: Konversi file tunggal
    # convert_to_png("path/to/image.jpg", "output_folder")
    
    # Contoh penggunaan 3: Konversi dan hapus file asli
    # convert_to_png("dataset", "dataset_png", keep_original=False)


if __name__ == "__main__":
    main()
