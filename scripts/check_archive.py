"""
Check contents of archive.zip to identify what dataset it contains.
"""

import zipfile
from pathlib import Path
import sys

def check_archive_contents(zip_path: str):
    """Check what's inside an archive.zip file."""

    if not Path(zip_path).exists():
        print(f"Error: File not found: {zip_path}")
        return

    print("="*70)
    print("Archive Contents Checker")
    print("="*70)
    print(f"File: {zip_path}")
    print(f"Size: {Path(zip_path).stat().st_size / (1024**3):.2f} GB")
    print()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()

            print(f"Total files/folders in archive: {len(file_list)}")
            print()

            # Show first 20 files
            print("First 20 files:")
            print("-"*70)
            for i, filename in enumerate(file_list[:20], 1):
                print(f"{i:3}. {filename}")

            if len(file_list) > 20:
                print(f"... and {len(file_list) - 20} more files")

            print()
            print("="*70)
            print("ANALYSIS:")
            print("="*70)

            # Check for ISIC patterns
            isic_images = [f for f in file_list if 'ISIC_' in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            csv_files = [f for f in file_list if f.endswith('.csv')]

            print(f"ISIC images found: {len(isic_images)}")
            print(f"CSV files found: {len(csv_files)}")

            if csv_files:
                print("\nCSV files:")
                for csv in csv_files:
                    print(f"  - {csv}")

            # Determine what this likely is
            print("\n" + "="*70)
            print("IDENTIFICATION:")
            print("="*70)

            if len(isic_images) > 9000:
                print("✓ This appears to be the TRAINING SET (~10,015 images)")
                print("  Perfect! This is what you need!")
            elif len(isic_images) > 500 and len(isic_images) < 2000:
                print("✓ This appears to be the TEST SET (~1,000 images)")
                print("  You already have this!")
            elif len(isic_images) > 100 and len(isic_images) < 300:
                print("✓ This appears to be the VALIDATION SET (~193 images)")
                print("  This is optional but useful!")
            elif len(isic_images) > 0:
                print(f"? Found {len(isic_images)} ISIC images")
                print("  Not sure which set this is")
            else:
                print("⚠ This doesn't appear to be ISIC image data")
                print("  It might be labels/metadata only")

            # Check folder structure
            folders = set([f.split('/')[0] for f in file_list if '/' in f])
            if folders:
                print(f"\nTop-level folders: {len(folders)}")
                for folder in sorted(list(folders)[:10]):
                    count = len([f for f in file_list if f.startswith(folder + '/')])
                    print(f"  - {folder}/ ({count} items)")

    except zipfile.BadZipFile:
        print("ERROR: This is not a valid ZIP file!")
    except Exception as e:
        print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    else:
        # Default Windows Downloads path
        import os
        zip_path = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'archive.zip')

    check_archive_contents(zip_path)
