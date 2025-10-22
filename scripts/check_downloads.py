"""
Check what ISIC2018 files you have downloaded and what you still need.

This script scans your Downloads folder and tells you exactly what to download next.
"""

import os
from pathlib import Path
import zipfile


def check_isic_downloads(downloads_dir: str = None):
    """
    Check which ISIC2018 files are present in Downloads folder.

    Args:
        downloads_dir: Path to Downloads folder (auto-detects if None)
    """
    if downloads_dir is None:
        # Auto-detect Downloads folder
        if os.name == 'nt':  # Windows
            downloads_dir = os.path.join(os.environ['USERPROFILE'], 'Downloads')
        else:  # Unix/Linux/Mac
            downloads_dir = os.path.join(os.path.expanduser('~'), 'Downloads')

    downloads_path = Path(downloads_dir)

    if not downloads_path.exists():
        print(f"Error: Downloads folder not found at {downloads_dir}")
        return

    print("="*70)
    print("ISIC2018 Dataset Download Checker")
    print("="*70)
    print(f"Scanning: {downloads_dir}\n")

    # Define what we're looking for
    required_files = {
        'training': {
            'images': 'ISIC2018_Task3_Training_Input.zip',
            'labels': 'ISIC2018_Task3_Training_GroundTruth.csv',
            'required': True,
            'description': 'Training set (~10,015 images) - REQUIRED for training',
            'size': '~5.3 GB'
        },
        'validation': {
            'images': 'ISIC2018_Task3_Validation_Input.zip',
            'labels': 'ISIC2018_Task3_Validation_GroundTruth.csv',
            'required': False,
            'description': 'Validation set (~193 images) - Optional but recommended',
            'size': '~100 MB'
        },
        'test': {
            'images': 'ISIC2018_Task1-2_Test_Input.zip',
            'labels': None,  # Test set has no labels
            'required': False,
            'description': 'Test set (~1,000 images, no labels) - For final evaluation',
            'size': '~500 MB'
        }
    }

    # Also check for the validation ground truth ZIP the user mentioned
    alternative_validation_labels = 'ISIC2018_Task3_Validation_GroundTruth.zip'

    results = {}

    # Check each file
    for dataset_name, files_info in required_files.items():
        results[dataset_name] = {}

        # Check images
        images_file = files_info['images']
        images_path = downloads_path / images_file
        results[dataset_name]['images'] = images_path.exists()

        # Check labels
        if files_info['labels']:
            labels_file = files_info['labels']
            labels_path = downloads_path / labels_file
            results[dataset_name]['labels'] = labels_path.exists()

            # For validation, also check the ZIP variant
            if dataset_name == 'validation' and not labels_path.exists():
                alt_labels_path = downloads_path / alternative_validation_labels
                if alt_labels_path.exists():
                    results[dataset_name]['labels'] = True
                    results[dataset_name]['labels_file'] = alternative_validation_labels
        else:
            results[dataset_name]['labels'] = None

    # Display results
    print("DOWNLOAD STATUS:")
    print("-" * 70)

    for dataset_name, files_info in required_files.items():
        status_symbol = "✓" if all(v for v in results[dataset_name].values() if v is not None) else "✗"
        required_tag = "[REQUIRED]" if files_info['required'] else "[OPTIONAL]"

        print(f"\n{status_symbol} {dataset_name.upper()} {required_tag}")
        print(f"  {files_info['description']}")
        print(f"  Size: {files_info['size']}")

        # Images status
        img_status = "✓ Found" if results[dataset_name]['images'] else "✗ Missing"
        print(f"  Images: {img_status} - {files_info['images']}")

        # Labels status
        if files_info['labels']:
            lbl_status = "✓ Found" if results[dataset_name]['labels'] else "✗ Missing"
            labels_filename = results[dataset_name].get('labels_file', files_info['labels'])
            print(f"  Labels: {lbl_status} - {labels_filename}")

    # Summary and recommendations
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)

    # Check if training set is complete
    training_complete = results['training']['images'] and results['training']['labels']

    if training_complete:
        print("\n✓ TRAINING SET COMPLETE!")
        print("  You have everything needed to train your model.")
        print("\n  Next steps:")
        print("  1. Run: scripts\\setup_isic_windows.bat")
        print("  2. Or manually run:")
        print("     python scripts/organize_isic_data.py \\")
        print("       --extract \"C:\\Users\\yaman\\Downloads\\ISIC2018_Task3_Training_Input.zip\" \\")
        print("       --extract-to \"C:\\Users\\yaman\\Downloads\\extracted\"")
        print("     python scripts/organize_isic_data.py \\")
        print("       --images-dir \"C:\\Users\\yaman\\Downloads\\extracted\\ISIC2018_Task3_Training_Input\" \\")
        print("       --groundtruth-csv \"C:\\Users\\yaman\\Downloads\\ISIC2018_Task3_Training_GroundTruth.csv\" \\")
        print("       --output-dir data\\raw \\")
        print("       --task3-format")
    else:
        print("\n✗ TRAINING SET INCOMPLETE")
        print("  You need the training data to train your model.")
        print("\n  Download from: https://challenge.isic-archive.com/data/")
        print("\n  Required files:")
        if not results['training']['images']:
            print(f"  • {required_files['training']['images']} ({required_files['training']['size']})")
        if not results['training']['labels']:
            print(f"  • {required_files['training']['labels']}")

    # Check validation
    validation_complete = results['validation']['images'] and results['validation']['labels']
    if validation_complete:
        print("\n✓ VALIDATION SET COMPLETE!")
        print("  You can use this to evaluate your model during training.")
    elif results['validation']['labels'] and not results['validation']['images']:
        print("\n⚠ VALIDATION LABELS FOUND, but images missing")
        print(f"  You have: {results['validation'].get('labels_file', required_files['validation']['labels'])}")
        print(f"  Still need: {required_files['validation']['images']}")
        print("  Download from: https://challenge.isic-archive.com/data/")

    # Check test
    if results['test']['images']:
        print("\n✓ TEST SET FOUND!")
        print("  You can use this for inference after training.")

    print("\n" + "="*70)

    # List all ISIC-related files found
    print("\nALL ISIC FILES IN DOWNLOADS:")
    print("-" * 70)
    isic_files = list(downloads_path.glob("ISIC*"))
    if isic_files:
        for f in sorted(isic_files):
            size = f.stat().st_size if f.is_file() else 0
            size_mb = size / (1024 * 1024)
            file_type = "FILE" if f.is_file() else "DIR"
            print(f"  [{file_type}] {f.name:50} {size_mb:>8.1f} MB")
    else:
        print("  No ISIC files found")

    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check ISIC2018 dataset downloads')
    parser.add_argument(
        '--downloads-dir',
        type=str,
        help='Path to Downloads folder (auto-detects if not specified)'
    )

    args = parser.parse_args()

    check_isic_downloads(args.downloads_dir)
