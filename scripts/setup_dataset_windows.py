"""
Setup ISIC2018 dataset from downloaded zip files.
Windows-specific version.

This script will:
1. Extract the training images and ground truth
2. Organize images into 3 simplified classes
3. Create the dataset structure for training
"""

import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths - modify these if needed
DOWNLOADS_DIR = Path(r"C:\Users\yaman\Downloads")
PROJECT_DIR = Path(r"C:\Users\yaman\Documents\quidquid")
OUTPUT_DIR = PROJECT_DIR / "data" / "raw_simplified"

# Zip files
TRAINING_IMAGES_ZIP = DOWNLOADS_DIR / "ISIC2018_Task3_Training_Input.zip"
GROUND_TRUTH_ZIP = DOWNLOADS_DIR / "ISIC2018_Task3_Training_GroundTruth(1).zip"

# Temporary extraction directory
TEMP_DIR = PROJECT_DIR / "data" / "temp_extract"

# Class mapping: 7 classes -> 3 classes
CLASS_MAPPING = {
    'MEL': 'malignant',      # Melanoma
    'NV': 'benign',          # Melanocytic nevus
    'BCC': 'malignant',      # Basal cell carcinoma
    'AKIEC': 'suspicious',   # Actinic keratosis
    'BKL': 'benign',         # Benign keratosis
    'DF': 'benign',          # Dermatofibroma
    'VASC': 'suspicious'     # Vascular lesion
}


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file with progress bar."""
    print(f"\nExtracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting"):
            zip_ref.extract(member, extract_to)

    print(f"Extracted to {extract_to}")


def organize_dataset():
    """Organize images into simplified class structure."""

    print("\n" + "=" * 60)
    print("ISIC2018 Dataset Setup for Training Round 2")
    print("=" * 60)

    # Create directories
    print("\nCreating directories...")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in ['benign', 'suspicious', 'malignant']:
        (OUTPUT_DIR / class_name).mkdir(exist_ok=True)

    # Extract training images
    if not TRAINING_IMAGES_ZIP.exists():
        print(f"\nERROR: Cannot find {TRAINING_IMAGES_ZIP}")
        print("Please ensure the file is in your Downloads folder.")
        return

    extract_zip(TRAINING_IMAGES_ZIP, TEMP_DIR)

    # Extract ground truth
    if not GROUND_TRUTH_ZIP.exists():
        print(f"\nERROR: Cannot find {GROUND_TRUTH_ZIP}")
        print("Please ensure the file is in your Downloads folder.")
        return

    extract_zip(GROUND_TRUTH_ZIP, TEMP_DIR)

    # Find the ground truth CSV
    print("\nLooking for ground truth CSV...")
    gt_csv = None
    for file in TEMP_DIR.rglob("*.csv"):
        if "GroundTruth" in file.name or "ISIC2018_Task3_Training" in file.name:
            gt_csv = file
            break

    if not gt_csv:
        print("ERROR: Could not find ground truth CSV file")
        print("Looking for any CSV in temp directory...")
        csv_files = list(TEMP_DIR.rglob("*.csv"))
        if csv_files:
            gt_csv = csv_files[0]
            print(f"Using: {gt_csv.name}")
        else:
            print("No CSV files found!")
            return

    print(f"Found ground truth: {gt_csv.name}")

    # Read ground truth
    print("\nReading ground truth...")
    df = pd.read_csv(gt_csv)
    print(f"Loaded {len(df)} image labels")
    print(f"\nColumns: {list(df.columns)}")

    # The CSV has columns: image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
    # Each row has 1.0 for the true class, 0.0 for others

    # Find images directory
    print("\nLooking for images directory...")
    images_dir = None
    for dir_path in TEMP_DIR.rglob("ISIC2018_Task3_Training_Input"):
        if dir_path.is_dir():
            images_dir = dir_path
            break

    if not images_dir:
        # Try looking for any directory with images
        for dir_path in TEMP_DIR.iterdir():
            if dir_path.is_dir():
                jpg_files = list(dir_path.glob("*.jpg"))
                if len(jpg_files) > 100:  # Should have thousands
                    images_dir = dir_path
                    break

    if not images_dir:
        print("ERROR: Could not find images directory")
        print(f"Please check {TEMP_DIR}")
        return

    print(f"Found images in: {images_dir}")

    # Organize images by class
    print("\nOrganizing images into 3 classes...")

    class_counts = {'benign': 0, 'suspicious': 0, 'malignant': 0}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_name = row['image']

        # Find which class has 1.0
        original_class = None
        for class_col in ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']:
            if class_col in row and row[class_col] == 1.0:
                original_class = class_col
                break

        if not original_class:
            print(f"\nWarning: No class found for {image_name}")
            continue

        # Map to simplified class
        simplified_class = CLASS_MAPPING[original_class]

        # Find the image file
        image_path = images_dir / f"{image_name}.jpg"
        if not image_path.exists():
            # Try without extension (might already have it)
            image_path = images_dir / image_name
            if not image_path.exists():
                print(f"\nWarning: Image not found: {image_name}")
                continue

        # Copy to appropriate class folder
        dest_path = OUTPUT_DIR / simplified_class / f"{image_name}.jpg"
        shutil.copy2(image_path, dest_path)
        class_counts[simplified_class] += 1

    print("\n" + "=" * 60)
    print("Dataset organization complete!")
    print("=" * 60)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name:12s}: {count:5d} images")
    print(f"  {'TOTAL':12s}: {sum(class_counts.values()):5d} images")

    # Clean up temp directory
    print("\nCleaning up temporary files...")
    try:
        shutil.rmtree(TEMP_DIR)
        print("Cleanup complete!")
    except Exception as e:
        print(f"Note: Could not delete temp directory: {e}")
        print(f"You can manually delete: {TEMP_DIR}")

    print("\n" + "=" * 60)
    print("Ready to train!")
    print("=" * 60)
    print("\nNext step: Run training with:")
    print("python -m src.model.train --data-dir data/raw_simplified --model-type resnet --num-classes 3 --epochs 30 --batch-size 8 --lr 0.0001 --device cpu")


if __name__ == "__main__":
    try:
        organize_dataset()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
