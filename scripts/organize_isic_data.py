"""
Script to organize ISIC dataset for training.

The ISIC (International Skin Imaging Collaboration) dataset needs to be
organized into class folders for training.
"""

import os
import shutil
import zipfile
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def organize_isic_with_csv(
    images_dir: str,
    labels_csv: str,
    output_dir: str,
    image_col: str = 'image',
    label_col: str = 'diagnosis'
):
    """
    Organize ISIC images into class folders using a CSV file with labels.

    Args:
        images_dir: Directory containing all images
        labels_csv: Path to CSV file with image names and labels
        output_dir: Output directory for organized data
        image_col: Column name for image filenames in CSV
        label_col: Column name for labels/diagnosis in CSV
    """
    print("Reading labels CSV...")
    df = pd.read_csv(labels_csv)

    print(f"Found {len(df)} labeled images")
    print(f"Classes: {df[label_col].unique()}")
    print(f"Class distribution:\n{df[label_col].value_counts()}")

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create class directories
    classes = df[label_col].unique()
    for class_name in classes:
        class_dir = output_path / str(class_name)
        class_dir.mkdir(exist_ok=True)

    # Copy images to appropriate class folders
    print("\nOrganizing images into class folders...")
    images_dir_path = Path(images_dir)

    copied = 0
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row[image_col]
        label = row[label_col]

        # Try different extensions
        source_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = images_dir_path / f"{image_name}{ext}"
            if potential_path.exists():
                source_file = potential_path
                break

        if source_file is None:
            # Try without adding extension (in case it's already in the name)
            potential_path = images_dir_path / image_name
            if potential_path.exists():
                source_file = potential_path

        if source_file and source_file.exists():
            dest_file = output_path / str(label) / source_file.name
            shutil.copy2(source_file, dest_file)
            copied += 1
        else:
            skipped += 1
            if skipped <= 5:  # Only print first few missing files
                print(f"Warning: Could not find image {image_name}")

    print(f"\nCompleted!")
    print(f"Copied: {copied} images")
    print(f"Skipped: {skipped} images (not found)")

    # Print final statistics
    print("\nFinal dataset structure:")
    for class_dir in output_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*')))
            print(f"  {class_dir.name}: {count} images")


def organize_isic_task3_format(
    images_dir: str,
    groundtruth_csv: str,
    output_dir: str
):
    """
    Organize ISIC Task 3 format (multi-hot encoded labels).

    ISIC Task 3 uses columns like: MEL, NV, BCC, AKIEC, BKL, DF, VASC
    Each row has 1.0 in the column for its class.
    """
    print("Reading ISIC Task 3 ground truth CSV...")
    df = pd.read_csv(groundtruth_csv)

    print(f"CSV columns: {df.columns.tolist()}")

    # Identify label columns (exclude 'image' or similar identifier columns)
    label_cols = [col for col in df.columns if col not in ['image', 'image_id', 'isic_id']]
    image_col = 'image' if 'image' in df.columns else df.columns[0]

    print(f"Image column: {image_col}")
    print(f"Label columns: {label_cols}")

    # Convert multi-hot to single label
    df['diagnosis'] = df[label_cols].idxmax(axis=1)

    print(f"\nClass distribution:")
    print(df['diagnosis'].value_counts())

    # Now organize using the standard function
    organize_isic_with_csv(
        images_dir=images_dir,
        labels_csv=None,  # We'll use the df directly
        output_dir=output_dir,
        image_col=image_col,
        label_col='diagnosis'
    )

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create class directories
    for class_name in label_cols:
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)

    # Copy images
    print("\nOrganizing images into class folders...")
    images_dir_path = Path(images_dir)

    copied = 0
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row[image_col]
        label = row['diagnosis']

        # Try different extensions
        source_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_path = images_dir_path / f"{image_name}{ext}"
            if potential_path.exists():
                source_file = potential_path
                break

        if source_file is None:
            potential_path = images_dir_path / image_name
            if potential_path.exists():
                source_file = potential_path

        if source_file and source_file.exists():
            dest_file = output_path / label / source_file.name
            shutil.copy2(source_file, dest_file)
            copied += 1
        else:
            skipped += 1

    print(f"\nCompleted!")
    print(f"Copied: {copied} images")
    print(f"Skipped: {skipped} images")

    print("\nFinal dataset structure:")
    for class_dir in output_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*')))
            print(f"  {class_dir.name}: {count} images")


def main():
    parser = argparse.ArgumentParser(
        description='Organize ISIC dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract zip file
  python organize_isic_data.py --extract /path/to/images.zip --extract-to ./extracted

  # Organize with CSV labels
  python organize_isic_data.py \\
      --images-dir ./extracted/images \\
      --labels-csv ./labels.csv \\
      --output-dir ./data/raw \\
      --image-col image \\
      --label-col diagnosis

  # Organize ISIC Task 3 format (multi-hot encoded)
  python organize_isic_data.py \\
      --images-dir ./extracted/images \\
      --groundtruth-csv ISIC2018_Task3_Training_GroundTruth.csv \\
      --output-dir ./data/raw \\
      --task3-format
        """
    )

    parser.add_argument('--extract', type=str, help='Zip file to extract')
    parser.add_argument('--extract-to', type=str, help='Directory to extract to')

    parser.add_argument('--images-dir', type=str, help='Directory containing images')
    parser.add_argument('--labels-csv', type=str, help='CSV file with labels')
    parser.add_argument('--groundtruth-csv', type=str, help='ISIC groundtruth CSV (Task 3 format)')
    parser.add_argument('--output-dir', type=str, help='Output directory for organized data')

    parser.add_argument('--image-col', type=str, default='image', help='Image column name in CSV')
    parser.add_argument('--label-col', type=str, default='diagnosis', help='Label column name in CSV')
    parser.add_argument('--task3-format', action='store_true', help='Use ISIC Task 3 multi-hot format')

    args = parser.parse_args()

    # Extract zip if requested
    if args.extract:
        if not args.extract_to:
            print("Error: --extract-to required when using --extract")
            return
        extract_zip(args.extract, args.extract_to)

    # Organize data if requested
    if args.images_dir and args.output_dir:
        if args.task3_format and args.groundtruth_csv:
            organize_isic_task3_format(
                images_dir=args.images_dir,
                groundtruth_csv=args.groundtruth_csv,
                output_dir=args.output_dir
            )
        elif args.labels_csv:
            organize_isic_with_csv(
                images_dir=args.images_dir,
                labels_csv=args.labels_csv,
                output_dir=args.output_dir,
                image_col=args.image_col,
                label_col=args.label_col
            )
        else:
            print("Error: Need either --labels-csv or (--groundtruth-csv with --task3-format)")


if __name__ == "__main__":
    main()
