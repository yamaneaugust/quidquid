"""
Simplify ISIC 7-class dataset to 3-class dataset.

Merges similar classes:
- Benign: NV, BKL, DF, VASC
- Suspicious: AKIEC
- Malignant: MEL, BCC
"""

import shutil
from pathlib import Path
import argparse
from tqdm import tqdm


# Class mapping from ISIC 7 classes to 3 simplified classes
DEFAULT_CLASS_MAPPING = {
    'MEL': 'malignant',      # Melanoma
    'BCC': 'malignant',      # Basal cell carcinoma
    'AKIEC': 'suspicious',   # Actinic keratosis (pre-cancerous)
    'NV': 'benign',          # Melanocytic nevus (mole)
    'BKL': 'benign',         # Benign keratosis
    'DF': 'benign',          # Dermatofibroma
    'VASC': 'benign',        # Vascular lesion
}


def simplify_classes(
    source_dir: str,
    target_dir: str,
    class_mapping: dict = None,
    copy: bool = True
):
    """
    Simplify multi-class dataset to fewer classes.

    Args:
        source_dir: Source directory with original classes
        target_dir: Target directory for simplified classes
        class_mapping: Dictionary mapping original -> new class names
        copy: If True, copy files; if False, move files
    """
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist!")
        return

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)

    # Create new class directories
    new_classes = set(class_mapping.values())
    for new_class in new_classes:
        (target_path / new_class).mkdir(exist_ok=True)

    print(f"Simplifying classes from {source_dir} to {target_dir}")
    print(f"\nClass mapping:")
    for orig, new in class_mapping.items():
        print(f"  {orig:10} -> {new}")

    # Process each original class
    total_copied = 0
    class_counts = {new_class: 0 for new_class in new_classes}

    for original_class, new_class in class_mapping.items():
        source_class_dir = source_path / original_class
        target_class_dir = target_path / new_class

        if not source_class_dir.exists():
            print(f"\nWarning: {original_class} directory not found, skipping...")
            continue

        # Get all images
        images = list(source_class_dir.glob("*.jpg")) + \
                 list(source_class_dir.glob("*.jpeg")) + \
                 list(source_class_dir.glob("*.png"))

        print(f"\nProcessing {original_class} -> {new_class}: {len(images)} images")

        # Copy/move images
        for img in tqdm(images, desc=f"  {original_class}"):
            target_file = target_class_dir / img.name

            # Handle name conflicts
            if target_file.exists():
                # Add original class name to avoid conflicts
                target_file = target_class_dir / f"{original_class}_{img.name}"

            if copy:
                shutil.copy2(img, target_file)
            else:
                shutil.move(str(img), str(target_file))

            total_copied += 1
            class_counts[new_class] += 1

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total images processed: {total_copied}")
    print(f"\nFinal class distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:15} : {count:5} images")

    print(f"\nSimplified dataset saved to: {target_dir}")
    print("\nYou can now train with:")
    print(f"  python -m src.model.train --data-dir {target_dir} --num-classes {len(new_classes)}")


def main():
    parser = argparse.ArgumentParser(
        description='Simplify ISIC 7-class dataset to 3-class dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/simplify_classes.py \\
      --source data/raw \\
      --target data/raw_simplified \\
      --copy
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        default='data/raw',
        help='Source directory with original classes'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='data/raw_simplified',
        help='Target directory for simplified classes'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving (default: copy)'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying'
    )

    args = parser.parse_args()

    # Determine copy vs move
    copy_files = not args.move

    simplify_classes(
        source_dir=args.source,
        target_dir=args.target,
        copy=copy_files
    )


if __name__ == "__main__":
    main()
