#!/usr/bin/env python3
"""
Download model and run evaluation.

This script:
1. Downloads the trained model from Google Drive
2. Prepares validation data
3. Runs comprehensive evaluation
"""

import os
import sys
from pathlib import Path
import gdown

# Configuration
GDRIVE_FILE_ID = "1zZvamTQyUpMY9lT6HIrBp7OHm-i243FG"  # ResNet50 model (86.56% accuracy)
MODEL_PATH = "data/models/best_model.pth"


def download_model():
    """Download model from Google Drive if not present."""
    model_path = Path(MODEL_PATH)

    if model_path.exists():
        print(f"✓ Model already exists at {MODEL_PATH}")
        return True

    print(f"Downloading model from Google Drive...")
    print(f"File ID: {GDRIVE_FILE_ID}")

    try:
        # Create directory if needed
        os.makedirs(model_path.parent, exist_ok=True)

        # Download from Google Drive
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, str(model_path), quiet=False)

        print(f"✓ Model downloaded successfully to {MODEL_PATH}")
        return True

    except Exception as e:
        print(f"✗ Failed to download model: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have internet connection")
        print("2. gdown is installed: pip install gdown")
        print("3. Google Drive file is accessible")
        return False


def check_validation_data(data_path: str):
    """Check if validation data exists."""
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"\n✗ Data directory not found: {data_path}")
        print("\nPlease prepare validation data with this structure:")
        print(f"{data_path}/")
        print("  ├── benign/")
        print("  │   ├── image1.jpg")
        print("  │   ├── image2.jpg")
        print("  │   └── ...")
        print("  ├── suspicious/")
        print("  │   └── ...")
        print("  └── malignant/")
        print("      └── ...")
        return False

    # Check for class directories
    class_names = ['benign', 'suspicious', 'malignant']
    found_classes = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        if class_dir.exists() and class_dir.is_dir():
            num_images = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
            found_classes.append(class_name)
            print(f"✓ Found {num_images} images in {class_name}/")
        else:
            print(f"✗ Missing class directory: {class_name}/")

    if len(found_classes) < 3:
        print("\n✗ Not all class directories found")
        return False

    print(f"\n✓ Validation data looks good!")
    return True


def run_evaluation(data_path: str, output_dir: str = "reports/evaluation"):
    """Run the evaluation script."""
    import subprocess

    cmd = [
        sys.executable,
        "evaluate_model.py",
        "--model", MODEL_PATH,
        "--data", data_path,
        "--output", output_dir,
        "--model-type", "resnet",
        "--batch-size", "32",
    ]

    print(f"\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Evaluation failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Could not find Python or evaluation script")
        return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download model and run evaluation")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to validation data directory (e.g., data/raw_simplified or path to validation split)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip model download (if already downloaded)"
    )

    args = parser.parse_args()

    print("="*80)
    print("MODEL EVALUATION SETUP")
    print("="*80)

    # Step 1: Download model
    if not args.skip_download:
        print("\n[1/3] Downloading model from Google Drive...")
        if not download_model():
            print("\n✗ Setup failed at model download step")
            return 1
    else:
        print("\n[1/3] Skipping model download (--skip-download)")

    # Step 2: Check validation data
    print(f"\n[2/3] Checking validation data at: {args.data}")
    if not check_validation_data(args.data):
        print("\n✗ Setup failed at data validation step")
        return 1

    # Step 3: Run evaluation
    print(f"\n[3/3] Running comprehensive evaluation...")
    if not run_evaluation(args.data, args.output):
        print("\n✗ Evaluation failed")
        return 1

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.output}/")
    print("\nGenerated files:")
    print(f"  • evaluation_results.json - All metrics in JSON format")
    print(f"  • confusion_matrix.png - Confusion matrix visualization")
    print(f"  • confusion_matrix_normalized.png - Normalized confusion matrix")
    print(f"  • roc_curves.png - ROC curves for each class")
    print(f"  • precision_recall_curves.png - Precision-Recall curves")
    print(f"  • class_distribution.png - Class distribution visualization")
    print("\nNext steps:")
    print("  1. Review the metrics and visualizations")
    print("  2. Check MODEL_IMPROVEMENTS.md for recommendations")
    print("  3. If needed, retrain with class balancing")

    return 0


if __name__ == "__main__":
    sys.exit(main())
