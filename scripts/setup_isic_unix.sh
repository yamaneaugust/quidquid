#!/bin/bash
# Unix/Linux/Mac script to set up ISIC dataset
# Run this after downloading ISIC2018_Task3_Training_Input.zip and CSV

echo "========================================"
echo "ISIC Dataset Setup for Unix/Linux/Mac"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org or your package manager"
    exit 1
fi

echo "Step 1: Activating virtual environment..."
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "ERROR: Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

echo ""
echo "Step 2: Extracting training images..."
DOWNLOADS="$HOME/Downloads"
ZIP_FILE="$DOWNLOADS/ISIC2018_Task3_Training_Input.zip"
EXTRACT_DIR="$DOWNLOADS/extracted"

if [ -f "$ZIP_FILE" ]; then
    python scripts/organize_isic_data.py --extract "$ZIP_FILE" --extract-to "$EXTRACT_DIR"
else
    echo "WARNING: $ZIP_FILE not found"
    echo "Please download it from: https://challenge.isic-archive.com/data/"
    echo ""
fi

echo ""
echo "Step 3: Organizing images into class folders..."
IMAGES_DIR="$EXTRACT_DIR/ISIC2018_Task3_Training_Input"
CSV_FILE="$DOWNLOADS/ISIC2018_Task3_Training_GroundTruth.csv"

if [ -f "$CSV_FILE" ]; then
    python scripts/organize_isic_data.py \
        --images-dir "$IMAGES_DIR" \
        --groundtruth-csv "$CSV_FILE" \
        --output-dir data/raw \
        --task3-format
else
    echo "WARNING: $CSV_FILE not found"
    echo "Please download it from: https://challenge.isic-archive.com/data/"
    echo ""
fi

echo ""
echo "Step 4: (Optional) Simplify to 3 classes..."
read -p "Do you want to simplify from 7 to 3 classes? (y/n): " SIMPLIFY

if [[ "$SIMPLIFY" == "y" ]] || [[ "$SIMPLIFY" == "Y" ]]; then
    python scripts/simplify_classes.py \
        --source data/raw \
        --target data/raw_simplified \
        --copy
    echo ""
    echo "Dataset simplified! You can now train with:"
    echo "  python -m src.model.train --data-dir data/raw_simplified --num-classes 3"
else
    echo ""
    echo "You can train with all 7 classes:"
    echo "  python -m src.model.train --data-dir data/raw --num-classes 7"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Verify data in data/raw/ folder"
echo "  2. Train your model (see commands above)"
echo "  3. Run inference with: python -m src.inference.predict"
echo ""
