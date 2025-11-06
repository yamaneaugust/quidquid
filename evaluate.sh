#!/bin/bash
# Linux/Mac shell script to evaluate the model
# Usage: ./evaluate.sh <path_to_validation_data>

set -e  # Exit on error

echo "==============================================================================="
echo "Model Evaluation Script"
echo "==============================================================================="
echo ""

# Check if data path provided
if [ -z "$1" ]; then
    echo "ERROR: Please provide the path to your validation data"
    echo ""
    echo "Usage:"
    echo "  ./evaluate.sh /path/to/validation/data"
    echo ""
    echo "Example:"
    echo "  ./evaluate.sh data/validation"
    echo ""
    echo "Your validation data should have this structure:"
    echo "  validation_data/"
    echo "    benign/"
    echo "      image1.jpg"
    echo "      image2.jpg"
    echo "    suspicious/"
    echo "      image1.jpg"
    echo "    malignant/"
    echo "      image1.jpg"
    echo ""
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ from your package manager"
    exit 1
fi

echo "[1/2] Installing required packages..."
pip3 install gdown matplotlib seaborn scikit-learn tqdm

echo ""
echo "[2/2] Running evaluation..."
python3 setup_and_evaluate.py --data "$1" --output reports/evaluation

echo ""
echo "==============================================================================="
echo "SUCCESS! Evaluation completed"
echo "==============================================================================="
echo ""
echo "Results are saved in: reports/evaluation/"
echo ""
echo "Open these files to view results:"
echo "  - reports/evaluation/confusion_matrix.png"
echo "  - reports/evaluation/roc_curves.png"
echo "  - reports/evaluation/precision_recall_curves.png"
echo "  - reports/evaluation/evaluation_results.json"
echo ""
echo "To view detailed recommendations, read: MODEL_IMPROVEMENTS.md"
echo ""
