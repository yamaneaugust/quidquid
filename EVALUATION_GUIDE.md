# How to Evaluate Your Model

This guide will help you run a comprehensive evaluation of your ResNet50 model with advanced metrics beyond simple accuracy.

---

## Prerequisites

Before running evaluation, you need:

1. **Trained Model**: Will be auto-downloaded from Google Drive (or use existing `data/models/best_model.pth`)
2. **Validation Data**: A folder with images organized by class
3. **Python 3.8+**: With required packages installed

---

## Step 1: Prepare Your Validation Data

Your validation data should be organized like this:

```
validation_data/
├── benign/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── suspicious/
│   ├── image1.jpg
│   └── ...
└── malignant/
    ├── image1.jpg
    └── ...
```

### Option A: Create a Validation Split

If you have all your data in `data/raw_simplified/`, you should create a train/validation split:

**Windows (PowerShell):**
```powershell
# Create a script to split your data (20% for validation)
python -c "
from pathlib import Path
import shutil
import random

src_dir = Path('data/raw_simplified')
train_dir = Path('data/train')
val_dir = Path('data/val')

for class_name in ['benign', 'suspicious', 'malignant']:
    # Get all images
    images = list((src_dir / class_name).glob('*.jpg'))
    random.shuffle(images)

    # Split 80/20
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Create directories
    (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    (val_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(img, train_dir / class_name / img.name)
    for img in val_images:
        shutil.copy(img, val_dir / class_name / img.name)

    print(f'{class_name}: {len(train_images)} train, {len(val_images)} val')
"
```

**Linux/Mac:**
```bash
python3 -c "
from pathlib import Path
import shutil
import random

src_dir = Path('data/raw_simplified')
train_dir = Path('data/train')
val_dir = Path('data/val')

for class_name in ['benign', 'suspicious', 'malignant']:
    images = list((src_dir / class_name).glob('*.jpg'))
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    for img in train_images:
        shutil.copy(img, train_dir / class_name / img.name)
    for img in val_images:
        shutil.copy(img, val_dir / class_name / img.name)
    print(f'{class_name}: {len(train_images)} train, {len(val_images)} val')
"
```

### Option B: Use Your Existing Split

If you already have a validation set, just note its path (e.g., `data/val` or `C:\Users\yaman\...\validation`)

---

## Step 2: Install Dependencies

Make sure you have the required packages:

**Windows:**
```powershell
pip install gdown matplotlib seaborn scikit-learn tqdm torch torchvision
```

**Linux/Mac:**
```bash
pip3 install gdown matplotlib seaborn scikit-learn tqdm torch torchvision
```

---

## Step 3: Run Evaluation

### Option A: Using the Automated Script (Easiest)

**Windows:**
```powershell
# Replace with your actual validation data path
.\evaluate_windows.bat "data\val"

# Or if your data is elsewhere:
.\evaluate_windows.bat "C:\Users\yaman\OneDrive\デスクトップ\modium\data\val"
```

**Linux/Mac:**
```bash
# Make script executable
chmod +x evaluate.sh

# Run evaluation
./evaluate.sh data/val
```

### Option B: Using Python Directly

**Windows:**
```powershell
# Download model and run evaluation
python setup_and_evaluate.py --data "data\val" --output "reports\evaluation"

# If model already downloaded, skip download step
python setup_and_evaluate.py --data "data\val" --skip-download
```

**Linux/Mac:**
```bash
python3 setup_and_evaluate.py --data data/val --output reports/evaluation
```

### Option C: Manual Steps (Most Control)

**1. Download the model (if needed):**
```bash
pip install gdown
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1zZvamTQyUpMY9lT6HIrBp7OHm-i243FG', 'data/models/best_model.pth')"
```

**2. Run evaluation:**
```bash
python evaluate_model.py \
  --model data/models/best_model.pth \
  --data data/val \
  --output reports/evaluation \
  --model-type resnet \
  --batch-size 32
```

---

## Step 4: Review Results

After evaluation completes, you'll find these files in `reports/evaluation/`:

### 1. **evaluation_results.json**
Complete metrics in JSON format:
```json
{
  "accuracy": 0.8656,
  "balanced_accuracy": 0.7234,
  "macro_f1": 0.7891,
  "class_metrics": {
    "malignant": {
      "precision": 0.78,
      "recall": 0.85,
      "f1_score": 0.81,
      "roc_auc": 0.92,
      "pr_auc": 0.76,
      "sensitivity": 0.85,
      "specificity": 0.95
    },
    ...
  }
}
```

### 2. **Confusion Matrix** (`confusion_matrix.png`)
Shows where the model makes mistakes:
- Diagonal = correct predictions
- Off-diagonal = misclassifications

**Look for:**
- Are malignant lesions being misclassified as benign? (critical!)
- Is one class dominating predictions?

### 3. **ROC Curves** (`roc_curves.png`)
Shows model's discriminative ability:
- Higher AUC = better
- > 0.90 = excellent
- > 0.80 = good
- < 0.70 = needs improvement

**Look for:**
- Is malignant ROC AUC high enough (target: > 0.85)?

### 4. **Precision-Recall Curves** (`precision_recall_curves.png`)
**More important than ROC for imbalanced data!**
- Shows precision vs recall trade-off
- Higher AUC = better
- PR AUC < ROC AUC indicates class imbalance

**Look for:**
- Is PR AUC << ROC AUC? → Class imbalance problem
- Is malignant PR AUC acceptable (target: > 0.75)?

### 5. **Class Distribution** (`class_distribution.png`)
Shows how many samples per class:
- If severely imbalanced (e.g., 10:1 ratio), accuracy is misleading
- Need to use weighted loss or focal loss in training

### 6. **Console Output**
The script also prints:
- Overall metrics summary
- Per-class metrics table
- Clinical interpretation for malignant detection
- Threshold analysis
- Diagnostic insights

---

## Step 5: Interpret Results

### Key Questions to Answer

1. **Is the model biased?**
   - Compare: Accuracy vs Balanced Accuracy
   - If gap > 10%: Model is biased toward majority class
   - **Solution:** Retrain with class weights

2. **Is malignant sensitivity high enough?**
   - Check: `class_metrics.malignant.recall`
   - **Target:** > 90% (to catch most malignant cases)
   - If < 80%: Critical issue - too many missed cases
   - **Solution:** Lower threshold or retrain with class weights

3. **Are there class imbalance issues?**
   - Compare: PR AUC vs ROC AUC
   - If PR AUC << ROC AUC (gap > 0.15): Imbalance problem
   - Check class distribution plot
   - **Solution:** Use weighted loss or focal loss

4. **Where is the model failing?**
   - Look at confusion matrix
   - Are specific classes always confused?
   - **Solution:** May need more training data or better augmentation

### Example Analysis

**Hypothetical Results:**
```
Overall Accuracy:        86.56%
Balanced Accuracy:       68.23%
Macro F1:                70.12%

Malignant Class:
  Sensitivity:           75%  ← Only catching 75% of malignant!
  Specificity:           95%
  ROC AUC:               0.91
  PR AUC:                0.67  ← Much lower than ROC AUC

Class Distribution:
  Benign:      800 (73%)  ← Heavy imbalance
  Suspicious:  150 (14%)
  Malignant:   150 (13%)
```

**Diagnosis:**
- ✗ **Class imbalance** (73% benign vs 13% malignant)
- ✗ **Low balanced accuracy** (68% vs 87% overall) → Biased toward benign
- ✗ **Low malignant sensitivity** (75%) → Missing 25% of cancer cases!
- ✗ **PR AUC << ROC AUC** → Imbalance affecting performance

**Recommended Actions:**
1. Retrain with `--class-weights balanced`
2. Target malignant sensitivity > 90%
3. Monitor PR AUC instead of ROC AUC
4. Consider lowering threshold for malignant detection

---

## Step 6: Take Action

Based on evaluation results:

### If Class Imbalance Detected

**Retrain with balanced weights:**
```bash
python src/model/train.py \
  --data-dir data/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --class-weights balanced \
  --save-dir data/models/balanced
```

### If Malignant Sensitivity Too Low

**Retrain with focal loss:**
```bash
python src/model/train.py \
  --data-dir data/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --focal-loss \
  --save-dir data/models/focal
```

### If Everything Looks Good

Congratulations! Your model is effective. Consider:
1. Deploying to production
2. Monitoring performance over time
3. Collecting more edge cases for continuous improvement

---

## Troubleshooting

### "Model file not found"
- Check if download completed: `ls data/models/best_model.pth`
- Try manual download from Google Drive
- Check internet connection

### "gdown error" or "Download failed"
- Install gdown: `pip install gdown`
- Check Google Drive link is public
- Try manual download: https://drive.google.com/file/d/1zZvamTQyUpMY9lT6HIrBp7OHm-i243FG/view

### "Data directory not found"
- Check path is correct (use absolute path if needed)
- Ensure directory structure matches expected format
- Verify class folders: benign/, suspicious/, malignant/

### "CUDA out of memory"
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use CPU instead: `--device cpu`
- Close other GPU applications

### "ModuleNotFoundError"
- Install missing packages:
  ```bash
  pip install torch torchvision matplotlib seaborn scikit-learn tqdm gdown
  ```

---

## Next Steps

1. **Review Results**: Look at all visualizations and metrics
2. **Read Documentation**: See `MODEL_IMPROVEMENTS.md` for detailed explanations
3. **Compare Models**: If you retrain, compare old vs new models
4. **Deploy Best Model**: Use the model with highest malignant sensitivity
5. **Monitor Production**: Track metrics over time

---

## Quick Reference

**Evaluate model:**
```bash
# Windows
.\evaluate_windows.bat "data\val"

# Linux/Mac
./evaluate.sh data/val
```

**Retrain with class balancing:**
```bash
python src/model/train.py --data-dir data/train --class-weights balanced
```

**Compare models:**
```bash
python evaluate_model.py --model data/models/baseline/best_model.pth --output reports/baseline
python evaluate_model.py --model data/models/balanced/best_model.pth --output reports/balanced
```

**Key metrics to watch:**
- Malignant Sensitivity (target: > 90%)
- Balanced Accuracy (should be close to overall accuracy)
- PR AUC for malignant (target: > 0.75)

---

## Additional Resources

- **MODEL_IMPROVEMENTS.md**: Comprehensive guide on improving effectiveness
- **QUICK_REFERENCE.md**: Quick command reference
- **TRAINING_ROUND_2.md**: Training guide for new models

---

**Need help?** Check the troubleshooting section or review the documentation files.

Good luck with your evaluation! 🎯
