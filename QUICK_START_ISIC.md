# Quick Start with ISIC Dataset

You've downloaded ISIC2018 test images. Here's what you need to do:

## Understanding What You Have

**ISIC2018_Task1-2_Test_Input.zip** contains:
- **Unlabeled test images** for evaluating trained models
- These are **NOT suitable for training** (no labels!)

## What You Need for Training

You need **ISIC Training Data WITH Labels**. Here are your options:

---

## Option 1: ISIC 2018 Challenge Dataset (Recommended)

### Download Links:

1. **Training Images**:
   - Go to: https://challenge.isic-archive.com/data/
   - Download: `ISIC2018_Task3_Training_Input.zip` (~10,015 images)

2. **Training Labels**:
   - Download: `ISIC2018_Task3_Training_GroundTruth.csv`
   - Contains 7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC

3. **Validation Images** (Optional):
   - Download: `ISIC2018_Task3_Validation_Input.zip` (~193 images)
   - Download: `ISIC2018_Task3_Validation_GroundTruth.csv`

### Classes in ISIC 2018:
- **MEL**: Melanoma (malignant)
- **NV**: Melanocytic nevus (benign)
- **BCC**: Basal cell carcinoma (malignant)
- **AKIEC**: Actinic keratosis (pre-cancerous)
- **BKL**: Benign keratosis
- **DF**: Dermatofibroma (benign)
- **VASC**: Vascular lesion (benign)

---

## Option 2: HAM10000 Dataset

- **Size**: 10,015 dermatoscopic images
- **Download**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **Format**: Comes with CSV labels
- Same classes as ISIC 2018 (it's the source dataset!)

---

## Step-by-Step Setup

### Step 1: Download Training Data

Download both:
1. `ISIC2018_Task3_Training_Input.zip` (images)
2. `ISIC2018_Task3_Training_GroundTruth.csv` (labels)

Place them in: `C:\Users\yaman\Downloads\`

### Step 2: Extract and Organize

**On Windows**, open PowerShell or Command Prompt:

```powershell
# Navigate to your project
cd path\to\quidquid

# Activate virtual environment
venv\Scripts\activate

# Extract the training images
python scripts/organize_isic_data.py ^
    --extract "C:\Users\yaman\Downloads\ISIC2018_Task3_Training_Input.zip" ^
    --extract-to "C:\Users\yaman\Downloads\extracted"

# Organize into class folders
python scripts/organize_isic_data.py ^
    --images-dir "C:\Users\yaman\Downloads\extracted\ISIC2018_Task3_Training_Input" ^
    --groundtruth-csv "C:\Users\yaman\Downloads\ISIC2018_Task3_Training_GroundTruth.csv" ^
    --output-dir "data\raw" ^
    --task3-format
```

**On Linux/Mac**:

```bash
# Navigate to your project
cd path/to/quidquid

# Activate virtual environment
source venv/bin/activate

# Extract
python scripts/organize_isic_data.py \
    --extract "/home/user/Downloads/ISIC2018_Task3_Training_Input.zip" \
    --extract-to "/home/user/Downloads/extracted"

# Organize
python scripts/organize_isic_data.py \
    --images-dir "/home/user/Downloads/extracted/ISIC2018_Task3_Training_Input" \
    --groundtruth-csv "/home/user/Downloads/ISIC2018_Task3_Training_GroundTruth.csv" \
    --output-dir "data/raw" \
    --task3-format
```

### Step 3: Verify Organization

Your `data/raw/` folder should now look like:

```
data/raw/
├── MEL/          # Melanoma (melanoma)
│   ├── ISIC_0000001.jpg
│   ├── ISIC_0000002.jpg
│   └── ...
├── NV/           # Melanocytic nevus (mole)
│   ├── ISIC_0000010.jpg
│   └── ...
├── BCC/          # Basal cell carcinoma
├── AKIEC/        # Actinic keratosis
├── BKL/          # Benign keratosis
├── DF/           # Dermatofibroma
└── VASC/         # Vascular lesion
```

Check the counts:

```bash
# Linux/Mac
ls data/raw/*/*.jpg | wc -l

# Windows PowerShell
(Get-ChildItem -Path data\raw -Recurse -Filter *.jpg).Count
```

### Step 4: Simplify Classes (Optional)

ISIC has 7 classes, but you might want to simplify to 3:
- **Benign**: NV, BKL, DF, VASC
- **Suspicious**: AKIEC
- **Malignant**: MEL, BCC

Create a script to merge classes if needed.

### Step 5: Train Your Model

```bash
# Train with all 7 classes
python -m src.model.train \
    --data-dir data/raw \
    --epochs 50 \
    --batch-size 32 \
    --num-classes 7

# Or specify class names
python -m src.model.train \
    --data-dir data/raw \
    --epochs 50 \
    --batch-size 32 \
    --num-classes 7
```

Expected training time:
- With GPU: 2-3 hours
- With CPU: 8-12 hours

### Step 6: Use Your Test Images

Now you can use the test images you downloaded for inference:

```bash
# Extract test images
unzip "C:\Users\yaman\Downloads\ISIC2018_Task1-2_Test_Input.zip" -d test_images

# Run inference on a test image
python -m src.inference.predict \
    --image test_images/ISIC_0034524.jpg \
    --model data/models/best_model.pth \
    --output reports/test_report.pdf \
    --classes MEL NV BCC AKIEC BKL DF VASC
```

---

## Alternative: Start Small with Sample Data

If downloads are taking too long, you can start with a small subset:

1. Manually create folders:
   ```
   data/raw/benign/
   data/raw/suspicious/
   data/raw/malignant/
   ```

2. Find ~50-100 sample images online or use personal photos (de-identified!)

3. Train a small proof-of-concept model

4. Later scale up with full ISIC dataset

---

## Dataset Class Mapping for Training

If you want to use 3 classes instead of 7:

```python
# Create this as scripts/simplify_classes.py
import shutil
from pathlib import Path

source_dir = Path("data/raw")
target_dir = Path("data/raw_simplified")

# Class mapping
class_mapping = {
    'MEL': 'malignant',
    'BCC': 'malignant',
    'AKIEC': 'suspicious',
    'NV': 'benign',
    'BKL': 'benign',
    'DF': 'benign',
    'VASC': 'benign',
}

for original_class, new_class in class_mapping.items():
    source_class_dir = source_dir / original_class
    target_class_dir = target_dir / new_class
    target_class_dir.mkdir(parents=True, exist_ok=True)

    if source_class_dir.exists():
        for img in source_class_dir.glob("*.jpg"):
            shutil.copy2(img, target_class_dir / img.name)
        print(f"Copied {original_class} -> {new_class}")

print("\nDone! Train with:")
print("python -m src.model.train --data-dir data/raw_simplified --num-classes 3")
```

---

## Troubleshooting

### "CSV file not found"
- Make sure you downloaded the GroundTruth CSV file
- Check the filename matches exactly

### "No images copied"
- Verify the extracted folder path
- Check that images are `.jpg` files
- Ensure paths don't have spaces or special characters

### "Out of memory during training"
- Reduce batch size: `--batch-size 16` or `--batch-size 8`
- Use CPU instead: `--device cpu`

### "Class imbalance warning"
- ISIC dataset is imbalanced (mostly NV/benign)
- The training script handles this automatically with weighted loss

---

## Next Steps After Organizing Data

1. ✅ Verify dataset structure
2. ✅ Check class distribution
3. ✅ Start training
4. ✅ Monitor training progress
5. ✅ Evaluate on validation set
6. ✅ Run inference on test images
7. ✅ Generate PDF reports

---

## Important Notes

- **Ethics**: ISIC data is for research/education only
- **Privacy**: Images are de-identified, keep them secure
- **Licenses**: Respect ISIC data usage terms
- **Medical Use**: Remember - NOT for clinical diagnosis!

---

## Quick Reference Commands

```bash
# 1. Extract training images
python scripts/organize_isic_data.py --extract <ZIP> --extract-to <DIR>

# 2. Organize with labels
python scripts/organize_isic_data.py \
    --images-dir <IMAGES> \
    --groundtruth-csv <CSV> \
    --output-dir data/raw \
    --task3-format

# 3. Train model
python -m src.model.train --data-dir data/raw --epochs 50

# 4. Run inference
python -m src.inference.predict --image <IMG> --model <MODEL> --output <PDF>
```

---

Need help? Check the main GETTING_STARTED.md or README.md!
