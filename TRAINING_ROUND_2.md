# Training Round 2 Setup Guide
## For Running on a Different Device

### Prerequisites
- Python 3.8+
- GPU recommended (CUDA-capable) but CPU works too
- At least 8GB RAM
- 10GB+ free disk space

---

## Step 1: Clone the Repository

```bash
# Clone the repo
git clone https://github.com/yamaneaugust/quidquid.git
cd quidquid

# Switch to your working branch
git checkout claude/cnn-lesion-screening-011CUM48AsuFKri5BSwi1P5r
```

---

## Step 2: Set Up Python Environment

### Option A: Using venv (Recommended)

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Option B: Using conda

```bash
conda create -n modium python=3.10
conda activate modium
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Step 3: Get the Dataset

You have two options:

### Option A: Copy from Your Current Computer

**If you still have the data on your laptop:**

1. Copy the entire `data/raw_simplified/` folder to a USB drive
2. Transfer to new device
3. Place in `quidquid/data/raw_simplified/`

Your folder structure should look like:
```
quidquid/
├── data/
│   └── raw_simplified/
│       ├── benign/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── suspicious/
│       │   └── ...
│       └── malignant/
│           └── ...
```

### Option B: Download Fresh from ISIC

**If you need to download again:**

1. Go to https://challenge.isic-archive.com/data/
2. Download:
   - ISIC2018_Task3_Training_Input.zip
   - ISIC2018_Task3_Training_GroundTruth.zip

3. Extract and organize:
```bash
# From quidquid directory
python scripts/organize_isic_data.py
python scripts/simplify_classes.py
```

---

## Step 4: Training Round 2 - Improved Model

### Option A: Continue Training Your Existing Model

```bash
# Resume training from your best checkpoint
python -m src.model.train \
    --data-dir data/raw_simplified \
    --num-classes 3 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.0001 \
    --device cuda  # or 'cpu' if no GPU
```

### Option B: Train with Transfer Learning (ResNet50) - RECOMMENDED

**This should get you to 85%+ accuracy!**

```bash
python -m src.model.train \
    --data-dir data/raw_simplified \
    --model-type resnet \
    --num-classes 3 \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.0001 \
    --device cuda
```

**Why ResNet is better:**
- Pretrained on ImageNet (1 million images)
- Transfer learning = faster convergence
- Better feature extraction
- Higher accuracy with less training time

**Training will take:**
- With GPU: 2-4 hours
- With CPU: 12-24 hours

---

## Step 5: Monitor Training

Watch the output for:
```
Epoch 1/30
Training: 100%|████████| loss: 0.6432, acc: 72.3%
Validation: loss: 0.5123, acc: 78.5%
Saved best model!

Epoch 2/30
Training: 100%|████████| loss: 0.4821, acc: 78.9%
Validation: loss: 0.4456, acc: 81.2%
Saved best model!
...
```

**What to look for:**
- Validation accuracy improving over epochs
- "Saved best model!" messages
- Early stopping when no improvement

---

## Step 6: After Training

### Test the New Model

```bash
# Test on a single image
python -m src.inference.predict \
    --image path/to/test_image.jpg \
    --model data/models/best_model.pth \
    --class-names benign suspicious malignant
```

### Compare with Old Model

**Your current model:**
- Architecture: Custom CNN
- Accuracy: ~79%
- Training time: ~40 epochs

**New ResNet model (expected):**
- Architecture: ResNet50 (pretrained)
- Accuracy: 85%+ (target)
- Training time: ~20-30 epochs

### Deploy the New Model

**If accuracy is better:**

1. **Update Google Drive:**
   - Upload new `best_model.pth` to Google Drive
   - Get new file ID
   - Update `app.py` line with new Google Drive file ID

2. **Test locally:**
```bash
streamlit run app.py
# Upload image and verify it works
```

3. **Deploy:**
```bash
git add data/models/best_model.pth  # if using Git LFS
git commit -m "Update model to ResNet50 - 85% accuracy"
git push
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
--batch-size 8  # or even 4
```

### "Training is slow"
- Use GPU if available
- Reduce image size (already optimized)
- Use fewer epochs for testing first

### "Validation accuracy not improving"
- Try different learning rate: `--lr 0.00001` or `--lr 0.001`
- Add more data augmentation
- Train longer (more epochs)

### "Model file too large for Git"
Use Git LFS or Google Drive as currently implemented.

---

## Advanced: Experiment with Hyperparameters

Create a training script to try multiple configurations:

```bash
# Try different learning rates
for lr in 0.0001 0.00001 0.001; do
    python -m src.model.train \
        --model-type resnet \
        --lr $lr \
        --save-dir data/models/lr_${lr}
done

# Compare results and pick the best
```

---

## Expected Results

**If everything goes well, you should see:**

```
Training completed!
Best validation accuracy: 86.3%
Models saved to: data/models/

Training history:
- Final train accuracy: 89.1%
- Final validation accuracy: 86.3%
- Total epochs: 27 (early stopping)
```

**This would be a significant improvement from 79% → 86%!**

---

## Next Steps After Training

1. **Document results** - Save training logs
2. **Blog about it** - "How I Improved Model Accuracy from 79% to 86%"
3. **Update app** - Deploy new model
4. **Tell professors** - "I just improved my model to 86% accuracy"
5. **Update Product Hunt** - Post an update about the improvement

---

## Quick Start Commands (Copy-Paste Ready)

**For Windows with GPU:**
```powershell
git clone https://github.com/yamaneaugust/quidquid.git
cd quidquid
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Copy your data to data/raw_simplified/

# Train with ResNet
python -m src.model.train --data-dir data/raw_simplified --model-type resnet --num-classes 3 --epochs 30 --batch-size 16 --lr 0.0001 --device cuda
```

**For Mac/Linux with CPU:**
```bash
git clone https://github.com/yamaneaugust/quidquid.git
cd quidquid
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt

# Copy your data to data/raw_simplified/

# Train with ResNet
python -m src.model.train --data-dir data/raw_simplified --model-type resnet --num-classes 3 --epochs 30 --batch-size 16 --lr 0.0001 --device cpu
```

---

## Tips for Success

1. **Use GPU if possible** - 10x faster training
2. **Start with small test** - Train for 2 epochs first to verify everything works
3. **Monitor progress** - Watch validation accuracy, should improve each epoch
4. **Save frequently** - Model auto-saves, but keep backups
5. **Document everything** - Save terminal output for your blog post

Good luck with training round 2! 🚀
