# Quick Reference: Model Improvement Commands

## Evaluate Current Model

```bash
# Basic evaluation
python evaluate_model.py

# Custom paths
python evaluate_model.py \
  --model data/models/best_model.pth \
  --data data/processed/val \
  --output reports/evaluation
```

**Outputs:** JSON metrics + visualizations (confusion matrix, ROC curves, PR curves)

---

## Training with Class Balancing

### Option 1: Balanced Weights (Recommended)

```bash
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --class-weights balanced \
  --save-dir data/models/balanced
```

**Best for:** Moderate imbalance (1:5 to 1:10 ratio)

### Option 2: Effective Number Weights

```bash
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --class-weights effective \
  --save-dir data/models/effective
```

**Best for:** Severe imbalance (> 1:10 ratio)

### Option 3: Focal Loss

```bash
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --focal-loss \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --save-dir data/models/focal
```

**Best for:** Hard-to-classify examples

---

## Key Metrics to Watch

| Metric | What It Means | Target |
|--------|---------------|--------|
| **Malignant Sensitivity** | % of malignant lesions caught | **> 90%** |
| **Malignant PR AUC** | Performance on rare class | **> 0.80** |
| **Balanced Accuracy** | Average per-class accuracy | **> 75%** |
| **Specificity** | % of benign correctly identified | **> 80%** |

---

## Decision Tree

```
Is balanced accuracy << overall accuracy?
├─ YES → Class imbalance problem
│   ├─ Is imbalance > 1:10?
│   │   ├─ YES → Use --class-weights effective
│   │   └─ NO → Use --class-weights balanced
│   └─ Is malignant sensitivity < 80%?
│       └─ YES → Try --focal-loss
└─ NO → Other issues (data quality, architecture, augmentation)
```

---

## Comparing Models

```bash
# Evaluate multiple models
for model in baseline balanced effective focal; do
  python evaluate_model.py \
    --model data/models/$model/best_model.pth \
    --data data/processed/val \
    --output reports/$model
done

# Compare results
cat reports/*/evaluation_results.json | grep -E "(accuracy|sensitivity|auc)"
```

---

## Critical Questions to Answer

1. **What's the class distribution?**
   - Check `class_distribution.png` in evaluation output

2. **Where is the model failing?**
   - Check `confusion_matrix.png`
   - Look for systematic errors (e.g., suspicious → benign)

3. **Is malignant sensitivity high enough?**
   - Check JSON output: `class_metrics.malignant.recall`
   - Target: > 0.90

4. **Is PR AUC << ROC AUC?**
   - Large gap (> 0.2) indicates imbalance issues
   - Focus on PR AUC for rare classes

5. **What threshold should I use?**
   - Check threshold analysis in evaluation output
   - For screening: Choose threshold with sensitivity > 95%
   - For triage: Balance sensitivity and specificity

---

## Common Pitfalls

❌ **Don't:** Focus only on overall accuracy
✅ **Do:** Prioritize malignant sensitivity and balanced accuracy

❌ **Don't:** Use default 0.5 threshold without analysis
✅ **Do:** Optimize threshold based on clinical priorities

❌ **Don't:** Ignore class imbalance
✅ **Do:** Use weighted loss or focal loss

❌ **Don't:** Train without validation metrics
✅ **Do:** Track per-class metrics during training

---

## Example Workflow

```bash
# 1. Evaluate baseline
python evaluate_model.py --output reports/baseline

# 2. Check if imbalanced (review class_distribution.png)
# → If yes, proceed to step 3

# 3. Retrain with balanced weights
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --class-weights balanced \
  --save-dir data/models/balanced

# 4. Evaluate balanced model
python evaluate_model.py \
  --model data/models/balanced/best_model.pth \
  --output reports/balanced

# 5. Compare metrics
echo "Baseline metrics:"
cat reports/baseline/evaluation_results.json | grep -A5 "class_metrics"

echo "Balanced metrics:"
cat reports/balanced/evaluation_results.json | grep -A5 "class_metrics"

# 6. Choose best model based on malignant sensitivity
```

---

## File Locations

- **Training script:** `src/model/train.py`
- **Evaluation script:** `evaluate_model.py`
- **Evaluation module:** `src/model/evaluate.py`
- **Saved models:** `data/models/`
- **Evaluation reports:** `reports/`
- **Detailed guide:** `MODEL_IMPROVEMENTS.md`
