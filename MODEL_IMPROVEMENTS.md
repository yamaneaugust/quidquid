# Model Effectiveness Improvements

## Executive Summary

Your ResNet50 model achieves **86.56% accuracy**, but accuracy alone is insufficient for medical screening. This document provides a comprehensive guide to improving model **effectiveness** through better metrics, class balancing, and threshold optimization.

---

## Table of Contents

1. [Why Accuracy Is Not Enough](#why-accuracy-is-not-enough)
2. [Key Problems with Current Approach](#key-problems-with-current-approach)
3. [Recommended Metrics](#recommended-metrics)
4. [Implementation Guide](#implementation-guide)
5. [Class Imbalance Solutions](#class-imbalance-solutions)
6. [Threshold Optimization](#threshold-optimization)
7. [Quick Start Guide](#quick-start-guide)

---

## Why Accuracy Is Not Enough

### The Accuracy Paradox

For medical screening with imbalanced data, **high accuracy can be misleading**:

**Example:**
- Dataset: 1000 images (900 benign, 50 suspicious, 50 malignant)
- A model that always predicts "benign" would achieve **90% accuracy**!
- But it would miss **100% of malignant lesions** (catastrophic for patients)

### Clinical Context Matters

In lesion screening:
- **False Negative** (missing cancer) → Patient dies
- **False Positive** (unnecessary biopsy) → Patient inconvenience

These have vastly different clinical costs, which accuracy doesn't capture.

---

## Key Problems with Current Approach

### 1. Class Imbalance

ISIC datasets typically have severe imbalance:
```
Benign:     ~70-80% of samples
Suspicious: ~10-15% of samples
Malignant:  ~5-15% of samples
```

**Impact:**
- Model biased toward majority class (benign)
- Poor performance on rare but critical classes (malignant)
- Accuracy inflated by correct benign predictions

### 2. Limited Metrics

Currently tracking only:
- ✓ Accuracy
- ✓ Loss

**Missing critical metrics:**
- ✗ Sensitivity/Recall (catching malignant cases)
- ✗ Specificity (avoiding false alarms)
- ✗ ROC AUC (overall discriminative ability)
- ✗ PR AUC (performance on imbalanced data)
- ✗ Per-class performance

### 3. No Class Weighting

Using standard CrossEntropyLoss without weights:
```python
criterion = nn.CrossEntropyLoss()  # All classes equal
```

**Problem:** Model treats missing a malignant case the same as misclassifying benign.

### 4. Fixed Threshold

Using default 0.5 probability threshold:
- May be suboptimal for clinical use case
- Doesn't account for cost asymmetry
- One-size-fits-all approach

---

## Recommended Metrics

### 1. **ROC AUC (Receiver Operating Characteristic)**

**What it measures:** Model's ability to discriminate between classes across all thresholds

**Interpretation:**
- 1.0 = Perfect classifier
- 0.5 = Random guessing
- > 0.8 = Good performance

**Why it matters:** Shows overall model quality independent of threshold choice

**Limitation:** Can be optimistic for imbalanced data

### 2. **PR AUC (Precision-Recall Area Under Curve)**

**What it measures:** Trade-off between precision and recall across thresholds

**Interpretation:**
- 1.0 = Perfect precision and recall
- Higher is better
- More informative than ROC AUC for imbalanced data

**Why it matters:** Better reflects performance on rare classes (e.g., malignant)

### 3. **Sensitivity / Recall / TPR**

**Formula:** `TP / (TP + FN)`

**What it measures:** Percentage of actual malignant cases correctly identified

**Example:**
- 50 malignant lesions in test set
- Model catches 45 → Sensitivity = 90%
- Misses 5 → Could be fatal

**Clinical priority:** **HIGH** - Missing cancer is catastrophic

### 4. **Specificity / TNR**

**Formula:** `TN / (TN + FP)`

**What it measures:** Percentage of benign cases correctly identified

**Example:**
- 900 benign lesions
- 850 correctly classified as benign → Specificity = 94.4%
- 50 false alarms → Unnecessary biopsies

**Clinical priority:** **MODERATE** - False positives are costly but not fatal

### 5. **Positive Predictive Value (PPV) / Precision**

**Formula:** `TP / (TP + FP)`

**What it measures:** When model says "malignant", how often is it correct?

**Example:**
- Model flags 100 lesions as malignant
- 45 actually malignant → PPV = 45%
- 55 false alarms

**Clinical impact:** Affects biopsy decision confidence

### 6. **Negative Predictive Value (NPV)**

**Formula:** `TN / (TN + FN)`

**What it measures:** When model says "benign", how often is it correct?

**Example:**
- Model says 850 lesions are benign
- 845 actually benign → NPV = 99.4%
- 5 missed malignant cases

**Clinical impact:** Reassurance value for "clear" results

### 7. **Balanced Accuracy**

**Formula:** `(Sensitivity_class1 + Sensitivity_class2 + ... ) / num_classes`

**What it measures:** Average per-class accuracy

**Why it matters:** Accounts for class imbalance

**Example:**
- Overall accuracy: 87%
- Balanced accuracy: 65%
- → Model biased toward majority class!

---

## Implementation Guide

### Step 1: Evaluate Current Model

Run comprehensive evaluation on your existing model:

```bash
python evaluate_model.py \
  --model data/models/best_model.pth \
  --data data/processed/val \
  --output reports/evaluation
```

**Output:**
- `evaluation_results.json` - All metrics in JSON format
- `confusion_matrix.png` - Visual breakdown of predictions
- `confusion_matrix_normalized.png` - Percentage-based view
- `roc_curves.png` - ROC curves for each class
- `precision_recall_curves.png` - PR curves for each class
- `class_distribution.png` - Class imbalance visualization

**What to look for:**
1. **Confusion Matrix:** Where is the model making mistakes?
2. **Per-Class Metrics:** Is malignant detection sensitivity high enough?
3. **ROC vs PR AUC:** Large gap indicates imbalance issues
4. **Balanced Accuracy:** Much lower than accuracy? Model is biased.

### Step 2: Retrain with Class Balancing

#### Option A: Balanced Class Weights (Recommended First)

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

**How it works:**
- Automatically calculates weights: `weight_i = total_samples / (num_classes × class_i_samples)`
- Applies higher loss penalty for misclassifying rare classes
- Encourages model to focus on minority classes

**Expected improvement:**
- Better balanced accuracy
- Higher sensitivity for malignant class
- May slightly reduce overall accuracy (acceptable trade-off)

#### Option B: Effective Number Class Weights (For Severe Imbalance)

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

**How it works:**
- Uses "Effective Number of Samples" from Class-Balanced Loss paper
- More aggressive reweighting: `weight_i = (1 - β) / (1 - β^n_i)`
- Better for extreme imbalance (e.g., 1:100 ratio)

#### Option C: Focal Loss (For Very Hard Examples)

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

**How it works:**
- Down-weights easy examples (high confidence correct predictions)
- Focuses training on hard misclassified examples
- Formula: `FL(p_t) = -α(1 - p_t)^γ log(p_t)`

**Parameters:**
- `alpha`: Weight for positive class (0.25 = 25% weight)
- `gamma`: Focusing parameter (2.0 standard, higher = more focus on hard examples)

**When to use:** When model gets most examples right but struggles with specific hard cases

### Step 3: Compare Models

Evaluate all trained models:

```bash
# Baseline (no balancing)
python evaluate_model.py \
  --model data/models/best_model.pth \
  --data data/processed/val \
  --output reports/baseline

# Balanced weights
python evaluate_model.py \
  --model data/models/balanced/best_model.pth \
  --data data/processed/val \
  --output reports/balanced

# Effective weights
python evaluate_model.py \
  --model data/models/effective/best_model.pth \
  --data data/processed/val \
  --output reports/effective

# Focal loss
python evaluate_model.py \
  --model data/models/focal/best_model.pth \
  --data data/processed/val \
  --output reports/focal
```

**Comparison criteria:**

| Metric | Priority | Target |
|--------|----------|--------|
| **Malignant Sensitivity** | 🔴 Critical | > 90% |
| **Malignant PR AUC** | 🔴 Critical | > 0.80 |
| **Balanced Accuracy** | 🟡 Important | > 75% |
| **Malignant Specificity** | 🟡 Important | > 80% |
| **Overall Accuracy** | 🟢 Nice-to-have | > 80% |

**Decision matrix:**
- If sensitivity < 90%: Use more aggressive balancing (effective or focal loss)
- If false positives too high: Optimize threshold (next section)
- If overall performance poor: Check data quality, augmentation, or try different architecture

---

## Class Imbalance Solutions

### Summary Table

| Method | Pros | Cons | When to Use |
|--------|------|------|-------------|
| **No Weighting** | Simple, fast convergence | Biased toward majority class | Balanced datasets only |
| **Balanced Weights** | Easy to implement, works well | May overweight minorities | Moderate imbalance (1:5 - 1:10) |
| **Effective Weights** | Handles severe imbalance | More sensitive to hyperparameters | Severe imbalance (> 1:10) |
| **Focal Loss** | Focuses on hard examples | Requires tuning α and γ | Hard misclassified examples |
| **Oversampling** | More training data for minorities | Risk of overfitting | Small minority class |
| **Undersampling** | Faster training | Wastes majority class data | Large datasets |

### Advanced Techniques (Not Yet Implemented)

#### 1. **Data Augmentation (Already Implemented)**

Current augmentation (in `ImagePreprocessor`):
- ✓ Random flips
- ✓ Rotation
- ✓ Color jitter
- ✓ Affine transforms

**Consider adding:**
- Cutout/Random erasing
- Mixup/CutMix
- Advanced domain-specific augmentation (e.g., hair removal simulation)

#### 2. **Oversampling Minority Classes**

Use `WeightedRandomSampler` to oversample rare classes:

```python
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
class_counts = Counter([label for _, label in dataset.samples])
sample_weights = [1.0 / class_counts[label] for _, label in dataset.samples]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

# Use in DataLoader
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

#### 3. **SMOTE (Synthetic Minority Oversampling)**

Generate synthetic examples for minority class in feature space:
- Requires extracting CNN features first
- Apply SMOTE in latent space
- More sophisticated than simple oversampling

---

## Threshold Optimization

### Why Threshold Matters

Default classification: `predicted_class = argmax(softmax(logits))`

For binary decisions (e.g., malignant vs not):
```python
is_malignant = proba[malignant_class] > threshold  # Default: 0.5
```

**Problem:** 0.5 may not be optimal for your clinical use case!

### Sensitivity-Specificity Trade-off

```
Lower Threshold (e.g., 0.3)
├─ ✓ Higher Sensitivity (catch more malignant)
├─ ✗ Lower Specificity (more false positives)
└─ Use case: Screening (don't miss cancer)

Higher Threshold (e.g., 0.7)
├─ ✗ Lower Sensitivity (miss some malignant)
├─ ✓ Higher Specificity (fewer false positives)
└─ Use case: Confirmatory test (high precision)
```

### Finding Optimal Threshold

The evaluation script includes threshold analysis:

```python
from src.model.evaluate import analyze_threshold_impact

threshold_results = analyze_threshold_impact(
    evaluator=evaluator,
    results=results,
    target_class_idx=2,  # malignant
)
```

**Example output:**
```
Threshold    Sensitivity    Specificity    Precision    FP    FN
0.10         0.9800         0.7500         0.3500       250   1
0.30         0.9400         0.8800         0.5200       120   3
0.50         0.8600         0.9500         0.7100       50    7
0.70         0.7200         0.9800         0.8500       20    14
```

**How to choose:**

1. **Screening Application** (prioritize sensitivity):
   - Target: Sensitivity ≥ 95%
   - Choose threshold where FN is minimized
   - Example: threshold = 0.30 → sensitivity 94%, only 3 missed cases

2. **Triage Application** (balance):
   - Maximize F1-score = `2 × (Precision × Recall) / (Precision + Recall)`
   - Example: threshold = 0.50 → balanced performance

3. **Confirmatory Application** (prioritize specificity):
   - Target: Specificity ≥ 95%
   - Choose threshold where FP is minimized
   - Example: threshold = 0.70 → specificity 98%, low false alarm rate

### Implementing Custom Threshold

Update your prediction code:

```python
# In src/inference/predict.py
def predict_with_threshold(self, image_path: str, threshold: float = 0.5):
    # Get probabilities
    probabilities = self.predict_image(image_path)[0]

    # Binary decision for malignant
    is_malignant = probabilities[2] > threshold  # Class 2 = malignant

    return {
        'is_malignant': is_malignant,
        'probabilities': probabilities,
        'threshold': threshold,
    }
```

---

## Quick Start Guide

### 1. Evaluate Current Model (5 minutes)

```bash
python evaluate_model.py \
  --model data/models/best_model.pth \
  --data data/processed/val
```

**Review:**
- Check malignant class sensitivity (target: > 90%)
- Check balanced accuracy vs accuracy gap
- Review confusion matrix for systematic errors

### 2. Identify Main Issue

**If balanced accuracy << accuracy:**
→ Class imbalance problem → Use balanced weights

**If malignant sensitivity < 80%:**
→ Missing critical cases → Use effective weights or focal loss

**If high confusion between suspicious/malignant:**
→ Hard examples → Try focal loss

**If dataset has > 1:10 imbalance:**
→ Severe imbalance → Use effective weights

### 3. Retrain with Appropriate Method

**Most common case (moderate imbalance):**
```bash
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --class-weights balanced
```

**Severe imbalance:**
```bash
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --class-weights effective
```

**Hard examples:**
```bash
python src/model/train.py \
  --data-dir data/processed/train \
  --model-type resnet \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.0001 \
  --focal-loss
```

### 4. Re-evaluate and Compare

```bash
python evaluate_model.py \
  --model data/models/best_model_balanced.pth \
  --data data/processed/val \
  --output reports/balanced
```

**Success criteria:**
- ✓ Malignant sensitivity > 90%
- ✓ Balanced accuracy improved by > 5%
- ✓ PR AUC for malignant > 0.80
- ✓ Acceptable false positive rate (< 20%)

### 5. Optimize Threshold

Review threshold analysis in evaluation output:
- Choose threshold based on clinical priorities
- Document decision in deployment guide
- Update prediction code with chosen threshold

---

## Expected Improvements

### Before (Current Model)

**Hypothetical metrics based on 87% accuracy:**
```
Overall Accuracy:        87%
Balanced Accuracy:       ~68% (estimated)
Malignant Sensitivity:   ~75% (estimated)
Malignant PR AUC:        ~0.65 (estimated)
```

**Problem:** Missing 25% of malignant cases!

### After (With Class Balancing)

**Expected metrics with balanced weights:**
```
Overall Accuracy:        82-85% (slight decrease)
Balanced Accuracy:       75-80% (significant increase)
Malignant Sensitivity:   85-92% (major improvement)
Malignant PR AUC:        0.75-0.85 (improvement)
```

**Benefit:** Catching 85-92% of malignant cases, missing only 8-15%

### Clinical Impact

**Scenario:** 1000 patient screening cohort (50 with malignant lesions)

| Model | Sensitivity | Caught | Missed | Clinical Outcome |
|-------|-------------|--------|--------|------------------|
| **Baseline** | 75% | 37-38 | 12-13 | 12-13 preventable deaths |
| **Balanced** | 90% | 45 | 5 | 5 preventable deaths |
| **Optimized** | 95% | 47-48 | 2-3 | 2-3 preventable deaths |

**Result:** **60-75% reduction in missed malignant cases**

---

## Additional Resources

### Papers and References

1. **Focal Loss for Dense Object Detection** (Lin et al., 2017)
   - https://arxiv.org/abs/1708.02002
   - Introduces focal loss for imbalanced classification

2. **Class-Balanced Loss Based on Effective Number of Samples** (Cui et al., 2019)
   - https://arxiv.org/abs/1901.05555
   - Effective number weighting method

3. **The Precision-Recall Plot Is More Informative than the ROC Plot** (Saito & Rehmsmeier, 2015)
   - Why PR curves matter for imbalanced data

4. **Learning from Imbalanced Data** (He & Garcia, 2009)
   - Comprehensive survey of imbalance handling techniques

### Tools and Libraries

- **scikit-learn**: Metrics, ROC curves, PR curves
- **imbalanced-learn**: SMOTE, advanced sampling techniques
- **tensorboard**: Track metrics during training
- **wandb**: Experiment tracking and comparison

---

## Monitoring in Production

### Key Metrics to Track

1. **Per-Class Performance:**
   - Monitor sensitivity for malignant class weekly
   - Alert if drops below threshold (e.g., < 85%)

2. **Confusion Matrix:**
   - Track systematic error patterns
   - Identify if new lesion types appear

3. **Calibration:**
   - Are predicted probabilities reliable?
   - Use calibration plots (reliability diagrams)

4. **Data Drift:**
   - Monitor if input distribution changes
   - Retrain if dataset shifts significantly

### A/B Testing Framework

When deploying new model:
1. Run both old and new model in parallel
2. Compare metrics on same test set
3. Require statistically significant improvement
4. Gradual rollout (10% → 50% → 100%)

---

## Conclusion

**Key Takeaways:**

1. ✅ **Accuracy is insufficient** for medical screening with imbalanced data
2. ✅ **Use comprehensive metrics**: ROC AUC, PR AUC, sensitivity, specificity
3. ✅ **Handle class imbalance**: Weighted loss or focal loss
4. ✅ **Optimize threshold** based on clinical priorities (sensitivity for screening)
5. ✅ **Evaluate properly**: Confusion matrix, per-class metrics, threshold analysis

**Next Steps:**

1. Run `evaluate_model.py` on current model
2. Identify main performance bottleneck
3. Retrain with appropriate class balancing
4. Compare results and iterate
5. Optimize threshold for deployment
6. Document final model configuration

**Remember:** In medical AI, **effectiveness > accuracy**. A 82% accurate model that catches 95% of malignant cases is far better than a 90% accurate model that only catches 70%.

---

## Questions?

If you encounter issues:
1. Check evaluation outputs in `reports/` directory
2. Review confusion matrix for systematic errors
3. Compare metrics across different training configurations
4. Ensure validation set is representative of deployment population

Good luck improving your model! 🎯
