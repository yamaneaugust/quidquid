# Getting Started Guide

This guide will walk you through setting up and using the Lesion Pre-Screening CNN system.

## ⚠️ Critical Disclaimer

**THIS SYSTEM IS NOT FOR MEDICAL DIAGNOSIS**

This tool is designed for:
- Educational purposes
- Pre-screening and documentation
- Risk quantification
- Research

Always consult qualified medical professionals for diagnosis and treatment.

---

## Step 1: Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended but not required)
- At least 8GB RAM
- 10GB free disk space

### Install Dependencies

```bash
# Clone or navigate to the repository
cd quidquid

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

---

## Step 2: Prepare Your Dataset

### Dataset Structure

Organize your lesion images in the following structure:

```
data/raw/
├── benign/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── suspicious/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── malignant/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

### Dataset Requirements

- **Format**: JPG, JPEG, or PNG
- **Size**: At least 224x224 pixels (higher is better)
- **Quality**: Clear, well-lit images
- **Quantity**: Minimum 500+ images per class (more is better)
- **Distribution**: Balanced across classes if possible

### Public Datasets (for educational use)

Consider these publicly available datasets:
- **HAM10000**: 10,000+ dermatoscopic images
- **ISIC Archive**: International Skin Imaging Collaboration
- **PAD-UFES-20**: Smartphone images of skin lesions

**Important**: Always respect data licensing and patient privacy!

---

## Step 3: Train the Model

### Basic Training

```bash
python -m src.model.train \
    --data-dir data/raw \
    --epochs 50 \
    --batch-size 32 \
    --model-type cnn
```

### Advanced Training Options

```bash
python -m src.model.train \
    --data-dir data/raw \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001 \
    --model-type resnet \
    --device cuda \
    --save-dir data/models
```

### Training Parameters

- `--data-dir`: Path to your dataset
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--model-type`: Model architecture: `cnn` or `resnet`
- `--device`: `cuda` or `cpu`
- `--val-split`: Validation split ratio (default: 0.2)

### What to Expect

Training on a GPU typically takes:
- Small dataset (1000 images): 10-20 minutes
- Medium dataset (5000 images): 1-2 hours
- Large dataset (10,000+ images): 3-5 hours

You'll see output like:
```
Epoch 1/50
Train Loss: 1.0234, Train Acc: 45.32%
Val Loss: 0.9876, Val Acc: 52.11%
Saved best model!

Epoch 2/50
...
```

---

## Step 4: Run Inference

### Single Image Analysis

```bash
python -m src.inference.predict \
    --image path/to/lesion.jpg \
    --model data/models/best_model.pth \
    --output report.pdf
```

### Generate Report with Patient ID

```bash
python -m src.inference.predict \
    --image path/to/lesion.jpg \
    --model data/models/best_model.pth \
    --output reports/patient_12345.pdf \
    --patient-id "12345"
```

### Inference Parameters

- `--image`: Path to lesion image (required)
- `--model`: Path to trained model (required)
- `--output`: Output PDF path (optional)
- `--patient-id`: Patient identifier (optional)
- `--classes`: Class names (space-separated)
- `--device`: `cuda` or `cpu`
- `--quiet`: Suppress verbose output

---

## Step 5: Interpreting Results

### Risk Score Components

The system provides several metrics:

1. **Overall Risk Score (0-100)**
   - 0-25: Low risk
   - 25-50: Moderate risk
   - 50-75: High risk
   - 75-100: Very high risk

2. **Model Confidence**
   - How certain the model is about its prediction
   - Higher confidence (>80%) is generally more reliable

3. **Visual Features (ABCDE Criteria)**
   - **A**symmetry: Is the lesion asymmetric?
   - **B**order irregularity: Are the borders irregular?
   - **C**olor variation: Multiple colors present?
   - **D**iameter: Would need scale reference
   - **E**volution: Would need historical comparison

### Example Output

```
Risk Assessment Results:
Overall Score: 65.5/100
Risk Level: High
Confidence: 85%

Recommendations:
  1. This system is NOT intended for medical diagnosis...
  2. HIGH RISK - Seek professional medical evaluation soon.
  3. Schedule dermatologist appointment within 1-2 weeks.
```

---

## Step 6: Understanding the PDF Report

The generated PDF includes:

1. **Header**: Date, disclaimers, patient ID
2. **Lesion Image**: The analyzed image
3. **Risk Assessment**: Overall score and level
4. **Risk Factors**: Detailed breakdown
5. **Recommendations**: Suggested actions
6. **Footer**: Important notes and system information

---

## Using the System Programmatically

### Python API Example

```python
from src.inference.predict import LesionPredictor

# Initialize predictor
predictor = LesionPredictor(
    model_path='data/models/best_model.pth',
    class_names=['benign', 'suspicious', 'malignant'],
    device='cuda'
)

# Process an image
risk_score = predictor.process_image(
    image_path='lesion.jpg',
    output_pdf='report.pdf',
    patient_id='12345'
)

# Access results
print(f"Risk Score: {risk_score.overall_score}")
print(f"Risk Level: {risk_score.risk_level.value}")
print(f"Confidence: {risk_score.confidence}")
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch size: `--batch-size 16`
- Use CPU: `--device cpu`

**2. Poor model performance**
- Need more training data
- Try data augmentation (enabled by default)
- Increase epochs: `--epochs 100`
- Try different model: `--model-type resnet`

**3. Low confidence predictions**
- Model needs more training
- Image quality issues
- Lesion not well-centered in image

**4. Import errors**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

---

## Best Practices

### Data Collection
1. Use consistent lighting and image quality
2. Ensure lesion is centered and in focus
3. Include diverse examples for each class
4. Maintain patient privacy (de-identify images)

### Model Training
1. Start with a baseline model (CNN)
2. Monitor validation loss (watch for overfitting)
3. Use early stopping
4. Save multiple checkpoints
5. Document training parameters

### Inference and Reporting
1. Always include proper disclaimers
2. Never present results as definitive diagnoses
3. Encourage professional medical consultation
4. Keep detailed logs of all predictions
5. Regularly validate model performance

---

## Next Steps

1. **Experiment with Notebooks**: See `notebooks/` for examples
2. **Customize the Model**: Modify architecture in `src/model/cnn.py`
3. **Adjust Risk Scoring**: Tune weights in `src/risk_assessment/scorer.py`
4. **Customize Reports**: Modify PDF template in `src/report/pdf_generator.py`

---

## Ethical and Legal Considerations

### Medical Device Regulations
- This system may require regulatory approval (FDA, CE, etc.) for clinical use
- Consult legal experts before deploying in healthcare settings

### Data Privacy
- Comply with HIPAA, GDPR, and other privacy regulations
- Implement proper data encryption and access controls
- De-identify patient information

### Bias and Fairness
- Ensure training data represents diverse populations
- Regularly audit for algorithmic bias
- Monitor performance across different demographics

### Transparency
- Clearly communicate system limitations
- Provide explanations for predictions
- Enable human oversight and intervention

---

## Support and Resources

- **Documentation**: See project README.md
- **Issues**: Report bugs and feature requests
- **Research**: See references for melanoma detection papers
- **Community**: Medical imaging and AI forums

---

## References

For educational purposes, refer to:
- ABCDE criteria for melanoma detection
- Dermoscopy guidelines
- Deep learning for medical image analysis papers
- Clinical dermatology textbooks

**Remember**: This system is a tool to assist, not replace, medical professionals.
