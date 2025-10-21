# Lesion Pre-Screening CNN System

A Convolutional Neural Network (CNN) based system for lesion pre-screening and documentation.

## ⚠️ IMPORTANT DISCLAIMER

**THIS SYSTEM IS NOT INTENDED FOR MEDICAL DIAGNOSIS**

This tool is designed for:
- Educational purposes
- Risk quantification and documentation
- Assisting in preliminary screening workflows
- Generating clinician-friendly reports

**Always consult qualified medical professionals for diagnosis and treatment decisions.**

## Features

- **CNN-based lesion analysis**: Deep learning model for image classification
- **Risk quantification**: Multi-factor risk scoring system
- **Clinician reports**: Automated PDF generation with detailed findings
- **Data augmentation**: Robust training with augmented datasets
- **Interpretability**: Attention maps and feature visualization

## Project Structure

```
quidquid/
├── src/
│   ├── model/          # CNN architecture and training
│   ├── preprocessing/  # Image processing utilities
│   ├── risk_assessment/# Risk scoring algorithms
│   ├── report/         # PDF report generation
│   └── inference/      # Prediction and inference
├── data/
│   ├── raw/           # Original lesion images
│   ├── processed/     # Preprocessed datasets
│   └── models/        # Trained model checkpoints
├── notebooks/         # Jupyter notebooks for analysis
└── tests/            # Unit and integration tests
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Dataset

```bash
# Organize images in data/raw/
# Structure: data/raw/{class_name}/image.jpg
```

### 2. Train the Model

```bash
python -m src.model.train --data-dir data/raw --epochs 50
```

### 3. Run Inference

```bash
python -m src.inference.predict --image path/to/lesion.jpg --output report.pdf
```

## Dataset Requirements

- **Image format**: JPG, PNG
- **Recommended size**: 224x224 or higher
- **Classes**: Organize by lesion type or risk category
- **Minimum samples**: 500+ images per class (more is better)

## Model Architecture

The system uses a CNN architecture optimized for medical image analysis:
- Input: 224x224x3 RGB images
- Multiple convolutional blocks with batch normalization
- Dropout for regularization
- Softmax output for multi-class classification

## Risk Scoring

The risk assessment module considers:
1. **Model confidence**: Prediction probability
2. **Feature characteristics**: Size, irregularity, color variation
3. **Uncertainty estimation**: Model confidence intervals
4. **Historical patterns**: Population-based risk factors

## PDF Reports

Generated reports include:
- Input image with highlighted regions
- Risk scores and confidence levels
- Model interpretation visualizations
- Recommendations for follow-up
- Clear disclaimer about non-diagnostic use

## Training Tips

1. **Data augmentation**: Enabled by default (rotation, flip, zoom)
2. **Class balancing**: Use weighted loss for imbalanced datasets
3. **Validation split**: 20% held out for validation
4. **Early stopping**: Prevents overfitting
5. **Learning rate scheduling**: Adaptive learning rate

## Contributing

This is a research/educational project. Contributions should maintain:
- Medical safety considerations
- Clear disclaimer messaging
- Code quality and documentation
- Ethical AI practices

## License

[Specify your license]

## Ethical Considerations

This system must be used responsibly:
- Not a replacement for professional medical diagnosis
- Should be validated in clinical settings before any real-world use
- Must comply with medical device regulations (FDA, CE, etc.)
- Patient privacy and data security are paramount
- Bias in training data must be carefully monitored

## Support

For questions or issues, please refer to the documentation or contact the development team.
