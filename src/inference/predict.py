"""
Inference Module for Lesion Pre-Screening

Ties together model prediction, risk assessment, and report generation.
Provides end-to-end functionality from image input to PDF report output.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

import torch
import torch.nn.functional as F

from src.model.cnn import create_model
from src.preprocessing.image_processing import ImagePreprocessor, LesionFeatureExtractor, load_image
from src.risk_assessment.scorer import LesionRiskScorer, RiskScore
from src.report.pdf_generator import generate_clinical_report
from src.visualization.gradcam import generate_gradcam_visualization


class LesionPredictor:
    """
    End-to-end lesion pre-screening predictor.

    Handles:
    1. Image loading and preprocessing
    2. CNN inference
    3. Visual feature extraction
    4. Risk assessment
    5. PDF report generation
    """

    def __init__(
        self,
        model_path: str,
        class_names: list,
        device: str = 'cpu'
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model checkpoint
            class_names: List of class names
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessor (no augmentation for inference)
        self.preprocessor = ImagePreprocessor(augment=False)

        # Initialize risk scorer
        self.risk_scorer = LesionRiskScorer(class_names=class_names)

        # Feature extractor
        self.feature_extractor = LesionFeatureExtractor()

        print("Predictor initialized successfully!")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        # Load checkpoint first to determine model type
        checkpoint = torch.load(model_path, map_location=self.device)

        # Try to load as ResNet first (new model), fallback to CNN (old model)
        model_type = 'resnet'  # Default to ResNet50 (current production model)

        try:
            # Create model architecture
            model = create_model(model_type=model_type, num_classes=self.num_classes)

            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            # If ResNet fails, try loading as CNN (backward compatibility)
            print(f"Failed to load as ResNet, trying CNN: {e}")
            model_type = 'cnn'
            model = create_model(model_type=model_type, num_classes=self.num_classes)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        print(f"Model loaded successfully as {model_type}")
        return model

    def predict_image(self, image_path: str, generate_heatmap: bool = True) -> Tuple[np.ndarray, Dict[str, float], Optional[np.ndarray]]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            generate_heatmap: Whether to generate Grad-CAM heatmap

        Returns:
            Tuple of (predictions, visual_features, heatmap_overlay)
            heatmap_overlay is None if generate_heatmap=False
        """
        # Preprocess image
        image_tensor = self.preprocessor.preprocess(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predictions = probabilities.cpu().numpy()[0]

        # Extract visual features
        image_array = load_image(image_path)
        visual_features = self.feature_extractor.extract_all_features(image_array)

        # Generate Grad-CAM heatmap
        heatmap_overlay = None
        if generate_heatmap:
            try:
                predicted_class = np.argmax(predictions)
                _, heatmap_overlay = generate_gradcam_visualization(
                    model=self.model,
                    image_tensor=image_tensor,
                    original_image=image_array,
                    target_class=predicted_class,
                    alpha=0.5
                )
            except Exception as e:
                print(f"Warning: Could not generate heatmap: {e}")
                heatmap_overlay = None

        return predictions, visual_features, heatmap_overlay

    def assess_risk(
        self,
        predictions: np.ndarray,
        visual_features: Dict[str, float]
    ) -> RiskScore:
        """
        Assess risk based on predictions and visual features.

        Args:
            predictions: Model prediction probabilities
            visual_features: Dictionary of visual feature scores

        Returns:
            RiskScore object
        """
        return self.risk_scorer.calculate_comprehensive_risk(
            model_predictions=predictions,
            visual_features=visual_features,
            class_names=self.class_names
        )

    def generate_report(
        self,
        image_path: str,
        risk_score: RiskScore,
        output_path: str,
        patient_id: Optional[str] = None
    ) -> str:
        """
        Generate PDF report.

        Args:
            image_path: Path to input image
            risk_score: RiskScore object
            output_path: Path where PDF will be saved
            patient_id: Optional patient identifier

        Returns:
            Path to generated PDF
        """
        return generate_clinical_report(
            image_path=image_path,
            risk_score=risk_score,
            output_path=output_path,
            patient_id=patient_id
        )

    def process_image(
        self,
        image_path: str,
        output_pdf: Optional[str] = None,
        patient_id: Optional[str] = None,
        verbose: bool = True
    ) -> RiskScore:
        """
        Complete end-to-end processing of a lesion image.

        Args:
            image_path: Path to input image
            output_pdf: Optional path for PDF report
            patient_id: Optional patient identifier
            verbose: Whether to print progress

        Returns:
            RiskScore object with assessment results
        """
        if verbose:
            print("\n" + "="*60)
            print("LESION PRE-SCREENING ANALYSIS")
            print("="*60)
            print(f"Input image: {image_path}")
            print()
            print("⚠️  DISCLAIMER: This system is NOT for medical diagnosis!")
            print("⚠️  Always consult a qualified healthcare professional!")
            print("="*60)
            print()

        # Step 1: Run prediction
        if verbose:
            print("Step 1: Running CNN inference...")
        predictions, visual_features = self.predict_image(image_path)

        if verbose:
            print("  Model predictions:")
            for i, (class_name, prob) in enumerate(zip(self.class_names, predictions)):
                print(f"    {class_name}: {prob:.2%}")
            print()

        # Step 2: Assess risk
        if verbose:
            print("Step 2: Assessing risk...")
        risk_score = self.assess_risk(predictions, visual_features)

        if verbose:
            print(f"  Overall Risk Score: {risk_score.overall_score:.1f}/100")
            print(f"  Risk Level: {risk_score.risk_level.value}")
            print(f"  Confidence: {risk_score.confidence:.1%}")
            print()

            print("  Visual Features:")
            for feature, value in visual_features.items():
                print(f"    {feature}: {value:.2f}")
            print()

        # Step 3: Generate report (if requested)
        if output_pdf:
            if verbose:
                print(f"Step 3: Generating PDF report...")
            pdf_path = self.generate_report(
                image_path=image_path,
                risk_score=risk_score,
                output_path=output_pdf,
                patient_id=patient_id
            )
            if verbose:
                print(f"  Report saved to: {pdf_path}")
                print()

        if verbose:
            print("Recommendations:")
            for i, rec in enumerate(risk_score.recommendations, 1):
                print(f"  {i}. {rec}")
            print()
            print("="*60)
            print("Analysis complete!")
            print("="*60)

        return risk_score


def main():
    """Command-line interface for lesion pre-screening."""
    parser = argparse.ArgumentParser(
        description='Lesion Pre-Screening CNN - NOT FOR MEDICAL DIAGNOSIS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT DISCLAIMER:
This system is for pre-screening and documentation purposes only.
It is NOT intended for medical diagnosis.
Always consult a qualified dermatologist or healthcare professional.

Examples:
  # Basic usage
  python -m src.inference.predict --image lesion.jpg --model data/models/best_model.pth

  # Generate PDF report
  python -m src.inference.predict --image lesion.jpg --model data/models/best_model.pth --output report.pdf

  # With patient ID
  python -m src.inference.predict --image lesion.jpg --model data/models/best_model.pth --output report.pdf --patient-id "12345"
        """
    )

    parser.add_argument('--image', type=str, required=True, help='Path to lesion image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, help='Output path for PDF report (optional)')
    parser.add_argument('--patient-id', type=str, help='Patient identifier (optional)')
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['benign', 'suspicious', 'malignant'],
        help='Class names (space-separated)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on'
    )
    parser.add_argument('--quiet', action='store_true', help='Suppress output messages')

    args = parser.parse_args()

    # Verify input image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Verify model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    try:
        # Create predictor
        predictor = LesionPredictor(
            model_path=args.model,
            class_names=args.classes,
            device=args.device
        )

        # Process image
        risk_score = predictor.process_image(
            image_path=args.image,
            output_pdf=args.output,
            patient_id=args.patient_id,
            verbose=not args.quiet
        )

        # Print summary to stdout (for programmatic use)
        if args.quiet:
            print(f"Risk Score: {risk_score.overall_score:.1f}")
            print(f"Risk Level: {risk_score.risk_level.value}")
            print(f"Confidence: {risk_score.confidence:.2f}")

    except Exception as e:
        print(f"Error during processing: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
