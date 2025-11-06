#!/usr/bin/env python3
"""
Evaluate trained model with comprehensive metrics.

This script loads a trained model and evaluates it on the validation/test set,
generating detailed metrics, visualizations, and reports.

Usage:
    python evaluate_model.py --model data/models/best_model.pth --data data/processed/val
    python evaluate_model.py --model data/models/best_model.pth --data data/processed/val --output reports/evaluation
"""

import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.model.cnn import create_model
from src.model.train import LesionDataset
from src.model.evaluate import ModelEvaluator, analyze_threshold_impact
from src.preprocessing.image_processing import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained lesion classification model')
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/val',
        help='Path to validation/test data directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/evaluation',
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='resnet',
        choices=['cnn', 'resnet'],
        help='Model architecture type'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Class names
    class_names = ['benign', 'suspicious', 'malignant']

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = create_model(model_type=args.model_type, num_classes=len(class_names))

    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")

    # Load dataset
    print(f"\nLoading data from: {args.data}")
    preprocessor = ImagePreprocessor(augment=False)  # No augmentation for evaluation

    dataset = LesionDataset(
        data_dir=args.data,
        preprocessor=preprocessor,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"✓ Loaded {len(dataset)} samples")

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=class_names,
    )

    # Run evaluation
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*80)

    results = evaluator.evaluate(dataloader)

    # Print summary
    evaluator.print_summary(results)

    # Save results to JSON
    results_file = output_dir / 'evaluation_results.json'
    results.save_json(results_file)
    print(f"\n✓ Results saved to: {results_file}")

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Confusion matrix
    print("\n1. Confusion Matrix...")
    evaluator.plot_confusion_matrix(
        results,
        save_path=output_dir / 'confusion_matrix.png',
        normalize=False
    )
    evaluator.plot_confusion_matrix(
        results,
        save_path=output_dir / 'confusion_matrix_normalized.png',
        normalize=True
    )

    # ROC curves
    print("\n2. ROC Curves...")
    evaluator.plot_roc_curves(
        results,
        save_path=output_dir / 'roc_curves.png'
    )

    # Precision-Recall curves
    print("\n3. Precision-Recall Curves...")
    evaluator.plot_precision_recall_curves(
        results,
        save_path=output_dir / 'precision_recall_curves.png'
    )

    # Class distribution
    print("\n4. Class Distribution...")
    evaluator.plot_class_distribution(
        results,
        save_path=output_dir / 'class_distribution.png'
    )

    # Threshold analysis for malignant class
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS (Malignant Class)")
    print("="*80)

    threshold_results = analyze_threshold_impact(
        evaluator=evaluator,
        results=results,
        target_class_idx=2,  # malignant
    )

    print(f"\nAnalyzing different decision thresholds for '{threshold_results['class_name']}' class:")
    print(f"\n{'Threshold':<12} {'Sensitivity':<15} {'Specificity':<15} {'Precision':<15} {'FP':<8} {'FN':<8}")
    print("-" * 85)

    for item in threshold_results['analysis']:
        print(
            f"{item['threshold']:<12.2f} "
            f"{item['sensitivity']:<15.4f} "
            f"{item['specificity']:<15.4f} "
            f"{item['precision']:<15.4f} "
            f"{item['fp']:<8d} "
            f"{item['fn']:<8d}"
        )

    print("\n💡 CLINICAL INTERPRETATION:")
    print("  • Lower threshold → Higher sensitivity (catch more malignant) but more false positives")
    print("  • Higher threshold → Higher specificity (fewer false alarms) but may miss malignant cases")
    print("  • For screening: Prioritize HIGH SENSITIVITY to avoid missing malignant lesions")
    print("  • For triage: Balance sensitivity and specificity based on clinical workflow")

    # Check for concerning patterns
    print("\n" + "="*80)
    print("DIAGNOSTIC INSIGHTS")
    print("="*80)

    # Check class imbalance
    total_samples = sum(results.class_distribution.values())
    imbalance_ratios = {
        name: total_samples / count
        for name, count in results.class_distribution.items()
    }

    max_imbalance = max(imbalance_ratios.values())
    if max_imbalance > 3:
        print("\n⚠️  CLASS IMBALANCE DETECTED:")
        for name, ratio in imbalance_ratios.items():
            print(f"  {name}: 1:{ratio:.1f} (imbalance ratio)")
        print("\n  Recommendations:")
        print("  1. Use class weights in loss function (see IMPROVEMENTS.md)")
        print("  2. Focus on per-class metrics (not just accuracy)")
        print("  3. Consider PR AUC instead of ROC AUC for rare classes")

    # Check if accuracy is misleading
    if results.accuracy > 0.8 and results.balanced_accuracy < 0.7:
        print("\n⚠️  ACCURACY IS MISLEADING:")
        print(f"  • Overall Accuracy: {results.accuracy:.1%}")
        print(f"  • Balanced Accuracy: {results.balanced_accuracy:.1%}")
        print("\n  The model may be biased toward majority class!")
        print("  Use balanced accuracy and per-class metrics instead.")

    # Check malignant detection performance
    if 'malignant' in results.class_metrics:
        mal_metrics = results.class_metrics['malignant']
        if mal_metrics.recall < 0.80:
            print("\n⚠️  LOW SENSITIVITY FOR MALIGNANT CLASS:")
            print(f"  • Sensitivity: {mal_metrics.recall:.1%}")
            print(f"  • Missing {(1-mal_metrics.recall)*100:.1f}% of malignant lesions")
            print("\n  Critical for medical screening! Recommendations:")
            print("  1. Increase weight for malignant class in loss function")
            print("  2. Lower decision threshold to catch more malignant cases")
            print("  3. Use focal loss to focus on hard examples")

    # Check if PR AUC is much lower than ROC AUC (indicates imbalance issues)
    for class_name, metrics in results.class_metrics.items():
        if metrics.roc_auc - metrics.pr_auc > 0.2:
            print(f"\n⚠️  IMBALANCE IMPACT ON '{class_name.upper()}' CLASS:")
            print(f"  • ROC AUC: {metrics.roc_auc:.3f}")
            print(f"  • PR AUC:  {metrics.pr_auc:.3f}")
            print(f"  • Gap:     {metrics.roc_auc - metrics.pr_auc:.3f}")
            print("\n  PR AUC is more informative for imbalanced data.")
            print("  Consider using PR AUC as your primary metric.")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review the visualizations and metrics")
    print("  2. Check IMPROVEMENTS.md for recommendations")
    print("  3. Consider retraining with class weights or focal loss")
    print("  4. Optimize threshold based on clinical priorities\n")


if __name__ == '__main__':
    main()
