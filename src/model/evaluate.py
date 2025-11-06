"""
Comprehensive model evaluation with clinical-relevant metrics.

This module provides detailed evaluation beyond simple accuracy, including:
- ROC AUC and PR AUC for each class
- Sensitivity, Specificity, PPV, NPV
- Confusion matrices
- Class-wise performance metrics
- Threshold optimization for clinical use cases
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    precision: float
    recall: float  # Also called sensitivity/TPR
    f1_score: float
    specificity: float  # TNR
    npv: float  # Negative Predictive Value
    ppv: float  # Positive Predictive Value (same as precision)
    support: int  # Number of true instances
    roc_auc: float
    pr_auc: float  # Area under Precision-Recall curve


@dataclass
class EvaluationResults:
    """Complete evaluation results for multi-class classification."""
    # Overall metrics
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float

    # Per-class metrics
    class_metrics: Dict[str, ClassMetrics]

    # Multi-class ROC AUC (one-vs-rest)
    macro_roc_auc: float
    weighted_roc_auc: float

    # Confusion matrix
    confusion_matrix: np.ndarray

    # Class distribution
    class_distribution: Dict[str, int]

    # Raw predictions for further analysis
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray  # Probability scores for each class

    def to_dict(self) -> Dict:
        """Convert to dictionary, handling numpy arrays."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        result['confusion_matrix'] = result['confusion_matrix'].tolist()
        result['y_true'] = result['y_true'].tolist()
        result['y_pred'] = result['y_pred'].tolist()
        result['y_proba'] = result['y_proba'].tolist()
        return result

    def save_json(self, filepath: Path):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ModelEvaluator:
    """
    Comprehensive model evaluation for medical image classification.

    Focuses on metrics relevant to clinical decision-making, where false negatives
    (missing malignant lesions) are typically more critical than false positives.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)

    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = True,
    ) -> EvaluationResults:
        """
        Perform comprehensive evaluation on a dataset.

        Args:
            dataloader: DataLoader for evaluation dataset
            return_predictions: Whether to return raw predictions

        Returns:
            EvaluationResults with all metrics
        """
        self.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []

        print("Collecting predictions...")
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Get predictions
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        # Concatenate all batches
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)

        # Calculate all metrics
        print("Calculating metrics...")
        results = self._calculate_metrics(y_true, y_pred, y_proba)

        return results

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> EvaluationResults:
        """Calculate all evaluation metrics."""

        # Overall accuracy
        accuracy = np.mean(y_true == y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Class distribution
        class_dist = {
            name: int(np.sum(y_true == i))
            for i, name in enumerate(self.class_names)
        }

        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            metrics = self._calculate_class_metrics(
                y_true, y_pred, y_proba[:, i], class_idx=i
            )
            class_metrics[class_name] = metrics

        # Calculate balanced accuracy (average of per-class recalls)
        recalls = [m.recall for m in class_metrics.values()]
        balanced_accuracy = np.mean(recalls)

        # Multi-class F1 scores
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        # Multi-class ROC AUC (one-vs-rest)
        try:
            # Binarize labels for multi-class ROC AUC
            macro_roc_auc = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
            weighted_roc_auc = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted'
            )
        except ValueError as e:
            print(f"Warning: Could not calculate multi-class ROC AUC: {e}")
            macro_roc_auc = 0.0
            weighted_roc_auc = 0.0

        return EvaluationResults(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            class_metrics=class_metrics,
            macro_roc_auc=macro_roc_auc,
            weighted_roc_auc=weighted_roc_auc,
            confusion_matrix=cm,
            class_distribution=class_dist,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    def _calculate_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        class_idx: int,
    ) -> ClassMetrics:
        """Calculate metrics for a single class (one-vs-rest)."""

        # Binarize: current class vs all others
        y_true_binary = (y_true == class_idx).astype(int)
        y_pred_binary = (y_pred == class_idx).astype(int)

        # True/False Positives/Negatives
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        # Calculate metrics with zero-division handling
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        support = int(np.sum(y_true_binary))

        # ROC AUC
        try:
            roc_auc = roc_auc_score(y_true_binary, y_proba)
        except ValueError:
            roc_auc = 0.0

        # Precision-Recall AUC
        try:
            pr_auc = average_precision_score(y_true_binary, y_proba)
        except ValueError:
            pr_auc = 0.0

        return ClassMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            npv=npv,
            ppv=precision,  # PPV is the same as precision
            support=support,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
        )

    def plot_confusion_matrix(
        self,
        results: EvaluationResults,
        save_path: Optional[Path] = None,
        normalize: bool = False,
    ):
        """
        Plot confusion matrix with proper labels.

        Args:
            results: Evaluation results
            save_path: Path to save figure
            normalize: Whether to normalize by true labels (show percentages)
        """
        cm = results.confusion_matrix

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_roc_curves(
        self,
        results: EvaluationResults,
        save_path: Optional[Path] = None,
    ):
        """
        Plot ROC curves for each class (one-vs-rest).

        Args:
            results: Evaluation results
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))

        for i, class_name in enumerate(self.class_names):
            # Binarize labels for this class
            y_true_binary = (results.y_true == i).astype(int)
            y_proba_class = results.y_proba[:, i]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba_class)
            roc_auc = results.class_metrics[class_name].roc_auc

            plt.plot(
                fpr, tpr,
                label=f'{class_name} (AUC = {roc_auc:.3f})',
                linewidth=2
            )

        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_precision_recall_curves(
        self,
        results: EvaluationResults,
        save_path: Optional[Path] = None,
    ):
        """
        Plot Precision-Recall curves for each class.

        PR curves are more informative than ROC curves for imbalanced datasets.

        Args:
            results: Evaluation results
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 8))

        for i, class_name in enumerate(self.class_names):
            # Binarize labels for this class
            y_true_binary = (results.y_true == i).astype(int)
            y_proba_class = results.y_proba[:, i]

            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_class)
            pr_auc = results.class_metrics[class_name].pr_auc

            # Baseline (random classifier)
            baseline = np.sum(y_true_binary) / len(y_true_binary)

            plt.plot(
                recall, precision,
                label=f'{class_name} (AP = {pr_auc:.3f})',
                linewidth=2
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision (PPV)', fontsize=12)
        plt.title('Precision-Recall Curves (One-vs-Rest)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curves saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_class_distribution(
        self,
        results: EvaluationResults,
        save_path: Optional[Path] = None,
    ):
        """
        Plot class distribution to visualize imbalance.

        Args:
            results: Evaluation results
            save_path: Path to save figure
        """
        plt.figure(figsize=(10, 6))

        classes = list(results.class_distribution.keys())
        counts = list(results.class_distribution.values())
        total = sum(counts)

        bars = plt.bar(classes, counts, color=['#2ecc71', '#f39c12', '#e74c3c'])

        # Add count and percentage labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = 100 * count / total
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        plt.ylabel('Number of Samples', fontsize=12)
        plt.xlabel('Class', fontsize=12)
        plt.title('Class Distribution in Dataset', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self, results: EvaluationResults):
        """Print a human-readable summary of evaluation results."""

        print("\n" + "="*80)
        print(" MODEL EVALUATION SUMMARY")
        print("="*80)

        print("\n📊 OVERALL METRICS:")
        print(f"  Accuracy:              {results.accuracy:.4f} ({results.accuracy*100:.2f}%)")
        print(f"  Balanced Accuracy:     {results.balanced_accuracy:.4f} ({results.balanced_accuracy*100:.2f}%)")
        print(f"  Macro F1-Score:        {results.macro_f1:.4f}")
        print(f"  Weighted F1-Score:     {results.weighted_f1:.4f}")
        print(f"  Macro ROC AUC:         {results.macro_roc_auc:.4f}")
        print(f"  Weighted ROC AUC:      {results.weighted_roc_auc:.4f}")

        print("\n📈 CLASS DISTRIBUTION:")
        total_samples = sum(results.class_distribution.values())
        for class_name, count in results.class_distribution.items():
            percentage = 100 * count / total_samples
            print(f"  {class_name:15s}: {count:5d} samples ({percentage:5.2f}%)")

        print("\n🎯 PER-CLASS METRICS:")
        print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC AUC':<12} {'PR AUC':<12} {'Specificity':<12}")
        print("-" * 95)

        for class_name, metrics in results.class_metrics.items():
            print(
                f"{class_name:<15} "
                f"{metrics.precision:<12.4f} "
                f"{metrics.recall:<12.4f} "
                f"{metrics.f1_score:<12.4f} "
                f"{metrics.roc_auc:<12.4f} "
                f"{metrics.pr_auc:<12.4f} "
                f"{metrics.specificity:<12.4f}"
            )

        print("\n🏥 CLINICAL METRICS (Sensitivity/Specificity):")
        print(f"\n{'Class':<15} {'Sensitivity':<15} {'Specificity':<15} {'PPV':<15} {'NPV':<15}")
        print("-" * 75)

        for class_name, metrics in results.class_metrics.items():
            print(
                f"{class_name:<15} "
                f"{metrics.recall:<15.4f} "  # Recall = Sensitivity
                f"{metrics.specificity:<15.4f} "
                f"{metrics.ppv:<15.4f} "
                f"{metrics.npv:<15.4f}"
            )

        print("\n" + "="*80)

        # Clinical interpretation for malignant class
        if 'malignant' in results.class_metrics:
            mal_metrics = results.class_metrics['malignant']
            print("\n⚕️  CLINICAL INTERPRETATION (Malignant Detection):")
            print(f"  • Sensitivity: {mal_metrics.recall:.2%} - Catches {mal_metrics.recall:.1%} of malignant lesions")
            print(f"  • Specificity: {mal_metrics.specificity:.2%} - Correctly identifies {mal_metrics.specificity:.1%} of non-malignant")
            print(f"  • PPV: {mal_metrics.ppv:.2%} - {mal_metrics.ppv:.1%} of 'malignant' predictions are correct")
            print(f"  • NPV: {mal_metrics.npv:.2%} - {mal_metrics.npv:.1%} of 'benign/suspicious' predictions are correct")
            print("="*80 + "\n")


def analyze_threshold_impact(
    evaluator: ModelEvaluator,
    results: EvaluationResults,
    target_class_idx: int = 2,  # Default: malignant
    thresholds: np.ndarray = None,
) -> Dict:
    """
    Analyze how different probability thresholds affect clinical metrics.

    Useful for understanding the trade-off between sensitivity and specificity,
    which is crucial for clinical decision-making.

    Args:
        evaluator: ModelEvaluator instance
        results: Evaluation results
        target_class_idx: Index of the class to analyze (e.g., malignant = 2)
        thresholds: Array of thresholds to test (default: 0.1 to 0.9)

    Returns:
        Dictionary with threshold analysis results
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)

    class_name = evaluator.class_names[target_class_idx]
    y_true_binary = (results.y_true == target_class_idx).astype(int)
    y_proba = results.y_proba[:, target_class_idx]

    threshold_analysis = []

    for threshold in thresholds:
        # Make predictions with this threshold
        y_pred_binary = (y_proba >= threshold).astype(int)

        # Calculate metrics
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        threshold_analysis.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        })

    return {
        'class_name': class_name,
        'analysis': threshold_analysis,
    }


def calculate_class_weights(
    dataset_path: Path,
    class_names: List[str],
    method: str = 'balanced',
) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced datasets.

    Args:
        dataset_path: Path to dataset directory
        class_names: List of class names
        method: 'balanced' (inverse frequency) or 'effective' (effective number of samples)

    Returns:
        Tensor of class weights
    """
    # Count samples per class
    class_counts = []
    for class_name in class_names:
        class_dir = dataset_path / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')))
            class_counts.append(count)
        else:
            class_counts.append(0)

    class_counts = np.array(class_counts)
    total_samples = np.sum(class_counts)

    if method == 'balanced':
        # Inverse frequency weighting
        weights = total_samples / (len(class_names) * class_counts)
    elif method == 'effective':
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / np.sum(weights) * len(class_names)
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"\n📊 Class weights ({method}):")
    for name, count, weight in zip(class_names, class_counts, weights):
        print(f"  {name:15s}: {count:5d} samples -> weight = {weight:.4f}")

    return torch.FloatTensor(weights)


if __name__ == '__main__':
    """Example usage and testing."""
    print("Model Evaluation Module")
    print("=" * 80)
    print("\nThis module provides comprehensive evaluation for lesion classification models.")
    print("\nKey features:")
    print("  • ROC AUC and PR AUC for each class")
    print("  • Sensitivity, Specificity, PPV, NPV")
    print("  • Confusion matrices and visualizations")
    print("  • Threshold analysis for clinical decision-making")
    print("  • Class imbalance handling with weighted loss")
    print("\nSee evaluate_model.py for usage examples.")
