"""
Risk Assessment and Scoring System

This module quantifies lesion risk based on CNN predictions,
visual features, and clinical guidelines.

NOT FOR MEDICAL DIAGNOSIS - Educational and documentation purposes only.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level categories."""
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


@dataclass
class RiskScore:
    """
    Container for risk assessment results.

    Attributes:
        overall_score: Overall risk score (0-100)
        risk_level: Categorical risk level
        confidence: Model confidence (0-1)
        factors: Dictionary of individual risk factors
        recommendations: List of recommended actions
    """
    overall_score: float
    risk_level: RiskLevel
    confidence: float
    factors: Dict[str, float]
    recommendations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'overall_score': round(self.overall_score, 2),
            'risk_level': self.risk_level.value,
            'confidence': round(self.confidence, 2),
            'factors': {k: round(v, 2) for k, v in self.factors.items()},
            'recommendations': self.recommendations
        }


class LesionRiskScorer:
    """
    Calculates comprehensive risk scores for lesions.

    The risk score is based on:
    1. CNN model predictions (class probabilities)
    2. Visual features (ABCDE criteria: Asymmetry, Border, Color, Diameter, Evolution)
    3. Confidence/uncertainty estimation
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize the risk scorer.

        Args:
            class_names: List of class names corresponding to model outputs
        """
        self.class_names = class_names or ["benign", "suspicious", "malignant"]

        # Risk weights for different classes (0-1 scale)
        # These should be calibrated based on clinical data
        self.class_risk_weights = {
            "benign": 0.1,
            "suspicious": 0.5,
            "malignant": 0.9,
            "melanoma": 1.0,
            "basal_cell": 0.7,
            "squamous_cell": 0.8,
            "nevus": 0.2,
        }

    def calculate_model_risk(
        self,
        predictions: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> Tuple[float, float]:
        """
        Calculate risk score from model predictions.

        Args:
            predictions: Model output probabilities (softmax)
            class_names: Optional class names (uses self.class_names if None)

        Returns:
            Tuple of (risk_score, confidence)
        """
        if class_names is None:
            class_names = self.class_names

        # Weighted risk score based on class probabilities
        risk_score = 0.0
        for prob, class_name in zip(predictions, class_names):
            weight = self.class_risk_weights.get(class_name.lower(), 0.5)
            risk_score += prob * weight

        # Confidence is the max probability (how certain the model is)
        confidence = float(np.max(predictions))

        return risk_score, confidence

    def calculate_visual_risk(self, visual_features: Dict[str, float]) -> float:
        """
        Calculate risk score from visual features (ABCDE criteria).

        Args:
            visual_features: Dictionary of feature name to value (0-1 scale)
                Expected keys: asymmetry, border_irregularity, color_variation

        Returns:
            Visual risk score (0-1)
        """
        # Weights for ABCDE criteria
        weights = {
            'asymmetry': 0.3,
            'border_irregularity': 0.3,
            'color_variation': 0.4,
        }

        visual_risk = 0.0
        for feature, value in visual_features.items():
            weight = weights.get(feature, 0.0)
            visual_risk += value * weight

        return visual_risk

    def calculate_comprehensive_risk(
        self,
        model_predictions: np.ndarray,
        visual_features: Dict[str, float],
        class_names: Optional[List[str]] = None
    ) -> RiskScore:
        """
        Calculate comprehensive risk score combining all factors.

        Args:
            model_predictions: Model output probabilities
            visual_features: Dictionary of visual feature scores
            class_names: Optional class names

        Returns:
            RiskScore object with detailed assessment
        """
        # Calculate model-based risk
        model_risk, confidence = self.calculate_model_risk(model_predictions, class_names)

        # Calculate visual feature risk
        visual_risk = self.calculate_visual_risk(visual_features)

        # Combine risks (weighted average)
        # Model prediction gets more weight if confidence is high
        model_weight = 0.6 + 0.2 * confidence  # 0.6 to 0.8
        visual_weight = 1.0 - model_weight

        overall_risk = model_risk * model_weight + visual_risk * visual_weight

        # Convert to 0-100 scale
        overall_score = overall_risk * 100

        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)

        # Get predicted class
        class_idx = int(np.argmax(model_predictions))
        if class_names is None:
            class_names = self.class_names
        predicted_class = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

        # Individual risk factors for transparency
        factors = {
            'model_prediction': model_risk,
            'model_confidence': confidence,
            'predicted_class': predicted_class,
            'visual_asymmetry': visual_features.get('asymmetry', 0.0),
            'visual_border': visual_features.get('border_irregularity', 0.0),
            'visual_color': visual_features.get('color_variation', 0.0),
            'visual_overall': visual_risk,
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score,
            risk_level,
            confidence,
            predicted_class
        )

        return RiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            confidence=confidence,
            factors=factors,
            recommendations=recommendations
        )

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """
        Determine categorical risk level from score.

        Args:
            score: Overall risk score (0-100)

        Returns:
            RiskLevel enum value
        """
        if score < 25:
            return RiskLevel.LOW
        elif score < 50:
            return RiskLevel.MODERATE
        elif score < 75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _generate_recommendations(
        self,
        score: float,
        risk_level: RiskLevel,
        confidence: float,
        predicted_class: str
    ) -> List[str]:
        """
        Generate clinical recommendations based on risk assessment.

        Args:
            score: Overall risk score
            risk_level: Risk level category
            confidence: Model confidence
            predicted_class: Predicted lesion class

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Universal recommendation
        recommendations.append(
            "This system is NOT intended for medical diagnosis. "
            "Always consult a qualified dermatologist or healthcare professional."
        )

        # Risk-based recommendations
        if risk_level == RiskLevel.LOW:
            recommendations.append("Risk appears low based on visual analysis.")
            recommendations.append("Consider routine skin checks and monitoring.")
            recommendations.append("Watch for any changes in size, shape, or color.")

        elif risk_level == RiskLevel.MODERATE:
            recommendations.append("Moderate risk detected - professional evaluation recommended.")
            recommendations.append("Schedule a dermatologist appointment for examination.")
            recommendations.append("Document and monitor for changes.")

        elif risk_level == RiskLevel.HIGH:
            recommendations.append("HIGH RISK - Seek professional medical evaluation soon.")
            recommendations.append("Schedule dermatologist appointment within 1-2 weeks.")
            recommendations.append("Bring this report and photos to your appointment.")

        else:  # VERY_HIGH
            recommendations.append("VERY HIGH RISK - Urgent medical evaluation strongly recommended.")
            recommendations.append("Contact a dermatologist immediately for expedited appointment.")
            recommendations.append("Do not delay - early detection is crucial.")

        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.append(
                f"Note: Model confidence is relatively low ({confidence:.1%}). "
                "Professional assessment is especially important."
            )

        # Class-specific recommendations
        if "malignant" in predicted_class.lower() or "melanoma" in predicted_class.lower():
            recommendations.append(
                "Model suggests potential malignancy - urgent professional evaluation essential."
            )

        return recommendations


def calculate_abcde_score(visual_features: Dict[str, float]) -> Dict[str, any]:
    """
    Calculate ABCDE criteria scores (common melanoma screening method).

    A - Asymmetry
    B - Border irregularity
    C - Color variation
    D - Diameter (would need actual measurements)
    E - Evolution (would need historical data)

    Args:
        visual_features: Dictionary of visual feature measurements

    Returns:
        Dictionary with ABCDE scores and interpretation
    """
    abcde = {
        'A_asymmetry': visual_features.get('asymmetry', 0.0),
        'B_border': visual_features.get('border_irregularity', 0.0),
        'C_color': visual_features.get('color_variation', 0.0),
        'D_diameter': 'Not measured (requires scale reference)',
        'E_evolution': 'Requires historical comparison',
    }

    # Count concerning features (score > 0.5)
    concerning_features = sum(1 for k, v in abcde.items() if isinstance(v, float) and v > 0.5)

    interpretation = {
        'scores': abcde,
        'concerning_features': concerning_features,
        'note': 'ABCDE is a clinical guideline. Professional evaluation required for diagnosis.'
    }

    return interpretation


if __name__ == "__main__":
    # Example usage
    scorer = LesionRiskScorer(class_names=["benign", "suspicious", "malignant"])

    # Simulate model predictions
    predictions = np.array([0.2, 0.3, 0.5])  # Example probabilities

    # Simulate visual features
    visual_features = {
        'asymmetry': 0.7,
        'border_irregularity': 0.6,
        'color_variation': 0.8,
    }

    # Calculate risk
    risk_score = scorer.calculate_comprehensive_risk(predictions, visual_features)

    print("Risk Assessment Results:")
    print(f"Overall Score: {risk_score.overall_score:.2f}/100")
    print(f"Risk Level: {risk_score.risk_level.value}")
    print(f"Confidence: {risk_score.confidence:.2%}")
    print(f"\nRecommendations:")
    for rec in risk_score.recommendations:
        print(f"  - {rec}")
