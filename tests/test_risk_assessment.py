"""
Unit tests for risk assessment.
"""

import pytest
import numpy as np

from src.risk_assessment.scorer import (
    LesionRiskScorer,
    RiskLevel,
    RiskScore,
    calculate_abcde_score
)


def test_risk_scorer_creation():
    """Test LesionRiskScorer creation."""
    scorer = LesionRiskScorer()
    assert scorer is not None


def test_model_risk_calculation():
    """Test model-based risk calculation."""
    scorer = LesionRiskScorer(class_names=['benign', 'suspicious', 'malignant'])

    predictions = np.array([0.7, 0.2, 0.1])  # Mostly benign
    risk, confidence = scorer.calculate_model_risk(predictions)

    # Risk should be relatively low
    assert 0 <= risk <= 1
    # Confidence should be high (max prob = 0.7)
    assert confidence == 0.7


def test_visual_risk_calculation():
    """Test visual feature-based risk calculation."""
    scorer = LesionRiskScorer()

    # Low risk features
    features = {
        'asymmetry': 0.1,
        'border_irregularity': 0.1,
        'color_variation': 0.1,
    }
    risk = scorer.calculate_visual_risk(features)
    assert risk < 0.3  # Should be low risk

    # High risk features
    features = {
        'asymmetry': 0.9,
        'border_irregularity': 0.9,
        'color_variation': 0.9,
    }
    risk = scorer.calculate_visual_risk(features)
    assert risk > 0.7  # Should be high risk


def test_comprehensive_risk():
    """Test comprehensive risk calculation."""
    scorer = LesionRiskScorer(class_names=['benign', 'suspicious', 'malignant'])

    # High risk scenario
    predictions = np.array([0.1, 0.2, 0.7])  # Likely malignant
    features = {
        'asymmetry': 0.8,
        'border_irregularity': 0.7,
        'color_variation': 0.9,
    }

    risk_score = scorer.calculate_comprehensive_risk(predictions, features)

    assert isinstance(risk_score, RiskScore)
    assert risk_score.overall_score > 50  # Should be high risk
    assert risk_score.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
    assert 0 <= risk_score.confidence <= 1


def test_risk_level_determination():
    """Test risk level categorization."""
    scorer = LesionRiskScorer()

    # Test each risk level
    assert scorer._determine_risk_level(10) == RiskLevel.LOW
    assert scorer._determine_risk_level(30) == RiskLevel.MODERATE
    assert scorer._determine_risk_level(60) == RiskLevel.HIGH
    assert scorer._determine_risk_level(90) == RiskLevel.VERY_HIGH


def test_recommendations_generation():
    """Test that recommendations are generated."""
    scorer = LesionRiskScorer(class_names=['benign', 'suspicious', 'malignant'])

    predictions = np.array([0.3, 0.3, 0.4])
    features = {
        'asymmetry': 0.5,
        'border_irregularity': 0.5,
        'color_variation': 0.5,
    }

    risk_score = scorer.calculate_comprehensive_risk(predictions, features)

    # Should have recommendations
    assert len(risk_score.recommendations) > 0

    # First recommendation should be the disclaimer
    assert "NOT intended for medical diagnosis" in risk_score.recommendations[0]


def test_risk_score_to_dict():
    """Test RiskScore conversion to dictionary."""
    risk_score = RiskScore(
        overall_score=65.5,
        risk_level=RiskLevel.HIGH,
        confidence=0.85,
        factors={'test': 0.5},
        recommendations=["Test recommendation"]
    )

    result = risk_score.to_dict()

    assert isinstance(result, dict)
    assert result['overall_score'] == 65.5
    assert result['risk_level'] == 'High'
    assert result['confidence'] == 0.85


def test_abcde_score():
    """Test ABCDE criteria scoring."""
    features = {
        'asymmetry': 0.7,
        'border_irregularity': 0.6,
        'color_variation': 0.8,
    }

    abcde = calculate_abcde_score(features)

    assert 'scores' in abcde
    assert 'concerning_features' in abcde
    assert 'note' in abcde


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
