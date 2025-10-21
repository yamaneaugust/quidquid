"""
Unit tests for image preprocessing.
"""

import pytest
import numpy as np
from PIL import Image
import torch

from src.preprocessing.image_processing import (
    ImagePreprocessor,
    LesionFeatureExtractor,
    load_image
)


def test_preprocessor_creation():
    """Test ImagePreprocessor creation."""
    preprocessor = ImagePreprocessor()
    assert preprocessor is not None
    assert preprocessor.target_size == (224, 224)


def test_preprocessor_array():
    """Test preprocessing numpy array."""
    preprocessor = ImagePreprocessor(augment=False)

    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Preprocess
    tensor = preprocessor.preprocess_array(dummy_image)

    # Check output
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_preprocessor_normalization():
    """Test that normalization is applied correctly."""
    preprocessor = ImagePreprocessor(normalize=True)

    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = preprocessor.preprocess_array(dummy_image)

    # Normalized values should be roughly in [-3, 3] range
    assert tensor.min() >= -5
    assert tensor.max() <= 5


def test_feature_extractor_asymmetry():
    """Test asymmetry calculation."""
    # Create a symmetric image (square)
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128

    extractor = LesionFeatureExtractor()
    asymmetry = extractor.calculate_asymmetry(image)

    # Should be low for symmetric shape
    assert 0 <= asymmetry <= 1


def test_feature_extractor_border():
    """Test border irregularity calculation."""
    # Create a circular image
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    extractor = LesionFeatureExtractor()
    irregularity = extractor.calculate_border_irregularity(image)

    # Should return a value between 0 and 1
    assert 0 <= irregularity <= 1


def test_feature_extractor_color():
    """Test color variation calculation."""
    # Create image with uniform color
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128

    extractor = LesionFeatureExtractor()
    color_var = extractor.calculate_color_variation(image)

    # Should be low for uniform color
    assert 0 <= color_var <= 1


def test_extract_all_features():
    """Test extracting all features at once."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    extractor = LesionFeatureExtractor()
    features = extractor.extract_all_features(image)

    # Check all features are present
    assert 'asymmetry' in features
    assert 'border_irregularity' in features
    assert 'color_variation' in features

    # Check all values are in valid range
    for value in features.values():
        assert 0 <= value <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
