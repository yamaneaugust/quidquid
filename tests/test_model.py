"""
Unit tests for CNN model.
"""

import pytest
import torch
import numpy as np

from src.model.cnn import LesionCNN, LesionResNet, create_model


def test_lesion_cnn_creation():
    """Test LesionCNN model creation."""
    model = LesionCNN(num_classes=3)
    assert model is not None
    assert model.num_classes == 3


def test_lesion_cnn_forward():
    """Test LesionCNN forward pass."""
    model = LesionCNN(num_classes=3)
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape
    assert output.shape == (batch_size, 3)


def test_model_factory():
    """Test model factory function."""
    cnn_model = create_model(model_type='cnn', num_classes=3)
    assert isinstance(cnn_model, LesionCNN)

    resnet_model = create_model(model_type='resnet', num_classes=3)
    assert isinstance(resnet_model, LesionResNet)


def test_model_parameter_count():
    """Test that model has reasonable number of parameters."""
    model = create_model(model_type='cnn', num_classes=3)
    total_params = sum(p.numel() for p in model.parameters())

    # Model should have parameters (not empty)
    assert total_params > 0
    # Model should not be unreasonably large
    assert total_params < 100_000_000  # Less than 100M parameters


def test_different_input_sizes():
    """Test model with different input sizes."""
    model = LesionCNN(num_classes=3)
    model.eval()

    # Test various input sizes (thanks to adaptive pooling)
    input_sizes = [(224, 224), (256, 256), (512, 512)]

    for size in input_sizes:
        dummy_input = torch.randn(2, 3, *size)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (2, 3)


def test_model_dropout():
    """Test that dropout is applied in training mode."""
    model = LesionCNN(num_classes=3, dropout_rate=0.5)

    # Same input in train mode should give different outputs due to dropout
    dummy_input = torch.randn(1, 3, 224, 224)

    model.train()
    output1 = model(dummy_input)
    output2 = model(dummy_input)

    # Outputs should be different in training mode
    assert not torch.allclose(output1, output2)

    # But same in eval mode
    model.eval()
    with torch.no_grad():
        output1 = model(dummy_input)
        output2 = model(dummy_input)
    assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
