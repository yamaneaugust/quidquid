"""
CNN Architecture for Lesion Classification

This module defines the convolutional neural network architecture
for lesion image analysis and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LesionCNN(nn.Module):
    """
    Convolutional Neural Network for lesion classification.

    Architecture:
    - 4 convolutional blocks with batch normalization
    - Max pooling for downsampling
    - Dropout for regularization
    - Fully connected layers for classification

    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability (default: 0.5)
        input_channels: Number of input channels (default: 3 for RGB)
    """

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.5,
        input_channels: int = 3
    ):
        super(LesionCNN, self).__init__()

        self.num_classes = num_classes

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)

        # Block 4
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def get_feature_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """
        Extract feature maps from a specific layer for visualization.

        Args:
            x: Input tensor
            layer_name: Name of the layer to extract features from

        Returns:
            Feature maps from the specified layer
        """
        # This can be extended for visualization purposes
        layers = {
            'conv1': lambda x: self.conv1(x),
            'conv2': lambda x: self.conv2(F.relu(self.bn1(self.conv1(x)))),
            # Add more layers as needed
        }

        if layer_name in layers:
            return layers[layer_name](x)
        else:
            raise ValueError(f"Layer {layer_name} not found")


class LesionResNet(nn.Module):
    """
    ResNet-inspired architecture for lesion classification.
    Uses residual connections for better gradient flow.
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super(LesionResNet, self).__init__()

        # Using pretrained ResNet as backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)

        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(
    model_type: str = 'cnn',
    num_classes: int = 3,
    dropout_rate: float = 0.5,
    pretrained: bool = False
) -> nn.Module:
    """
    Factory function to create a model.

    Args:
        model_type: Type of model ('cnn' or 'resnet')
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Whether to use pretrained weights (for resnet only)

    Returns:
        Instantiated model
    """
    if model_type == 'cnn':
        return LesionCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_type == 'resnet':
        return LesionResNet(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the model
    model = create_model(model_type='cnn', num_classes=3)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
