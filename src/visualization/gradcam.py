"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model interpretability.

Generates heatmaps showing which regions of the image the model focuses on.
"""

import torch
import torch.nn.functional as F
import numpy as np
# import cv2  # Commented out - causes deployment issues with system dependencies
from typing import Tuple, Optional
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM implementation for CNN visualization.

    Shows which parts of the image the model is looking at when making predictions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Initialize Grad-CAM.

        Args:
            model: The CNN model
            target_layer: The layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook to save forward pass activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients."""
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (uses predicted class if None)

        Returns:
            CAM heatmap as numpy array (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate CAM
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU (only positive influence)
        cam = F.relu(cam)

        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Generate colored heatmap.

        Args:
            input_tensor: Input image tensor
            target_class: Target class to visualize
            colormap: Matplotlib colormap to use (default: 'jet')

        Returns:
            Colored heatmap (H, W, 3) in range [0, 255]
        """
        # Get CAM
        cam = self.generate_cam(input_tensor, target_class)

        # Resize to input size using PIL
        input_size = input_tensor.shape[2:]
        cam_pil = Image.fromarray(np.uint8(255 * cam))
        cam_resized = cam_pil.resize((input_size[1], input_size[0]), Image.LANCZOS)
        cam_resized = np.array(cam_resized) / 255.0

        # Apply colormap using matplotlib
        cmap = cm.get_cmap(colormap)
        heatmap = cmap(cam_resized)[:, :, :3]  # Remove alpha channel
        heatmap = np.uint8(255 * heatmap)

        return heatmap

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.

        Args:
            image: Original image (H, W, 3) in range [0, 255]
            heatmap: Colored heatmap (H, W, 3) in range [0, 255]
            alpha: Overlay transparency (0=only image, 1=only heatmap)

        Returns:
            Overlaid image (H, W, 3)
        """
        # Ensure same size using PIL
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap_pil = Image.fromarray(heatmap)
            heatmap_pil = heatmap_pil.resize((image.shape[1], image.shape[0]), Image.LANCZOS)
            heatmap = np.array(heatmap_pil)

        # Ensure uint8
        image = np.uint8(image)
        heatmap = np.uint8(heatmap)

        # Overlay using numpy (equivalent to cv2.addWeighted)
        overlaid = np.uint8((1 - alpha) * image + alpha * heatmap)

        return overlaid


def generate_gradcam_visualization(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM visualization.

    Args:
        model: CNN model (LesionCNN or LesionResNet)
        image_tensor: Preprocessed image tensor (1, C, H, W)
        original_image: Original image array (H, W, 3) in range [0, 255]
        target_class: Target class to visualize (None = predicted class)
        alpha: Overlay transparency

    Returns:
        Tuple of (heatmap, overlaid_image)
    """
    # Get the last convolutional layer
    # Different architectures have different layer names
    target_layer = None

    # Check if it's a ResNet (has 'backbone' attribute)
    if hasattr(model, 'backbone'):
        # ResNet50 - use layer4 (last residual block)
        if hasattr(model.backbone, 'layer4'):
            target_layer = model.backbone.layer4[-1]
    else:
        # LesionCNN - look for conv8 or use last conv layer
        for name, module in model.named_modules():
            if 'conv8' in name or (isinstance(module, torch.nn.Conv2d) and 'conv' in name):
                target_layer = module

    # Fallback: find the last conv layer in the entire model
    if target_layer is None:
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        if conv_layers:
            target_layer = conv_layers[-1]
        else:
            raise ValueError("No convolutional layers found in model")

    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap = gradcam.generate_heatmap(image_tensor, target_class)

    # Resize original image to match input size if needed using PIL
    input_size = (image_tensor.shape[3], image_tensor.shape[2])  # (W, H)
    if original_image.shape[:2] != input_size[::-1]:
        original_pil = Image.fromarray(original_image)
        original_pil = original_pil.resize(input_size, Image.LANCZOS)
        original_resized = np.array(original_pil)
    else:
        original_resized = original_image

    # Overlay
    overlaid = gradcam.overlay_heatmap(original_resized, heatmap, alpha)

    # Heatmap and overlaid are already in RGB format (matplotlib colormap output)
    return heatmap, overlaid
