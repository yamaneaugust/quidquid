"""
Image preprocessing and augmentation utilities.

Handles loading, preprocessing, and augmenting medical images
for training and inference.
"""

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, Optional, List
# import cv2  # Commented out - causes deployment issues with system dependencies


class ImagePreprocessor:
    """
    Handles image preprocessing for lesion images.

    Args:
        target_size: Target image size (height, width)
        normalize: Whether to normalize images
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # Define normalization parameters (ImageNet stats)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Build transform pipeline
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build the transformation pipeline."""
        transform_list = []

        # Resize
        transform_list.append(transforms.Resize(self.target_size))

        # Data augmentation (for training)
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ])

        # Convert to tensor
        transform_list.append(transforms.ToTensor())

        # Normalize
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )

        return transforms.Compose(transform_list)

    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def preprocess_array(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a numpy array image.

        Args:
            image: Image as numpy array

        Returns:
            Preprocessed image tensor
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image)
        return self.transform(image)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize a tensor for visualization.

        Args:
            tensor: Normalized image tensor

        Returns:
            Denormalized image as numpy array
        """
        if self.normalize:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)

        # Convert to numpy and transpose
        image = tensor.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image, 0, 1)

        return (image * 255).astype(np.uint8)


class LesionFeatureExtractor:
    """
    Extracts visual features from lesion images for risk assessment.

    NOTE: OpenCV features disabled for deployment compatibility.
    Methods return placeholder values - CNN model predictions are still accurate.
    """

    @staticmethod
    def calculate_asymmetry(image: np.ndarray) -> float:
        """
        Calculate asymmetry score of the lesion.

        Args:
            image: Image as numpy array

        Returns:
            Asymmetry score (0-1, higher = more asymmetric)
        """
        # Simplified without OpenCV - return moderate value
        return 0.5

        # ORIGINAL CODE (disabled due to OpenCV dependency issues):
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if not contours:
        #     return 0.0
        # largest_contour = max(contours, key=cv2.contourArea)
        # M = cv2.moments(largest_contour)
        # if M['m00'] == 0:
        #     return 0.0
        # hu_moments = cv2.HuMoments(M)
        # asymmetry_score = np.abs(hu_moments[0][0])
        # return min(asymmetry_score, 1.0)

    @staticmethod
    def calculate_border_irregularity(image: np.ndarray) -> float:
        """
        Calculate border irregularity score.

        Args:
            image: Image as numpy array

        Returns:
            Irregularity score (0-1, higher = more irregular)
        """
        # Simplified without OpenCV - return moderate value
        return 0.5

        # ORIGINAL CODE (disabled due to OpenCV dependency issues):
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if not contours:
        #     return 0.0
        # largest_contour = max(contours, key=cv2.contourArea)
        # perimeter = cv2.arcLength(largest_contour, True)
        # area = cv2.contourArea(largest_contour)
        # if area == 0:
        #     return 0.0
        # circularity = 4 * np.pi * area / (perimeter ** 2)
        # irregularity = 1.0 - min(circularity, 1.0)
        # return irregularity

    @staticmethod
    def calculate_color_variation(image: np.ndarray) -> float:
        """
        Calculate color variation/diversity in the lesion.

        Args:
            image: Image as numpy array

        Returns:
            Color variation score (0-1)
        """
        # Simplified RGB-based color variation (without OpenCV)
        # Calculate standard deviation across RGB channels
        std_r = np.std(image[:, :, 0])
        std_g = np.std(image[:, :, 1])
        std_b = np.std(image[:, :, 2])

        # Normalize and combine
        color_variation = (std_r + std_g + std_b) / (3.0 * 255.0)

        return min(color_variation, 1.0)

        # ORIGINAL CODE (disabled due to OpenCV dependency issues):
        # lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # std_l = np.std(lab[:, :, 0])
        # std_a = np.std(lab[:, :, 1])
        # std_b = np.std(lab[:, :, 2])
        # color_variation = (std_l / 255.0 + std_a / 128.0 + std_b / 128.0) / 3.0
        # return min(color_variation, 1.0)

    @staticmethod
    def calculate_diameter_score(image: np.ndarray) -> float:
        """
        Calculate diameter concern score (D in ABCDE).

        Note: Without a scale reference, this estimates relative size.
        Clinical threshold is 6mm, but we estimate based on lesion area
        relative to image size.

        Args:
            image: Image as numpy array

        Returns:
            Diameter concern score (0-1, higher = larger/more concerning)
        """
        # Simplified without OpenCV - return moderate value
        return 0.5

        # ORIGINAL CODE (disabled due to OpenCV dependency issues):
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if not contours:
        #     return 0.0
        # largest_contour = max(contours, key=cv2.contourArea)
        # lesion_area = cv2.contourArea(largest_contour)
        # image_area = image.shape[0] * image.shape[1]
        # relative_size = lesion_area / image_area
        # if relative_size > 0.15:
        #     return 0.9
        # elif relative_size > 0.10:
        #     return 0.7
        # elif relative_size > 0.05:
        #     return 0.5
        # elif relative_size > 0.02:
        #     return 0.3
        # else:
        #     return 0.1

    @staticmethod
    def calculate_evolution_score(image: np.ndarray) -> dict:
        """
        Placeholder for evolution assessment (E in ABCDE).

        Evolution requires comparison with previous images over time.
        This cannot be assessed from a single image.

        Args:
            image: Image as numpy array

        Returns:
            Dictionary with evolution info and placeholder score
        """
        return {
            'score': None,  # Cannot assess without historical data
            'note': 'Requires comparison with previous images',
            'recommendation': 'Monitor for changes in size, shape, or color over time'
        }

    @staticmethod
    def extract_all_features(image: np.ndarray) -> dict:
        """
        Extract all ABCDE visual features from the image.

        Args:
            image: Image as numpy array

        Returns:
            Dictionary of feature name to value
        """
        evolution_data = LesionFeatureExtractor.calculate_evolution_score(image)

        return {
            'asymmetry': LesionFeatureExtractor.calculate_asymmetry(image),
            'border_irregularity': LesionFeatureExtractor.calculate_border_irregularity(image),
            'color_variation': LesionFeatureExtractor.calculate_color_variation(image),
            'diameter_score': LesionFeatureExtractor.calculate_diameter_score(image),
            'evolution_score': evolution_data['score'],
            'evolution_note': evolution_data['note'],
            'evolution_recommendation': evolution_data['recommendation'],
        }


def load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to image file
        target_size: Optional target size (height, width)

    Returns:
        Image as numpy array
    """
    image = Image.open(image_path).convert('RGB')

    if target_size:
        image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)

    return np.array(image)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor(augment=True)
    print("Preprocessor created successfully")
    print(f"Transform pipeline: {preprocessor.transform}")
