"""
Skin and Lesion Image Validator

Detects whether an uploaded image is likely to be a skin lesion or not.
Rejects non-skin images (text, objects, screenshots, etc.) before analysis.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple


class SkinLesionValidator:
    """Validates if an image contains a skin lesion"""

    def __init__(self):
        # Skin color ranges in HSV
        # Lower bound: Lighter skin tones
        self.skin_hsv_lower1 = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_hsv_upper1 = np.array([20, 170, 255], dtype=np.uint8)

        # Upper bound: Darker skin tones
        self.skin_hsv_lower2 = np.array([0, 10, 60], dtype=np.uint8)
        self.skin_hsv_upper2 = np.array([30, 200, 255], dtype=np.uint8)

    def validate_image(self, image_path: str) -> Tuple[bool, str, dict]:
        """
        Validate if image is likely a skin lesion

        Returns:
            (is_valid, reason, metrics):
                - is_valid: True if likely a skin lesion
                - reason: Explanation if rejected
                - metrics: Dictionary of validation metrics
        """
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False, "Unable to load image", {}

            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Initialize metrics
            metrics = {}

            # 1. Skin Color Detection
            skin_percentage = self._check_skin_color(hsv)
            metrics['skin_percentage'] = skin_percentage

            # 2. Edge Density (text/documents have high edge density)
            edge_density = self._check_edge_density(gray)
            metrics['edge_density'] = edge_density

            # 3. Color Diversity (screenshots/text have limited colors)
            color_diversity = self._check_color_diversity(image)
            metrics['color_diversity'] = color_diversity

            # 4. Brightness uniformity (overly bright/dark images)
            brightness_score = self._check_brightness(gray)
            metrics['brightness_score'] = brightness_score

            # 5. Aspect ratio check (extreme aspect ratios unlikely for lesion photos)
            aspect_ratio = image.shape[1] / image.shape[0]
            metrics['aspect_ratio'] = aspect_ratio

            # Validation logic
            rejection_reasons = []

            # Check skin color percentage
            if skin_percentage < 15:
                rejection_reasons.append(f"Low skin-like colors ({skin_percentage:.1f}% - expected >15%)")

            # Check edge density (text/documents have very high edge density)
            if edge_density > 0.25:
                rejection_reasons.append(f"High edge density ({edge_density:.2f} - suggests text/document)")

            # Check color diversity (screenshots/text have limited palette)
            if color_diversity < 30:
                rejection_reasons.append(f"Low color diversity ({color_diversity:.0f} - suggests artificial image)")

            # Check brightness
            if brightness_score < 30 or brightness_score > 240:
                rejection_reasons.append(f"Extreme brightness ({brightness_score:.0f} - expected 30-240)")

            # Check aspect ratio (extreme ratios unlikely)
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                rejection_reasons.append(f"Unusual aspect ratio ({aspect_ratio:.2f} - expected 0.4-2.5)")

            # Decision
            if rejection_reasons:
                is_valid = False
                reason = "This does not appear to be a skin lesion image:\n• " + "\n• ".join(rejection_reasons)
            else:
                is_valid = True
                reason = "Image appears to be a valid skin lesion photograph"

            return is_valid, reason, metrics

        except Exception as e:
            return False, f"Error validating image: {str(e)}", {}

    def _check_skin_color(self, hsv_image: np.ndarray) -> float:
        """Calculate percentage of skin-like colors in image"""
        # Create masks for both skin tone ranges
        mask1 = cv2.inRange(hsv_image, self.skin_hsv_lower1, self.skin_hsv_upper1)
        mask2 = cv2.inRange(hsv_image, self.skin_hsv_lower2, self.skin_hsv_upper2)

        # Combine masks
        skin_mask = cv2.bitwise_or(mask1, mask2)

        # Calculate percentage
        skin_pixels = np.sum(skin_mask > 0)
        total_pixels = skin_mask.size
        skin_percentage = (skin_pixels / total_pixels) * 100

        return skin_percentage

    def _check_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density (text/documents have high values)"""
        # Detect edges
        edges = cv2.Canny(gray_image, 50, 150)

        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        edge_density = edge_pixels / total_pixels

        return edge_density

    def _check_color_diversity(self, image: np.ndarray) -> float:
        """Calculate color diversity (screenshots have low diversity)"""
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3)

        # Calculate unique colors (sample to avoid memory issues)
        if len(pixels) > 10000:
            pixels = pixels[np.random.choice(len(pixels), 10000, replace=False)]

        # Count unique colors
        unique_colors = len(np.unique(pixels, axis=0))

        return unique_colors

    def _check_brightness(self, gray_image: np.ndarray) -> float:
        """Check average brightness"""
        return float(np.mean(gray_image))


# Convenience function
def is_skin_lesion_image(image_path: str) -> Tuple[bool, str]:
    """
    Quick validation function

    Returns:
        (is_valid, reason): True if image is likely a skin lesion, with explanation
    """
    validator = SkinLesionValidator()
    is_valid, reason, _ = validator.validate_image(image_path)
    return is_valid, reason
