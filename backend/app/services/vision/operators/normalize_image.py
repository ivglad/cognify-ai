"""
NormalizeImage operator for image normalization.
"""
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Union

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class NormalizeImage:
    """
    Normalize image with configurable mean, std, and scaling.
    Supports both HWC and CHW formats.
    """
    
    def __init__(self, 
                 scale: float = 1.0/255.0,
                 mean: Optional[List[float]] = None,
                 std: Optional[List[float]] = None,
                 order: str = 'hwc'):
        """
        Initialize NormalizeImage operator.
        
        Args:
            scale: Scaling factor to apply before normalization
            mean: Mean values for normalization (default: ImageNet means)
            std: Standard deviation values for normalization (default: ImageNet stds)
            order: Image format - 'hwc' for Height-Width-Channel or 'chw' for Channel-Height-Width
        """
        self.scale = scale
        self.mean = np.array(mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std or [0.229, 0.224, 0.225], dtype=np.float32)
        self.order = order.lower()
        
        if self.order not in ['hwc', 'chw']:
            raise ValueError(f"Invalid order '{order}'. Must be 'hwc' or 'chw'")
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize image in input data.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with normalized image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Normalize image
            normalized_img = self._normalize_image(img)
            data['image'] = normalized_img
            
            return data
            
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            data['normalize_error'] = str(e)
            return data
    
    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image with mean and std."""
        try:
            # Convert to float32 for normalization
            img = img.astype(np.float32)
            
            # Apply scaling
            if self.scale != 1.0:
                img *= self.scale
            
            # Determine reshape dimensions based on order
            if self.order == 'chw':
                if len(img.shape) == 3:
                    # CHW format: (C, H, W)
                    shape = (img.shape[0], 1, 1)
                else:
                    # Assume single channel
                    shape = (1,)
            else:
                if len(img.shape) == 3:
                    # HWC format: (H, W, C)
                    shape = (1, 1, img.shape[2])
                else:
                    # Assume single channel
                    shape = (1,)
            
            # Reshape mean and std to match image dimensions
            if len(img.shape) == 3:
                # Multi-channel image
                num_channels = img.shape[0] if self.order == 'chw' else img.shape[2]
                
                # Adjust mean and std to match number of channels
                if len(self.mean) != num_channels:
                    if num_channels == 1:
                        # Grayscale: use first channel values
                        mean = np.array([self.mean[0]], dtype=np.float32)
                        std = np.array([self.std[0]], dtype=np.float32)
                    else:
                        # Repeat values to match channels
                        mean = np.tile(self.mean[:1], num_channels)
                        std = np.tile(self.std[:1], num_channels)
                else:
                    mean = self.mean[:num_channels]
                    std = self.std[:num_channels]
                
                mean = mean.reshape(shape)
                std = std.reshape(shape)
            else:
                # Single channel image
                mean = self.mean[0]
                std = self.std[0]
            
            # Apply normalization: (img - mean) / std
            normalized_img = (img - mean) / std
            
            return normalized_img
            
        except Exception as e:
            logger.error(f"Normalization computation failed: {e}")
            return img
    
    def denormalize(self, img: np.ndarray) -> np.ndarray:
        """
        Denormalize image (reverse normalization).
        
        Args:
            img: Normalized image
            
        Returns:
            Denormalized image
        """
        try:
            # Determine reshape dimensions based on order
            if self.order == 'chw':
                if len(img.shape) == 3:
                    shape = (img.shape[0], 1, 1)
                else:
                    shape = (1,)
            else:
                if len(img.shape) == 3:
                    shape = (1, 1, img.shape[2])
                else:
                    shape = (1,)
            
            # Reshape mean and std
            if len(img.shape) == 3:
                num_channels = img.shape[0] if self.order == 'chw' else img.shape[2]
                
                if len(self.mean) != num_channels:
                    if num_channels == 1:
                        mean = np.array([self.mean[0]], dtype=np.float32)
                        std = np.array([self.std[0]], dtype=np.float32)
                    else:
                        mean = np.tile(self.mean[:1], num_channels)
                        std = np.tile(self.std[:1], num_channels)
                else:
                    mean = self.mean[:num_channels]
                    std = self.std[:num_channels]
                
                mean = mean.reshape(shape)
                std = std.reshape(shape)
            else:
                mean = self.mean[0]
                std = self.std[0]
            
            # Reverse normalization: img * std + mean
            denormalized_img = img * std + mean
            
            # Reverse scaling
            if self.scale != 1.0:
                denormalized_img /= self.scale
            
            # Convert back to uint8 if needed
            if self.scale == 1.0/255.0:
                denormalized_img = np.clip(denormalized_img, 0, 255).astype(np.uint8)
            
            return denormalized_img
            
        except Exception as e:
            logger.error(f"Denormalization failed: {e}")
            return img


class StandardizeImage:
    """
    Standardize image to zero mean and unit variance.
    Alternative to NormalizeImage for cases where dataset statistics are unknown.
    """
    
    def __init__(self, per_channel: bool = True):
        """
        Initialize StandardizeImage operator.
        
        Args:
            per_channel: Whether to standardize each channel separately
        """
        self.per_channel = per_channel
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize image in input data.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with standardized image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Standardize image
            standardized_img = self._standardize_image(img)
            data['image'] = standardized_img
            
            # Store statistics for potential denormalization
            data['standardization_stats'] = {
                'mean': self.computed_mean,
                'std': self.computed_std
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Image standardization failed: {e}")
            data['standardize_error'] = str(e)
            return data
    
    def _standardize_image(self, img: np.ndarray) -> np.ndarray:
        """Standardize image to zero mean and unit variance."""
        try:
            # Convert to float32
            img = img.astype(np.float32)
            
            if self.per_channel and len(img.shape) == 3:
                # Standardize each channel separately
                standardized_img = np.zeros_like(img)
                means = []
                stds = []
                
                for c in range(img.shape[2]):
                    channel = img[:, :, c]
                    mean = np.mean(channel)
                    std = np.std(channel)
                    
                    # Avoid division by zero
                    if std == 0:
                        std = 1.0
                    
                    standardized_img[:, :, c] = (channel - mean) / std
                    means.append(mean)
                    stds.append(std)
                
                self.computed_mean = np.array(means)
                self.computed_std = np.array(stds)
            else:
                # Standardize entire image
                mean = np.mean(img)
                std = np.std(img)
                
                # Avoid division by zero
                if std == 0:
                    std = 1.0
                
                standardized_img = (img - mean) / std
                self.computed_mean = mean
                self.computed_std = std
            
            return standardized_img
            
        except Exception as e:
            logger.error(f"Standardization computation failed: {e}")
            return img