"""
ToCHWImage operator for converting image format from HWC to CHW.
"""
import numpy as np
from typing import Dict, Any

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ToCHWImage:
    """
    Convert image from HWC (Height-Width-Channel) to CHW (Channel-Height-Width) format.
    This is commonly needed for deep learning models that expect channel-first format.
    """
    
    def __init__(self):
        """Initialize ToCHWImage operator."""
        pass
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert image format from HWC to CHW.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with converted image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Convert image format
            converted_img = self._convert_to_chw(img)
            data['image'] = converted_img
            
            return data
            
        except Exception as e:
            logger.error(f"Image format conversion failed: {e}")
            data['to_chw_error'] = str(e)
            return data
    
    def _convert_to_chw(self, img: np.ndarray) -> np.ndarray:
        """Convert image from HWC to CHW format."""
        try:
            if len(img.shape) == 2:
                # Grayscale image: (H, W) -> (1, H, W)
                return np.expand_dims(img, axis=0)
            elif len(img.shape) == 3:
                # Color image: (H, W, C) -> (C, H, W)
                return img.transpose((2, 0, 1))
            elif len(img.shape) == 4:
                # Batch of images: (B, H, W, C) -> (B, C, H, W)
                return img.transpose((0, 3, 1, 2))
            else:
                logger.warning(f"Unexpected image shape: {img.shape}")
                return img
                
        except Exception as e:
            logger.error(f"Format conversion computation failed: {e}")
            return img


class ToHWCImage:
    """
    Convert image from CHW (Channel-Height-Width) to HWC (Height-Width-Channel) format.
    This is the reverse operation of ToCHWImage.
    """
    
    def __init__(self):
        """Initialize ToHWCImage operator."""
        pass
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert image format from CHW to HWC.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with converted image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Convert image format
            converted_img = self._convert_to_hwc(img)
            data['image'] = converted_img
            
            return data
            
        except Exception as e:
            logger.error(f"Image format conversion failed: {e}")
            data['to_hwc_error'] = str(e)
            return data
    
    def _convert_to_hwc(self, img: np.ndarray) -> np.ndarray:
        """Convert image from CHW to HWC format."""
        try:
            if len(img.shape) == 2:
                # Already in HW format, return as is
                return img
            elif len(img.shape) == 3:
                # Color image: (C, H, W) -> (H, W, C)
                return img.transpose((1, 2, 0))
            elif len(img.shape) == 4:
                # Batch of images: (B, C, H, W) -> (B, H, W, C)
                return img.transpose((0, 2, 3, 1))
            else:
                logger.warning(f"Unexpected image shape: {img.shape}")
                return img
                
        except Exception as e:
            logger.error(f"Format conversion computation failed: {e}")
            return img


class KeepKeys:
    """
    Keep only specified keys in the data dictionary.
    Useful for cleaning up data pipeline and reducing memory usage.
    """
    
    def __init__(self, keys: list):
        """
        Initialize KeepKeys operator.
        
        Args:
            keys: List of keys to keep in the data dictionary
        """
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Keep only specified keys in data dictionary.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary with only specified keys
        """
        try:
            filtered_data = {}
            
            for key in self.keys:
                if key in data:
                    filtered_data[key] = data[key]
                else:
                    logger.warning(f"Key '{key}' not found in data")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Key filtering failed: {e}")
            return data


class Permute:
    """
    Permute dimensions of image tensor.
    Provides flexible dimension reordering for various format conversions.
    """
    
    def __init__(self, order: tuple):
        """
        Initialize Permute operator.
        
        Args:
            order: Tuple specifying the new order of dimensions
                  e.g., (2, 0, 1) to convert HWC to CHW
        """
        self.order = order
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Permute image dimensions.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with permuted image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Validate order
            if len(self.order) != len(img.shape):
                raise ValueError(f"Order length {len(self.order)} doesn't match image dimensions {len(img.shape)}")
            
            # Permute dimensions
            permuted_img = img.transpose(self.order)
            data['image'] = permuted_img
            
            return data
            
        except Exception as e:
            logger.error(f"Image permutation failed: {e}")
            data['permute_error'] = str(e)
            return data