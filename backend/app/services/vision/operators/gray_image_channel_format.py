"""
GrayImageChannelFormat operator for grayscale image processing.
"""
import cv2
import numpy as np
from typing import Dict, Any, Optional

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class GrayImageChannelFormat:
    """
    Convert image to grayscale and format channels appropriately.
    Supports various grayscale conversion methods and channel formatting options.
    """
    
    def __init__(self, 
                 inverse: bool = False,
                 method: str = 'cv2',
                 keep_dims: bool = True):
        """
        Initialize GrayImageChannelFormat operator.
        
        Args:
            inverse: If False, add channel at beginning (CHW), if True, add at end (HWC)
            method: Grayscale conversion method ('cv2', 'luminance', 'average')
            keep_dims: Whether to keep channel dimension for grayscale images
        """
        self.inverse = inverse
        self.method = method
        self.keep_dims = keep_dims
        
        # Validate method
        valid_methods = ['cv2', 'luminance', 'average']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert image to grayscale and format channels.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with grayscale image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Convert to grayscale and format
            gray_img, conversion_info = self._convert_to_grayscale(img)
            data['image'] = gray_img
            data['grayscale_info'] = conversion_info
            
            return data
            
        except Exception as e:
            logger.error(f"Grayscale conversion failed: {e}")
            data['grayscale_error'] = str(e)
            return data
    
    def _convert_to_grayscale(self, img: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Convert image to grayscale and format channels."""
        try:
            original_shape = img.shape
            
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                if img.shape[2] == 3:
                    # RGB/BGR to grayscale
                    gray_img = self._apply_grayscale_conversion(img)
                elif img.shape[2] == 1:
                    # Already single channel, just squeeze
                    gray_img = img.squeeze(axis=2)
                else:
                    # Multi-channel, use first channel or convert
                    if img.shape[2] == 4:
                        # RGBA, ignore alpha
                        gray_img = self._apply_grayscale_conversion(img[:, :, :3])
                    else:
                        # Use first channel
                        gray_img = img[:, :, 0]
                        logger.warning(f"Using first channel from {img.shape[2]}-channel image")
            else:
                # Already grayscale
                gray_img = img
            
            # Format channels
            if self.keep_dims:
                if len(gray_img.shape) == 2:
                    if not self.inverse:
                        # Add channel dimension at beginning: (H, W) -> (1, H, W)
                        gray_img = np.expand_dims(gray_img, axis=0)
                    else:
                        # Add channel dimension at end: (H, W) -> (H, W, 1)
                        gray_img = np.expand_dims(gray_img, axis=-1)
            
            conversion_info = {
                'original_shape': original_shape,
                'final_shape': gray_img.shape,
                'method': self.method,
                'inverse': self.inverse,
                'keep_dims': self.keep_dims
            }
            
            return gray_img, conversion_info
            
        except Exception as e:
            logger.error(f"Grayscale conversion computation failed: {e}")
            return img, {}
    
    def _apply_grayscale_conversion(self, img: np.ndarray) -> np.ndarray:
        """Apply grayscale conversion using specified method."""
        try:
            if self.method == 'cv2':
                # Use OpenCV's BGR to GRAY conversion
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif self.method == 'luminance':
                # Use luminance formula: 0.299*R + 0.587*G + 0.114*B
                # Assuming BGR format (OpenCV default)
                weights = np.array([0.114, 0.587, 0.299])  # BGR weights
                return np.dot(img, weights).astype(img.dtype)
            elif self.method == 'average':
                # Simple average of channels
                return np.mean(img, axis=2).astype(img.dtype)
            else:
                # Fallback to OpenCV
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        except Exception as e:
            logger.error(f"Grayscale conversion method failed: {e}")
            # Fallback to simple average
            return np.mean(img, axis=2).astype(img.dtype)


class ColorSpaceConverter:
    """
    Convert between different color spaces.
    """
    
    def __init__(self, 
                 src_format: str = 'BGR',
                 dst_format: str = 'RGB'):
        """
        Initialize ColorSpaceConverter.
        
        Args:
            src_format: Source color format
            dst_format: Destination color format
        """
        self.src_format = src_format.upper()
        self.dst_format = dst_format.upper()
        
        # Define conversion mapping
        self.conversion_map = {
            ('BGR', 'RGB'): cv2.COLOR_BGR2RGB,
            ('RGB', 'BGR'): cv2.COLOR_RGB2BGR,
            ('BGR', 'GRAY'): cv2.COLOR_BGR2GRAY,
            ('RGB', 'GRAY'): cv2.COLOR_RGB2GRAY,
            ('GRAY', 'BGR'): cv2.COLOR_GRAY2BGR,
            ('GRAY', 'RGB'): cv2.COLOR_GRAY2RGB,
            ('BGR', 'HSV'): cv2.COLOR_BGR2HSV,
            ('RGB', 'HSV'): cv2.COLOR_RGB2HSV,
            ('HSV', 'BGR'): cv2.COLOR_HSV2BGR,
            ('HSV', 'RGB'): cv2.COLOR_HSV2RGB,
            ('BGR', 'LAB'): cv2.COLOR_BGR2LAB,
            ('RGB', 'LAB'): cv2.COLOR_RGB2LAB,
            ('LAB', 'BGR'): cv2.COLOR_LAB2BGR,
            ('LAB', 'RGB'): cv2.COLOR_LAB2RGB,
        }
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert color space of image.
        
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
            
            # Convert color space
            converted_img, conversion_info = self._convert_color_space(img)
            data['image'] = converted_img
            data['color_conversion_info'] = conversion_info
            
            return data
            
        except Exception as e:
            logger.error(f"Color space conversion failed: {e}")
            data['color_conversion_error'] = str(e)
            return data
    
    def _convert_color_space(self, img: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Convert image color space."""
        try:
            # Check if conversion is needed
            if self.src_format == self.dst_format:
                return img, {
                    'src_format': self.src_format,
                    'dst_format': self.dst_format,
                    'conversion_applied': False
                }
            
            # Get conversion code
            conversion_key = (self.src_format, self.dst_format)
            if conversion_key not in self.conversion_map:
                raise ValueError(f"Conversion from {self.src_format} to {self.dst_format} not supported")
            
            conversion_code = self.conversion_map[conversion_key]
            
            # Apply conversion
            converted_img = cv2.cvtColor(img, conversion_code)
            
            conversion_info = {
                'src_format': self.src_format,
                'dst_format': self.dst_format,
                'conversion_code': conversion_code,
                'conversion_applied': True,
                'original_shape': img.shape,
                'final_shape': converted_img.shape
            }
            
            return converted_img, conversion_info
            
        except Exception as e:
            logger.error(f"Color space conversion computation failed: {e}")
            return img, {}


class ChannelSelector:
    """
    Select specific channels from multi-channel image.
    """
    
    def __init__(self, channels: list):
        """
        Initialize ChannelSelector.
        
        Args:
            channels: List of channel indices to select
        """
        self.channels = channels
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select channels from image.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with channel-selected image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Select channels
            selected_img, selection_info = self._select_channels(img)
            data['image'] = selected_img
            data['channel_selection_info'] = selection_info
            
            return data
            
        except Exception as e:
            logger.error(f"Channel selection failed: {e}")
            data['channel_selection_error'] = str(e)
            return data
    
    def _select_channels(self, img: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        """Select specified channels from image."""
        try:
            if len(img.shape) < 3:
                raise ValueError("Image must have at least 3 dimensions for channel selection")
            
            num_channels = img.shape[2]
            
            # Validate channel indices
            for ch in self.channels:
                if ch < 0 or ch >= num_channels:
                    raise ValueError(f"Channel index {ch} out of range [0, {num_channels-1}]")
            
            # Select channels
            if len(self.channels) == 1:
                # Single channel selection
                selected_img = img[:, :, self.channels[0]]
            else:
                # Multiple channel selection
                selected_img = img[:, :, self.channels]
            
            selection_info = {
                'original_channels': num_channels,
                'selected_channels': self.channels,
                'final_channels': len(self.channels),
                'original_shape': img.shape,
                'final_shape': selected_img.shape
            }
            
            return selected_img, selection_info
            
        except Exception as e:
            logger.error(f"Channel selection computation failed: {e}")
            return img, {}