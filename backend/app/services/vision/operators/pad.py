"""
Pad operator for image padding operations.
"""
import cv2
import numpy as np
from typing import Dict, Any, Union, List, Optional, Tuple

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class Pad:
    """
    Pad image to specified size or make it divisible by a certain value.
    Supports various padding modes and strategies.
    """
    
    def __init__(self, 
                 size: Optional[Union[int, List[int]]] = None,
                 size_div: int = 32,
                 pad_mode: str = 'constant',
                 pad_value: Union[int, float, List[Union[int, float]]] = 0,
                 center: bool = False):
        """
        Initialize Pad operator.
        
        Args:
            size: Target size [height, width] or single value for square
            size_div: Make image size divisible by this value
            pad_mode: Padding mode ('constant', 'edge', 'reflect', 'symmetric')
            pad_value: Value to use for constant padding
            center: Whether to center the image in the padded area
        """
        self.size = size
        self.size_div = size_div
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.center = center
        
        # Validate pad_mode
        valid_modes = ['constant', 'edge', 'reflect', 'symmetric']
        if pad_mode not in valid_modes:
            raise ValueError(f"Invalid pad_mode '{pad_mode}'. Must be one of {valid_modes}")
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pad image in input data.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with padded image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Pad image
            padded_img, pad_info = self._pad_image(img)
            data['image'] = padded_img
            data['pad_info'] = pad_info
            
            return data
            
        except Exception as e:
            logger.error(f"Image padding failed: {e}")
            data['pad_error'] = str(e)
            return data
    
    def _pad_image(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pad image according to configuration."""
        try:
            h, w = img.shape[:2]
            
            # Calculate target dimensions
            if self.size is not None:
                if isinstance(self.size, int):
                    target_h = target_w = self.size
                else:
                    target_h, target_w = self.size
            else:
                # Make divisible by size_div
                target_h = max(int(np.ceil(h / self.size_div) * self.size_div), self.size_div)
                target_w = max(int(np.ceil(w / self.size_div) * self.size_div), self.size_div)
            
            # Calculate padding amounts
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            if pad_h == 0 and pad_w == 0:
                # No padding needed
                return img, {'pad_top': 0, 'pad_bottom': 0, 'pad_left': 0, 'pad_right': 0}
            
            # Calculate padding distribution
            if self.center:
                # Center the image
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
            else:
                # Pad to bottom and right
                pad_top = 0
                pad_bottom = pad_h
                pad_left = 0
                pad_right = pad_w
            
            # Apply padding
            padded_img = self._apply_padding(img, pad_top, pad_bottom, pad_left, pad_right)
            
            pad_info = {
                'pad_top': pad_top,
                'pad_bottom': pad_bottom,
                'pad_left': pad_left,
                'pad_right': pad_right,
                'original_shape': (h, w),
                'padded_shape': padded_img.shape[:2]
            }
            
            return padded_img, pad_info
            
        except Exception as e:
            logger.error(f"Padding computation failed: {e}")
            return img, {}
    
    def _apply_padding(self, img: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
        """Apply padding using OpenCV."""
        try:
            # Convert pad_mode to OpenCV border type
            border_type_map = {
                'constant': cv2.BORDER_CONSTANT,
                'edge': cv2.BORDER_REPLICATE,
                'reflect': cv2.BORDER_REFLECT,
                'symmetric': cv2.BORDER_REFLECT_101
            }
            
            border_type = border_type_map[self.pad_mode]
            
            # Handle pad_value for different image types
            if self.pad_mode == 'constant':
                if isinstance(self.pad_value, (int, float)):
                    value = self.pad_value
                elif isinstance(self.pad_value, list):
                    # Multi-channel value
                    if len(img.shape) == 3 and len(self.pad_value) == img.shape[2]:
                        value = self.pad_value
                    else:
                        value = self.pad_value[0] if self.pad_value else 0
                else:
                    value = 0
            else:
                value = None
            
            # Apply padding
            if value is not None:
                padded_img = cv2.copyMakeBorder(
                    img, top, bottom, left, right, border_type, value=value
                )
            else:
                padded_img = cv2.copyMakeBorder(
                    img, top, bottom, left, right, border_type
                )
            
            return padded_img
            
        except Exception as e:
            logger.error(f"Padding application failed: {e}")
            return img


class CropPad:
    """
    Crop or pad image to exact target size.
    Combines cropping and padding operations for precise size control.
    """
    
    def __init__(self, 
                 target_size: Union[int, List[int]],
                 pad_mode: str = 'constant',
                 pad_value: Union[int, float, List[Union[int, float]]] = 0,
                 crop_mode: str = 'center'):
        """
        Initialize CropPad operator.
        
        Args:
            target_size: Target size [height, width] or single value for square
            pad_mode: Padding mode for when image is smaller than target
            pad_value: Value to use for constant padding
            crop_mode: Cropping mode for when image is larger than target ('center', 'top_left')
        """
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = tuple(target_size)
        
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.crop_mode = crop_mode
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crop or pad image to target size.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with resized image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Process image
            processed_img, operation_info = self._crop_pad_image(img)
            data['image'] = processed_img
            data['crop_pad_info'] = operation_info
            
            return data
            
        except Exception as e:
            logger.error(f"Crop-pad operation failed: {e}")
            data['crop_pad_error'] = str(e)
            return data
    
    def _crop_pad_image(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Crop or pad image to target size."""
        try:
            h, w = img.shape[:2]
            target_h, target_w = self.target_size
            
            operation_info = {
                'original_shape': (h, w),
                'target_shape': (target_h, target_w),
                'operations': []
            }
            
            result_img = img
            
            # Handle height
            if h > target_h:
                # Crop height
                if self.crop_mode == 'center':
                    start_h = (h - target_h) // 2
                    end_h = start_h + target_h
                else:  # top_left
                    start_h = 0
                    end_h = target_h
                
                result_img = result_img[start_h:end_h, :]
                operation_info['operations'].append(f'crop_height_{start_h}_{end_h}')
                
            elif h < target_h:
                # Pad height
                pad_h = target_h - h
                pad_top = pad_h // 2 if self.crop_mode == 'center' else 0
                pad_bottom = pad_h - pad_top
                
                result_img = self._apply_padding_1d(result_img, pad_top, pad_bottom, axis=0)
                operation_info['operations'].append(f'pad_height_{pad_top}_{pad_bottom}')
            
            # Handle width
            h_new, w_new = result_img.shape[:2]
            if w_new > target_w:
                # Crop width
                if self.crop_mode == 'center':
                    start_w = (w_new - target_w) // 2
                    end_w = start_w + target_w
                else:  # top_left
                    start_w = 0
                    end_w = target_w
                
                result_img = result_img[:, start_w:end_w]
                operation_info['operations'].append(f'crop_width_{start_w}_{end_w}')
                
            elif w_new < target_w:
                # Pad width
                pad_w = target_w - w_new
                pad_left = pad_w // 2 if self.crop_mode == 'center' else 0
                pad_right = pad_w - pad_left
                
                result_img = self._apply_padding_1d(result_img, pad_left, pad_right, axis=1)
                operation_info['operations'].append(f'pad_width_{pad_left}_{pad_right}')
            
            operation_info['final_shape'] = result_img.shape[:2]
            
            return result_img, operation_info
            
        except Exception as e:
            logger.error(f"Crop-pad computation failed: {e}")
            return img, {}
    
    def _apply_padding_1d(self, img: np.ndarray, pad_before: int, pad_after: int, axis: int) -> np.ndarray:
        """Apply padding along a specific axis."""
        try:
            if axis == 0:
                # Pad height
                return cv2.copyMakeBorder(
                    img, pad_before, pad_after, 0, 0, 
                    cv2.BORDER_CONSTANT, value=self.pad_value
                )
            else:
                # Pad width
                return cv2.copyMakeBorder(
                    img, 0, 0, pad_before, pad_after, 
                    cv2.BORDER_CONSTANT, value=self.pad_value
                )
                
        except Exception as e:
            logger.error(f"1D padding failed: {e}")
            return img