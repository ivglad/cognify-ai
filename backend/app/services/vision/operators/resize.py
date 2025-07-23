"""
Resize operators for image resizing operations.
"""
import cv2
import numpy as np
from typing import Dict, Any, Union, List, Optional, Tuple

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class LinearResize:
    """
    Resize image with linear interpolation and optional aspect ratio preservation.
    """
    
    def __init__(self, 
                 target_size: Union[int, List[int]],
                 keep_ratio: bool = True,
                 interp: int = cv2.INTER_LINEAR,
                 max_size: Optional[int] = None):
        """
        Initialize LinearResize operator.
        
        Args:
            target_size: Target size [height, width] or single value for square
            keep_ratio: Whether to keep aspect ratio
            interp: Interpolation method (cv2.INTER_*)
            max_size: Maximum size for any dimension when keep_ratio=True
        """
        if isinstance(target_size, int):
            self.target_size = [target_size, target_size]
        else:
            self.target_size = list(target_size)
        
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.max_size = max_size
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resize image in input data.
        
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
            
            # Resize image
            resized_img, resize_info = self._resize_image(img)
            data['image'] = resized_img
            data['resize_info'] = resize_info
            
            return data
            
        except Exception as e:
            logger.error(f"Image resizing failed: {e}")
            data['resize_error'] = str(e)
            return data
    
    def _resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize image according to configuration."""
        try:
            origin_shape = img.shape[:2]  # (height, width)
            h, w = origin_shape
            
            if self.keep_ratio:
                # Calculate scale to fit within target size
                scale_h = self.target_size[0] / h
                scale_w = self.target_size[1] / w
                scale = min(scale_h, scale_w)
                
                # Apply max_size constraint if specified
                if self.max_size is not None:
                    max_scale_h = self.max_size / h
                    max_scale_w = self.max_size / w
                    max_scale = min(max_scale_h, max_scale_w)
                    scale = min(scale, max_scale)
                
                # Calculate new dimensions
                new_h = int(h * scale)
                new_w = int(w * scale)
            else:
                # Use exact target size
                new_h, new_w = self.target_size
                scale = None
            
            # Resize image
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=self.interp)
            
            resize_info = {
                'original_shape': origin_shape,
                'target_shape': tuple(self.target_size),
                'final_shape': (new_h, new_w),
                'scale': scale,
                'keep_ratio': self.keep_ratio,
                'interpolation': self.interp
            }
            
            return resized_img, resize_info
            
        except Exception as e:
            logger.error(f"Resize computation failed: {e}")
            return img, {}


class Resize:
    """
    General resize operator with multiple scaling strategies.
    """
    
    def __init__(self, 
                 target_size: Optional[Union[int, List[int]]] = None,
                 scale_factor: Optional[Union[float, List[float]]] = None,
                 interp: int = cv2.INTER_LINEAR,
                 keep_ratio: bool = False):
        """
        Initialize Resize operator.
        
        Args:
            target_size: Target size [height, width] or single value
            scale_factor: Scale factor(s) for resizing
            interp: Interpolation method
            keep_ratio: Whether to keep aspect ratio
        """
        if target_size is not None and scale_factor is not None:
            raise ValueError("Cannot specify both target_size and scale_factor")
        
        if target_size is None and scale_factor is None:
            raise ValueError("Must specify either target_size or scale_factor")
        
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.interp = interp
        self.keep_ratio = keep_ratio
        
        # Normalize inputs
        if isinstance(target_size, int):
            self.target_size = [target_size, target_size]
        elif isinstance(target_size, list):
            self.target_size = list(target_size)
        
        if isinstance(scale_factor, (int, float)):
            self.scale_factor = [float(scale_factor), float(scale_factor)]
        elif isinstance(scale_factor, list):
            self.scale_factor = [float(x) for x in scale_factor]
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resize image in input data.
        
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
            
            # Resize image
            resized_img, resize_info = self._resize_image(img)
            data['image'] = resized_img
            data['resize_info'] = resize_info
            
            return data
            
        except Exception as e:
            logger.error(f"Image resizing failed: {e}")
            data['resize_error'] = str(e)
            return data
    
    def _resize_image(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resize image according to configuration."""
        try:
            h, w = img.shape[:2]
            
            if self.target_size is not None:
                # Resize to target size
                if self.keep_ratio:
                    # Calculate scale to fit within target size
                    scale_h = self.target_size[0] / h
                    scale_w = self.target_size[1] / w
                    scale = min(scale_h, scale_w)
                    
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                else:
                    new_h, new_w = self.target_size
                
                scale_info = {
                    'scale_h': new_h / h,
                    'scale_w': new_w / w
                }
            else:
                # Resize by scale factor
                if len(self.scale_factor) == 1:
                    scale_h = scale_w = self.scale_factor[0]
                else:
                    scale_h, scale_w = self.scale_factor
                
                new_h = int(h * scale_h)
                new_w = int(w * scale_w)
                
                scale_info = {
                    'scale_h': scale_h,
                    'scale_w': scale_w
                }
            
            # Resize image
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=self.interp)
            
            resize_info = {
                'original_shape': (h, w),
                'final_shape': (new_h, new_w),
                'scale_info': scale_info,
                'interpolation': self.interp,
                'keep_ratio': self.keep_ratio
            }
            
            return resized_img, resize_info
            
        except Exception as e:
            logger.error(f"Resize computation failed: {e}")
            return img, {}


class DetResizeForTest:
    """
    Specialized resize operator for detection model testing.
    Maintains aspect ratio and pads to target size.
    """
    
    def __init__(self, 
                 target_size: Union[int, List[int]] = 640,
                 interp: int = cv2.INTER_LINEAR,
                 pad_value: Union[int, List[int]] = 114):
        """
        Initialize DetResizeForTest operator.
        
        Args:
            target_size: Target size for detection model
            interp: Interpolation method
            pad_value: Value for padding
        """
        if isinstance(target_size, int):
            self.target_size = [target_size, target_size]
        else:
            self.target_size = list(target_size)
        
        self.interp = interp
        self.pad_value = pad_value
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resize and pad image for detection testing.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with processed image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(img)}")
            
            # Process image
            processed_img, process_info = self._process_for_detection(img)
            data['image'] = processed_img
            data['det_resize_info'] = process_info
            
            return data
            
        except Exception as e:
            logger.error(f"Detection resize failed: {e}")
            data['det_resize_error'] = str(e)
            return data
    
    def _process_for_detection(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process image for detection model."""
        try:
            h, w = img.shape[:2]
            target_h, target_w = self.target_size
            
            # Calculate scale to fit within target size while maintaining aspect ratio
            scale = min(target_h / h, target_w / w)
            
            # Calculate new dimensions
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Resize image
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=self.interp)
            
            # Create padded image
            padded_img = np.full((target_h, target_w, img.shape[2] if len(img.shape) == 3 else 1), 
                               self.pad_value, dtype=img.dtype)
            
            # Place resized image in padded image (top-left alignment)
            if len(img.shape) == 3:
                padded_img[:new_h, :new_w, :] = resized_img
            else:
                padded_img[:new_h, :new_w] = resized_img
            
            process_info = {
                'original_shape': (h, w),
                'resized_shape': (new_h, new_w),
                'final_shape': (target_h, target_w),
                'scale': scale,
                'pad_value': self.pad_value
            }
            
            return padded_img, process_info
            
        except Exception as e:
            logger.error(f"Detection processing failed: {e}")
            return img, {}


class SmartResize:
    """
    Smart resize operator that chooses the best strategy based on image properties.
    """
    
    def __init__(self, 
                 target_size: Union[int, List[int]],
                 min_scale: float = 0.1,
                 max_scale: float = 10.0,
                 quality_threshold: float = 0.5):
        """
        Initialize SmartResize operator.
        
        Args:
            target_size: Target size [height, width] or single value
            min_scale: Minimum allowed scale factor
            max_scale: Maximum allowed scale factor
            quality_threshold: Quality threshold for choosing interpolation method
        """
        if isinstance(target_size, int):
            self.target_size = [target_size, target_size]
        else:
            self.target_size = list(target_size)
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.quality_threshold = quality_threshold
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smart resize image in input data.
        
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
            
            # Smart resize image
            resized_img, resize_info = self._smart_resize(img)
            data['image'] = resized_img
            data['smart_resize_info'] = resize_info
            
            return data
            
        except Exception as e:
            logger.error(f"Smart resize failed: {e}")
            data['smart_resize_error'] = str(e)
            return data
    
    def _smart_resize(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Smart resize with automatic parameter selection."""
        try:
            h, w = img.shape[:2]
            target_h, target_w = self.target_size
            
            # Calculate scale factors
            scale_h = target_h / h
            scale_w = target_w / w
            scale = min(scale_h, scale_w)
            
            # Clamp scale to allowed range
            scale = max(self.min_scale, min(self.max_scale, scale))
            
            # Choose interpolation method based on scale
            if scale > 1.0:
                # Upscaling - use cubic for better quality
                interp = cv2.INTER_CUBIC
            elif scale < self.quality_threshold:
                # Significant downscaling - use area interpolation
                interp = cv2.INTER_AREA
            else:
                # Normal scaling - use linear
                interp = cv2.INTER_LINEAR
            
            # Calculate final dimensions
            final_h = int(h * scale)
            final_w = int(w * scale)
            
            # Resize image
            resized_img = cv2.resize(img, (final_w, final_h), interpolation=interp)
            
            resize_info = {
                'original_shape': (h, w),
                'final_shape': (final_h, final_w),
                'scale': scale,
                'interpolation': interp,
                'interpolation_name': self._get_interp_name(interp),
                'strategy': self._get_strategy_name(scale)
            }
            
            return resized_img, resize_info
            
        except Exception as e:
            logger.error(f"Smart resize computation failed: {e}")
            return img, {}
    
    def _get_interp_name(self, interp: int) -> str:
        """Get interpolation method name."""
        interp_names = {
            cv2.INTER_NEAREST: 'nearest',
            cv2.INTER_LINEAR: 'linear',
            cv2.INTER_CUBIC: 'cubic',
            cv2.INTER_AREA: 'area',
            cv2.INTER_LANCZOS4: 'lanczos4'
        }
        return interp_names.get(interp, 'unknown')
    
    def _get_strategy_name(self, scale: float) -> str:
        """Get strategy name based on scale."""
        if scale > 1.0:
            return 'upscale'
        elif scale < self.quality_threshold:
            return 'significant_downscale'
        else:
            return 'normal_scale'