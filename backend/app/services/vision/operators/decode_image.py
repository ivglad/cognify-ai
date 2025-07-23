"""
DecodeImage operator for decoding images from various formats.
"""
import cv2
import numpy as np
from typing import Dict, Any, Union, Optional
from pathlib import Path

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class DecodeImage:
    """
    Decode image from bytes, file path, or numpy array.
    Supports multiple image formats and provides fallback mechanisms.
    """
    
    def __init__(self, 
                 to_rgb: bool = True,
                 channel_first: bool = False,
                 ignore_orientation: bool = False):
        """
        Initialize DecodeImage operator.
        
        Args:
            to_rgb: Convert BGR to RGB
            channel_first: Return image in CHW format instead of HWC
            ignore_orientation: Ignore EXIF orientation data
        """
        self.to_rgb = to_rgb
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode image from input data.
        
        Args:
            data: Dictionary containing image data under 'image' key
            
        Returns:
            Dictionary with decoded image under 'image' key
        """
        try:
            img = data.get('image')
            
            if img is None:
                raise ValueError("No image data found in input")
            
            # Decode based on input type
            if isinstance(img, bytes):
                decoded_img = self._decode_from_bytes(img)
            elif isinstance(img, str):
                decoded_img = self._decode_from_path(img)
            elif isinstance(img, np.ndarray):
                decoded_img = self._process_numpy_array(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Apply post-processing
            if decoded_img is not None:
                decoded_img = self._post_process(decoded_img)
                data['image'] = decoded_img
            else:
                raise ValueError("Failed to decode image")
            
            return data
            
        except Exception as e:
            logger.error(f"Image decoding failed: {e}")
            # Return original data with error flag
            data['decode_error'] = str(e)
            return data
    
    def _decode_from_bytes(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Decode image from bytes."""
        try:
            # Convert bytes to numpy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            
            # Decode using OpenCV
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                # Try alternative decoding methods
                logger.warning("OpenCV decoding failed, trying alternative methods")
                return self._fallback_decode_bytes(img_bytes)
            
            return img
            
        except Exception as e:
            logger.error(f"Bytes decoding failed: {e}")
            return None
    
    def _decode_from_path(self, img_path: str) -> Optional[np.ndarray]:
        """Decode image from file path."""
        try:
            path = Path(img_path)
            
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Read image using OpenCV
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            
            if img is None:
                # Try alternative reading methods
                logger.warning(f"OpenCV reading failed for {img_path}, trying alternatives")
                return self._fallback_decode_path(img_path)
            
            return img
            
        except Exception as e:
            logger.error(f"Path decoding failed: {e}")
            return None
    
    def _process_numpy_array(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Process numpy array image."""
        try:
            # Ensure image is in correct format
            if len(img.shape) == 2:
                # Grayscale to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3:
                if img.shape[2] == 1:
                    # Single channel to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    # RGBA to BGR
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Ensure uint8 dtype
            if img.dtype != np.uint8:
                if img.dtype == np.float32 or img.dtype == np.float64:
                    # Assume normalized values [0, 1]
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            return img
            
        except Exception as e:
            logger.error(f"Numpy array processing failed: {e}")
            return None
    
    def _fallback_decode_bytes(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Fallback decoding methods for bytes."""
        try:
            # Try PIL as fallback
            try:
                from PIL import Image
                import io
                
                img_pil = Image.open(io.BytesIO(img_bytes))
                img_array = np.array(img_pil)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                return img_array
                
            except ImportError:
                logger.warning("PIL not available for fallback decoding")
            except Exception as e:
                logger.warning(f"PIL fallback failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback bytes decoding failed: {e}")
            return None
    
    def _fallback_decode_path(self, img_path: str) -> Optional[np.ndarray]:
        """Fallback decoding methods for file paths."""
        try:
            # Try PIL as fallback
            try:
                from PIL import Image
                
                img_pil = Image.open(img_path)
                img_array = np.array(img_pil)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                return img_array
                
            except ImportError:
                logger.warning("PIL not available for fallback decoding")
            except Exception as e:
                logger.warning(f"PIL fallback failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback path decoding failed: {e}")
            return None
    
    def _post_process(self, img: np.ndarray) -> np.ndarray:
        """Apply post-processing to decoded image."""
        try:
            # Convert BGR to RGB if requested
            if self.to_rgb and len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to channel-first format if requested
            if self.channel_first and len(img.shape) == 3:
                img = img.transpose((2, 0, 1))  # HWC to CHW
            
            return img
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return img