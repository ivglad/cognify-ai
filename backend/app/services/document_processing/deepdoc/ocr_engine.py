"""
OCR engine for text detection and recognition.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import trio
import numpy as np
import cv2

from app.services.document_processing.deepdoc.vision_models import vision_models_manager
from app.services.document_processing.deepdoc.layout_recognizer import LayoutBlock, LayoutBlockType
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """A detected text region with position and content."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "bbox": self.bbox,
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "metadata": self.metadata or {}
        }


@dataclass
class OCRResult:
    """Complete OCR result for a page or region."""
    text_regions: List[TextRegion]
    full_text: str
    page_number: Optional[int] = None
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text_regions": [region.to_dict() for region in self.text_regions],
            "full_text": self.full_text,
            "page_number": self.page_number,
            "processing_time": self.processing_time,
            "metadata": self.metadata or {}
        }


class OCREngine:
    """
    OCR engine for text detection and recognition using ONNX models.
    """
    
    def __init__(self):
        self.vision_manager = vision_models_manager
        self.confidence_threshold = settings.DEEPDOC_CONFIDENCE_THRESHOLD
        self._initialized = False
        
        # Character mapping for text recognition (simplified)
        self.char_mapping = self._create_character_mapping()
    
    def _create_character_mapping(self) -> Dict[int, str]:
        """Create character mapping for text recognition."""
        # Simplified character set - in real implementation this would be more comprehensive
        chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        chars += "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        
        mapping = {i: char for i, char in enumerate(chars)}
        mapping[len(chars)] = ""  # Blank token
        
        return mapping
    
    async def initialize(self):
        """Initialize the OCR engine."""
        if self._initialized:
            return
        
        try:
            # Initialize vision models manager
            if not self.vision_manager._initialized:
                await self.vision_manager.initialize()
            
            self._initialized = True
            logger.info("OCREngine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OCREngine: {e}")
            raise
    
    async def extract_text_from_image(self, 
                                    image: np.ndarray,
                                    page_number: Optional[int] = None) -> OCRResult:
        """
        Extract text from an image using OCR.
        
        Args:
            image: Input image as numpy array
            page_number: Optional page number
            
        Returns:
            OCR result with detected text regions
        """
        if not self._initialized:
            await self.initialize()
        
        import time
        start_time = time.time()
        
        try:
            logger.debug(f"Starting OCR for {'page ' + str(page_number) if page_number else 'image'}")
            
            # Step 1: Detect text regions
            text_boxes = await self._detect_text_regions(image)
            
            # Step 2: Recognize text in each region
            text_regions = []
            
            for box in text_boxes:
                try:
                    # Extract region from image
                    region_image = self._extract_region(image, box["bbox"])
                    
                    if region_image is not None:
                        # Recognize text in region
                        recognized_text = await self._recognize_text(region_image)
                        
                        if recognized_text and recognized_text.strip():
                            text_region = TextRegion(
                                bbox=box["bbox"],
                                text=recognized_text.strip(),
                                confidence=min(box["confidence"], 0.9),  # Conservative confidence
                                metadata={"detection_confidence": box["confidence"]}
                            )
                            text_regions.append(text_region)
                            
                except Exception as e:
                    logger.debug(f"Failed to process text region: {e}")
                    continue
            
            # Combine all text
            full_text = self._combine_text_regions(text_regions)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text_regions=text_regions,
                full_text=full_text,
                page_number=page_number,
                processing_time=processing_time,
                metadata={
                    "total_regions": len(text_regions),
                    "detection_method": "onnx" if self._models_available() else "fallback"
                }
            )
            
            logger.info(f"OCR completed: {len(text_regions)} regions, {len(full_text)} chars in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"OCR failed: {e}")
            
            return OCRResult(
                text_regions=[],
                full_text="",
                page_number=page_number,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
    async def _detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text regions in image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected text boxes with confidence scores
        """
        try:
            # Check if text detection model is available
            model_info = self.vision_manager.get_model_info("text_detection")
            if not model_info.get("loaded", False):
                logger.warning("Text detection model not available, using fallback")
                return await self._fallback_text_detection(image)
            
            # Preprocess image for model
            input_image = await self.vision_manager.preprocess_image(
                image, 
                target_size=(640, 640),
                normalize=True
            )
            
            if input_image is None:
                logger.error("Image preprocessing failed for text detection")
                return await self._fallback_text_detection(image)
            
            # Run text detection model
            output = await self.vision_manager.run_inference("text_detection", input_image)
            
            if output is None:
                logger.error("Text detection model inference failed")
                return await self._fallback_text_detection(image)
            
            # Post-process detection output
            text_boxes = await self.vision_manager.postprocess_detection(
                output,
                confidence_threshold=self.confidence_threshold
            )
            
            # Scale boxes back to original image size
            scale_x = image.shape[1] / 640
            scale_y = image.shape[0] / 640
            
            scaled_boxes = []
            for box in text_boxes:
                bbox = box["bbox"]
                scaled_bbox = (
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y
                )
                
                scaled_boxes.append({
                    "bbox": scaled_bbox,
                    "confidence": box["confidence"]
                })
            
            return scaled_boxes
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return await self._fallback_text_detection(image)
    
    async def _fallback_text_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback text detection using traditional computer vision.
        
        Args:
            image: Input image
            
        Returns:
            List of detected text regions
        """
        try:
            logger.debug("Using fallback text detection")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_boxes = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area and aspect ratio
                if area < 100:  # Too small
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Heuristic: text regions usually have reasonable aspect ratios
                if aspect_ratio < 0.1 or aspect_ratio > 20:
                    continue
                
                text_boxes.append({
                    "bbox": (float(x), float(y), float(x + w), float(y + h)),
                    "confidence": 0.6  # Default confidence for fallback
                })
            
            return text_boxes
            
        except Exception as e:
            logger.error(f"Fallback text detection failed: {e}")
            return []
    
    def _extract_region(self, 
                       image: np.ndarray, 
                       bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Extract region from image based on bounding box.
        
        Args:
            image: Source image
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Extracted region or None if failed
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Extract region
            region = image[y1:y2, x1:x2]
            
            # Ensure minimum size
            if region.shape[0] < 8 or region.shape[1] < 8:
                return None
            
            return region
            
        except Exception as e:
            logger.debug(f"Region extraction failed: {e}")
            return None
    
    async def _recognize_text(self, region_image: np.ndarray) -> Optional[str]:
        """
        Recognize text in a region image.
        
        Args:
            region_image: Image region containing text
            
        Returns:
            Recognized text or None if failed
        """
        try:
            # Check if text recognition model is available
            model_info = self.vision_manager.get_model_info("text_recognition")
            if not model_info.get("loaded", False):
                logger.debug("Text recognition model not available, using fallback")
                return await self._fallback_text_recognition(region_image)
            
            # Preprocess region for recognition
            input_region = await self._preprocess_text_region(region_image)
            
            if input_region is None:
                return None
            
            # Run text recognition model
            output = await self.vision_manager.run_inference("text_recognition", input_region)
            
            if output is None:
                return await self._fallback_text_recognition(region_image)
            
            # Decode text from model output
            recognized_text = self._decode_text_output(output)
            
            return recognized_text
            
        except Exception as e:
            logger.debug(f"Text recognition failed: {e}")
            return await self._fallback_text_recognition(region_image)
    
    async def _preprocess_text_region(self, region_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess text region for recognition model.
        
        Args:
            region_image: Input region image
            
        Returns:
            Preprocessed image ready for recognition model
        """
        try:
            # Convert to grayscale if needed
            if len(region_image.shape) == 3:
                gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = region_image
            
            # Resize to model input size (32x128 is common for text recognition)
            resized = cv2.resize(gray, (128, 32))
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            # Convert to RGB format for model
            rgb_image = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            
            # Preprocess using vision manager
            input_image = await self.vision_manager.preprocess_image(
                rgb_image,
                target_size=(32, 128),
                normalize=True
            )
            
            return input_image
            
        except Exception as e:
            logger.debug(f"Text region preprocessing failed: {e}")
            return None
    
    def _decode_text_output(self, model_output: np.ndarray) -> str:
        """
        Decode text from recognition model output.
        
        Args:
            model_output: Raw model output
            
        Returns:
            Decoded text string
        """
        try:
            # Simplified decoding - real implementation would be more sophisticated
            if len(model_output.shape) == 3:  # (batch, sequence, vocab)
                sequence = model_output[0]  # Remove batch dimension
                
                # Get most likely character at each position
                char_indices = np.argmax(sequence, axis=1)
                
                # Convert indices to characters
                text = ""
                prev_char = None
                
                for char_idx in char_indices:
                    if char_idx in self.char_mapping:
                        char = self.char_mapping[char_idx]
                        
                        # Simple CTC-like decoding: remove duplicates and blanks
                        if char != prev_char and char != "":
                            text += char
                        prev_char = char
                
                return text.strip()
            
            return ""
            
        except Exception as e:
            logger.debug(f"Text decoding failed: {e}")
            return ""
    
    async def _fallback_text_recognition(self, region_image: np.ndarray) -> Optional[str]:
        """
        Fallback text recognition using simple heuristics.
        
        Args:
            region_image: Region image
            
        Returns:
            Placeholder text or None
        """
        try:
            # This is a very basic fallback - in real implementation you might use:
            # - Tesseract OCR
            # - EasyOCR
            # - Other OCR libraries
            
            # For now, return a placeholder based on region characteristics
            height, width = region_image.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > 5:  # Very wide region
                return "[TITLE_TEXT]"
            elif aspect_ratio > 2:  # Wide region
                return "[HEADER_TEXT]"
            else:  # Regular text
                return "[TEXT_CONTENT]"
                
        except Exception as e:
            logger.debug(f"Fallback text recognition failed: {e}")
            return None
    
    def _combine_text_regions(self, text_regions: List[TextRegion]) -> str:
        """
        Combine text regions into full text with proper spacing.
        
        Args:
            text_regions: List of text regions
            
        Returns:
            Combined text string
        """
        if not text_regions:
            return ""
        
        # Sort regions by reading order (top to bottom, left to right)
        sorted_regions = sorted(text_regions, key=lambda r: (r.bbox[1], r.bbox[0]))
        
        combined_text = []
        prev_y = None
        
        for region in sorted_regions:
            text = region.text.strip()
            if not text:
                continue
            
            # Add line break if this region is significantly below the previous one
            if prev_y is not None:
                y_diff = region.bbox[1] - prev_y
                if y_diff > 20:  # Threshold for new line
                    combined_text.append("\n")
            
            combined_text.append(text)
            prev_y = region.bbox[3]  # Bottom of current region
        
        return " ".join(combined_text)
    
    def _models_available(self) -> bool:
        """Check if OCR models are available."""
        detection_info = self.vision_manager.get_model_info("text_detection")
        recognition_info = self.vision_manager.get_model_info("text_recognition")
        
        return (detection_info.get("loaded", False) and 
                recognition_info.get("loaded", False))
    
    async def extract_text_from_layout_blocks(self, 
                                            blocks: List[LayoutBlock],
                                            image: np.ndarray) -> List[LayoutBlock]:
        """
        Extract text from layout blocks using OCR.
        
        Args:
            blocks: List of layout blocks
            image: Source image
            
        Returns:
            Blocks with OCR text content
        """
        try:
            for block in blocks:
                # Only process text-like blocks
                if block.block_type in [LayoutBlockType.TEXT, LayoutBlockType.TITLE, LayoutBlockType.HEADER]:
                    # Extract region
                    region_image = self._extract_region(image, block.bbox)
                    
                    if region_image is not None:
                        # Recognize text
                        recognized_text = await self._recognize_text(region_image)
                        
                        if recognized_text:
                            block.text_content = recognized_text
                            if not block.metadata:
                                block.metadata = {}
                            block.metadata["ocr_method"] = "onnx" if self._models_available() else "fallback"
            
            return blocks
            
        except Exception as e:
            logger.error(f"OCR text extraction from blocks failed: {e}")
            return blocks
    
    async def get_ocr_stats(self) -> Dict[str, Any]:
        """Get OCR engine statistics."""
        try:
            detection_info = self.vision_manager.get_model_info("text_detection")
            recognition_info = self.vision_manager.get_model_info("text_recognition")
            
            return {
                "initialized": self._initialized,
                "detection_model_available": detection_info.get("loaded", False),
                "recognition_model_available": recognition_info.get("loaded", False),
                "confidence_threshold": self.confidence_threshold,
                "character_set_size": len(self.char_mapping),
                "fallback_available": True
            }
            
        except Exception as e:
            logger.error(f"OCR stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
ocr_engine = OCREngine()