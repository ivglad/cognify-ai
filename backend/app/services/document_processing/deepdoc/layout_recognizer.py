"""
PDF layout recognition using computer vision models.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import trio
import numpy as np
import cv2
from PIL import Image
import pdfplumber

from app.services.document_processing.deepdoc.vision_models import vision_models_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class LayoutBlockType(Enum):
    """Types of layout blocks that can be detected."""
    TEXT = "text"
    TITLE = "title"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    LIST = "list"
    UNKNOWN = "unknown"


@dataclass
class LayoutBlock:
    """A detected layout block with position and type information."""
    block_type: LayoutBlockType
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    page_number: int
    text_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "block_type": self.block_type.value,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "page_number": self.page_number,
            "text_content": self.text_content,
            "metadata": self.metadata or {}
        }
    
    def get_area(self) -> float:
        """Calculate area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class LayoutRecognizer:
    """
    PDF layout recognition system using computer vision models.
    """
    
    def __init__(self):
        self.vision_manager = vision_models_manager
        self.confidence_threshold = settings.DEEPDOC_CONFIDENCE_THRESHOLD
        self.zoom_factor = 3  # High resolution rendering
        self._initialized = False
    
    async def initialize(self):
        """Initialize the layout recognizer."""
        if self._initialized:
            return
        
        try:
            # Initialize vision models manager
            if not self.vision_manager._initialized:
                await self.vision_manager.initialize()
            
            self._initialized = True
            logger.info("LayoutRecognizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LayoutRecognizer: {e}")
            raise
    
    async def analyze_pdf_layout(self, 
                               pdf_bytes: bytes,
                               page_range: Optional[Tuple[int, int]] = None) -> List[LayoutBlock]:
        """
        Analyze PDF layout and detect blocks.
        
        Args:
            pdf_bytes: PDF file content as bytes
            page_range: Optional (start_page, end_page) tuple (1-indexed)
            
        Returns:
            List of detected layout blocks
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info("Starting PDF layout analysis")
            
            # Extract pages as images
            page_images = await self._pdf_to_images(pdf_bytes, page_range)
            
            if not page_images:
                logger.warning("No pages extracted from PDF")
                return []
            
            # Analyze each page
            all_blocks = []
            
            for page_num, page_image in page_images:
                logger.debug(f"Analyzing layout for page {page_num}")
                
                page_blocks = await self._analyze_page_layout(page_image, page_num)
                all_blocks.extend(page_blocks)
            
            logger.info(f"Layout analysis completed: {len(all_blocks)} blocks detected")
            return all_blocks
            
        except Exception as e:
            logger.error(f"PDF layout analysis failed: {e}")
            return []
    
    async def _pdf_to_images(self, 
                           pdf_bytes: bytes,
                           page_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, np.ndarray]]:
        """
        Convert PDF pages to high-resolution images.
        
        Args:
            pdf_bytes: PDF content
            page_range: Optional page range
            
        Returns:
            List of (page_number, image_array) tuples
        """
        try:
            # Use trio to run PDF processing in thread
            page_images = await trio.to_thread.run_sync(
                self._pdf_to_images_sync,
                pdf_bytes,
                page_range
            )
            
            return page_images
            
        except Exception as e:
            logger.error(f"PDF to images conversion failed: {e}")
            return []
    
    def _pdf_to_images_sync(self, 
                          pdf_bytes: bytes,
                          page_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, np.ndarray]]:
        """Convert PDF to images synchronously."""
        import io
        
        page_images = []
        
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                total_pages = len(pdf.pages)
                
                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages
                
                logger.info(f"Converting pages {start_page}-{end_page} to images")
                
                for page_num in range(start_page - 1, end_page):  # Convert to 0-indexed
                    try:
                        page = pdf.pages[page_num]
                        
                        # Render page to image with high resolution
                        pil_image = page.to_image(resolution=150 * self.zoom_factor).original
                        
                        # Convert PIL to numpy array
                        image_array = np.array(pil_image)
                        
                        # Convert RGBA to RGB if necessary
                        if image_array.shape[2] == 4:
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                        elif image_array.shape[2] == 1:
                            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                        
                        page_images.append((page_num + 1, image_array))  # Convert back to 1-indexed
                        
                    except Exception as e:
                        logger.error(f"Failed to convert page {page_num + 1}: {e}")
                        continue
            
            return page_images
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return []
    
    async def _analyze_page_layout(self, 
                                 page_image: np.ndarray, 
                                 page_number: int) -> List[LayoutBlock]:
        """
        Analyze layout of a single page.
        
        Args:
            page_image: Page image as numpy array
            page_number: Page number (1-indexed)
            
        Returns:
            List of layout blocks for this page
        """
        try:
            # Check if layout model is available
            model_info = self.vision_manager.get_model_info("layout")
            if not model_info.get("loaded", False):
                logger.warning("Layout model not available, using fallback detection")
                return await self._fallback_layout_detection(page_image, page_number)
            
            # Preprocess image for model
            input_image = await self.vision_manager.preprocess_image(
                page_image, 
                target_size=(1024, 1024),
                normalize=True
            )
            
            if input_image is None:
                logger.error("Image preprocessing failed")
                return []
            
            # Run layout detection model
            output = await self.vision_manager.run_inference("layout", input_image)
            
            if output is None:
                logger.error("Layout model inference failed")
                return await self._fallback_layout_detection(page_image, page_number)
            
            # Post-process model output
            blocks = await self._postprocess_layout_output(
                output, 
                page_image.shape[:2], 
                page_number
            )
            
            return blocks
            
        except Exception as e:
            logger.error(f"Page layout analysis failed: {e}")
            return await self._fallback_layout_detection(page_image, page_number)
    
    async def _postprocess_layout_output(self, 
                                       model_output: np.ndarray,
                                       original_size: Tuple[int, int],
                                       page_number: int) -> List[LayoutBlock]:
        """
        Post-process layout model output to extract blocks.
        
        Args:
            model_output: Raw model output
            original_size: Original image size (height, width)
            page_number: Page number
            
        Returns:
            List of layout blocks
        """
        try:
            blocks = []
            
            # Assuming model output is segmentation mask with shape (1, num_classes, H, W)
            if len(model_output.shape) == 4:
                batch_size, num_classes, height, width = model_output.shape
                
                # Scale factors to convert from model coordinates to original image
                scale_y = original_size[0] / height
                scale_x = original_size[1] / width
                
                # Define class mapping
                class_mapping = {
                    0: LayoutBlockType.TEXT,
                    1: LayoutBlockType.TITLE,
                    2: LayoutBlockType.TABLE,
                    3: LayoutBlockType.FIGURE,
                    4: LayoutBlockType.HEADER
                }
                
                # Process each class
                for class_id in range(min(num_classes, len(class_mapping))):
                    class_mask = model_output[0, class_id]  # Remove batch dimension
                    
                    # Threshold the mask
                    binary_mask = (class_mask > self.confidence_threshold).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(
                        binary_mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Convert contours to bounding boxes
                    for contour in contours:
                        if cv2.contourArea(contour) < 100:  # Filter small regions
                            continue
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Scale to original image coordinates
                        x1 = x * scale_x
                        y1 = y * scale_y
                        x2 = (x + w) * scale_x
                        y2 = (y + h) * scale_y
                        
                        # Calculate confidence (average mask value in region)
                        mask_region = class_mask[y:y+h, x:x+w]
                        confidence = float(np.mean(mask_region))
                        
                        if confidence >= self.confidence_threshold:
                            block = LayoutBlock(
                                block_type=class_mapping.get(class_id, LayoutBlockType.UNKNOWN),
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                page_number=page_number
                            )
                            blocks.append(block)
            
            # Sort blocks by reading order (top to bottom, left to right)
            blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            
            return blocks
            
        except Exception as e:
            logger.error(f"Layout output postprocessing failed: {e}")
            return []
    
    async def _fallback_layout_detection(self, 
                                       page_image: np.ndarray, 
                                       page_number: int) -> List[LayoutBlock]:
        """
        Fallback layout detection using traditional computer vision.
        
        Args:
            page_image: Page image
            page_number: Page number
            
        Returns:
            List of detected blocks
        """
        try:
            logger.debug(f"Using fallback layout detection for page {page_number}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(page_image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            blocks = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < 1000:  # Skip very small regions
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic heuristics for block type classification
                aspect_ratio = w / h
                relative_area = area / (page_image.shape[0] * page_image.shape[1])
                
                # Classify based on simple heuristics
                if y < page_image.shape[0] * 0.1:  # Top 10% of page
                    block_type = LayoutBlockType.HEADER
                elif y > page_image.shape[0] * 0.9:  # Bottom 10% of page
                    block_type = LayoutBlockType.FOOTER
                elif aspect_ratio > 3 and h < 50:  # Wide and short
                    block_type = LayoutBlockType.TITLE
                elif aspect_ratio < 0.5 and relative_area > 0.1:  # Tall and large
                    block_type = LayoutBlockType.FIGURE
                else:
                    block_type = LayoutBlockType.TEXT
                
                block = LayoutBlock(
                    block_type=block_type,
                    bbox=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=0.7,  # Default confidence for fallback
                    page_number=page_number,
                    metadata={"detection_method": "fallback"}
                )
                
                blocks.append(block)
            
            # Sort by reading order
            blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            
            return blocks
            
        except Exception as e:
            logger.error(f"Fallback layout detection failed: {e}")
            return []
    
    async def extract_text_from_blocks(self, 
                                     blocks: List[LayoutBlock],
                                     pdf_bytes: bytes) -> List[LayoutBlock]:
        """
        Extract text content from layout blocks using pdfplumber.
        
        Args:
            blocks: List of layout blocks
            pdf_bytes: Original PDF content
            
        Returns:
            Blocks with text content filled
        """
        try:
            # Extract text using pdfplumber in thread
            blocks_with_text = await trio.to_thread.run_sync(
                self._extract_text_sync,
                blocks,
                pdf_bytes
            )
            
            return blocks_with_text
            
        except Exception as e:
            logger.error(f"Text extraction from blocks failed: {e}")
            return blocks
    
    def _extract_text_sync(self, 
                          blocks: List[LayoutBlock], 
                          pdf_bytes: bytes) -> List[LayoutBlock]:
        """Extract text from blocks synchronously."""
        import io
        
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                # Group blocks by page
                blocks_by_page = {}
                for block in blocks:
                    page_num = block.page_number
                    if page_num not in blocks_by_page:
                        blocks_by_page[page_num] = []
                    blocks_by_page[page_num].append(block)
                
                # Extract text for each page
                for page_num, page_blocks in blocks_by_page.items():
                    if page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]  # Convert to 0-indexed
                        
                        for block in page_blocks:
                            # Extract text from block region
                            x1, y1, x2, y2 = block.bbox
                            
                            # Convert coordinates to pdfplumber format if needed
                            # This is a simplified approach - real implementation would need coordinate transformation
                            try:
                                cropped_page = page.crop((x1, y1, x2, y2))
                                text = cropped_page.extract_text()
                                
                                if text and text.strip():
                                    block.text_content = text.strip()
                                    
                            except Exception as e:
                                logger.debug(f"Text extraction failed for block: {e}")
                                continue
            
            return blocks
            
        except Exception as e:
            logger.error(f"Synchronous text extraction failed: {e}")
            return blocks
    
    async def get_layout_stats(self) -> Dict[str, Any]:
        """Get layout recognition statistics."""
        try:
            model_info = self.vision_manager.get_model_info("layout")
            
            return {
                "initialized": self._initialized,
                "model_available": model_info.get("loaded", False),
                "confidence_threshold": self.confidence_threshold,
                "zoom_factor": self.zoom_factor,
                "supported_block_types": [bt.value for bt in LayoutBlockType],
                "fallback_available": True
            }
            
        except Exception as e:
            logger.error(f"Layout stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
layout_recognizer = LayoutRecognizer()