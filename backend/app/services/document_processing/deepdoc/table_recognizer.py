"""
Table structure recognition and reconstruction system.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import trio
import numpy as np
import cv2
import pandas as pd

from app.services.document_processing.deepdoc.vision_models import vision_models_manager
from app.services.document_processing.deepdoc.layout_recognizer import LayoutBlock, LayoutBlockType
from app.services.document_processing.deepdoc.ocr_engine import ocr_engine
from app.core.config import settings

logger = logging.getLogger(__name__)


class TableCellType(Enum):
    """Types of table cells."""
    HEADER = "header"
    DATA = "data"
    MERGED = "merged"
    EMPTY = "empty"


@dataclass
class TableCell:
    """A table cell with position and content."""
    row: int
    col: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text: str
    cell_type: TableCellType
    confidence: float
    rowspan: int = 1
    colspan: int = 1
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "row": self.row,
            "col": self.col,
            "bbox": self.bbox,
            "text": self.text,
            "cell_type": self.cell_type.value,
            "confidence": self.confidence,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "metadata": self.metadata or {}
        }


@dataclass
class TableStructure:
    """Complete table structure with cells and metadata."""
    cells: List[TableCell]
    num_rows: int
    num_cols: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    page_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cells": [cell.to_dict() for cell in self.cells],
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "page_number": self.page_number,
            "metadata": self.metadata or {}
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert table to pandas DataFrame."""
        try:
            # Create empty DataFrame with correct dimensions
            df = pd.DataFrame(index=range(self.num_rows), columns=range(self.num_cols))
            df = df.fillna("")
            
            # Fill cells
            for cell in self.cells:
                if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                    df.iloc[cell.row, cell.col] = cell.text
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame conversion failed: {e}")
            return pd.DataFrame()
    
    def to_html(self) -> str:
        """Convert table to HTML format."""
        try:
            df = self.to_dataframe()
            
            if df.empty:
                return "<table></table>"
            
            # Convert to HTML with proper styling
            html = df.to_html(
                index=False,
                classes="table table-striped",
                table_id=f"table_{id(self)}",
                escape=False
            )
            
            return html
            
        except Exception as e:
            logger.error(f"HTML conversion failed: {e}")
            return "<table></table>"
    
    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        try:
            df = self.to_dataframe()
            
            if df.empty:
                return ""
            
            # Convert to markdown
            markdown = df.to_markdown(index=False)
            
            return markdown
            
        except Exception as e:
            logger.error(f"Markdown conversion failed: {e}")
            return ""


class TableRecognizer:
    """
    Table structure recognition system using computer vision and OCR.
    """
    
    def __init__(self):
        self.vision_manager = vision_models_manager
        self.ocr_engine = ocr_engine
        self.confidence_threshold = settings.DEEPDOC_CONFIDENCE_THRESHOLD
        self._initialized = False
    
    async def initialize(self):
        """Initialize the table recognizer."""
        if self._initialized:
            return
        
        try:
            # Initialize vision models manager
            if not self.vision_manager._initialized:
                await self.vision_manager.initialize()
            
            # Initialize OCR engine
            if not self.ocr_engine._initialized:
                await self.ocr_engine.initialize()
            
            self._initialized = True
            logger.info("TableRecognizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TableRecognizer: {e}")
            raise
    
    async def recognize_table_structure(self, 
                                      table_block: LayoutBlock,
                                      image: np.ndarray) -> Optional[TableStructure]:
        """
        Recognize structure of a table block.
        
        Args:
            table_block: Layout block identified as table
            image: Source image containing the table
            
        Returns:
            Recognized table structure or None if failed
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.debug(f"Recognizing table structure for block on page {table_block.page_number}")
            
            # Extract table region from image
            table_image = self._extract_table_region(image, table_block.bbox)
            
            if table_image is None:
                logger.error("Failed to extract table region")
                return None
            
            # Detect table structure
            structure_info = await self._detect_table_structure(table_image)
            
            if not structure_info:
                logger.warning("No table structure detected, using fallback")
                structure_info = await self._fallback_table_detection(table_image)
            
            # Extract text from table cells
            table_structure = await self._extract_table_text(
                table_image, 
                structure_info, 
                table_block
            )
            
            return table_structure
            
        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            return None
    
    def _extract_table_region(self, 
                            image: np.ndarray, 
                            bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Extract table region from image.
        
        Args:
            image: Source image
            bbox: Table bounding box
            
        Returns:
            Extracted table image or None if failed
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
            
            # Extract region with some padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            table_region = image[y1:y2, x1:x2]
            
            return table_region
            
        except Exception as e:
            logger.error(f"Table region extraction failed: {e}")
            return None
    
    async def _detect_table_structure(self, table_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect table structure using ONNX model.
        
        Args:
            table_image: Table region image
            
        Returns:
            Table structure information or None if failed
        """
        try:
            # Check if table structure model is available
            model_info = self.vision_manager.get_model_info("table_structure")
            if not model_info.get("loaded", False):
                logger.warning("Table structure model not available")
                return None
            
            # Preprocess image for model
            input_image = await self.vision_manager.preprocess_image(
                table_image,
                target_size=(512, 512),
                normalize=True
            )
            
            if input_image is None:
                logger.error("Table image preprocessing failed")
                return None
            
            # Run table structure model
            output = await self.vision_manager.run_inference("table_structure", input_image)
            
            if output is None:
                logger.error("Table structure model inference failed")
                return None
            
            # Post-process model output
            structure_info = await self._postprocess_table_output(output, table_image.shape[:2])
            
            return structure_info
            
        except Exception as e:
            logger.error(f"Table structure detection failed: {e}")
            return None
    
    async def _postprocess_table_output(self, 
                                      model_output: np.ndarray,
                                      original_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Post-process table structure model output.
        
        Args:
            model_output: Raw model output
            original_size: Original image size (height, width)
            
        Returns:
            Processed structure information
        """
        try:
            # Assuming model output is segmentation mask with different classes for table components
            if len(model_output.shape) == 4:  # (batch, classes, height, width)
                batch_size, num_classes, height, width = model_output.shape
                
                # Scale factors
                scale_y = original_size[0] / height
                scale_x = original_size[1] / width
                
                # Extract different table components
                structure_info = {
                    "cells": [],
                    "rows": [],
                    "columns": [],
                    "grid_lines": []
                }
                
                # Process each class (simplified approach)
                for class_id in range(min(num_classes, 6)):  # Assuming 6 classes max
                    class_mask = model_output[0, class_id]
                    
                    # Threshold the mask
                    binary_mask = (class_mask > self.confidence_threshold).astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(
                        binary_mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    # Process contours based on class
                    for contour in contours:
                        if cv2.contourArea(contour) < 50:  # Filter small regions
                            continue
                        
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Scale to original coordinates
                        scaled_bbox = (
                            x * scale_x,
                            y * scale_y,
                            (x + w) * scale_x,
                            (y + h) * scale_y
                        )
                        
                        confidence = float(np.mean(class_mask[y:y+h, x:x+w]))
                        
                        if class_id == 0:  # Cell regions
                            structure_info["cells"].append({
                                "bbox": scaled_bbox,
                                "confidence": confidence
                            })
                        elif class_id == 1:  # Row separators
                            structure_info["rows"].append({
                                "bbox": scaled_bbox,
                                "confidence": confidence
                            })
                        elif class_id == 2:  # Column separators
                            structure_info["columns"].append({
                                "bbox": scaled_bbox,
                                "confidence": confidence
                            })
                
                return structure_info
            
            return None
            
        except Exception as e:
            logger.error(f"Table output postprocessing failed: {e}")
            return None
    
    async def _fallback_table_detection(self, table_image: np.ndarray) -> Dict[str, Any]:
        """
        Fallback table structure detection using traditional computer vision.
        
        Args:
            table_image: Table region image
            
        Returns:
            Basic table structure information
        """
        try:
            logger.debug("Using fallback table structure detection")
            
            # Convert to grayscale
            gray = cv2.cvtColor(table_image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Detect horizontal and vertical lines
            horizontal_lines = self._detect_lines(binary, "horizontal")
            vertical_lines = self._detect_lines(binary, "vertical")
            
            # Create grid from lines
            cells = self._create_cell_grid(horizontal_lines, vertical_lines, table_image.shape[:2])
            
            structure_info = {
                "cells": cells,
                "rows": horizontal_lines,
                "columns": vertical_lines,
                "grid_lines": horizontal_lines + vertical_lines,
                "detection_method": "fallback"
            }
            
            return structure_info
            
        except Exception as e:
            logger.error(f"Fallback table detection failed: {e}")
            return {
                "cells": [],
                "rows": [],
                "columns": [],
                "grid_lines": []
            }
    
    def _detect_lines(self, binary_image: np.ndarray, direction: str) -> List[Dict[str, Any]]:
        """
        Detect horizontal or vertical lines in binary image.
        
        Args:
            binary_image: Binary image
            direction: "horizontal" or "vertical"
            
        Returns:
            List of detected lines
        """
        try:
            if direction == "horizontal":
                # Create horizontal kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            else:
                # Create vertical kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Apply morphological operations
            lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_lines = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio
                if direction == "horizontal" and w > h * 5:
                    detected_lines.append({
                        "bbox": (float(x), float(y), float(x + w), float(y + h)),
                        "confidence": 0.8,
                        "direction": direction
                    })
                elif direction == "vertical" and h > w * 5:
                    detected_lines.append({
                        "bbox": (float(x), float(y), float(x + w), float(y + h)),
                        "confidence": 0.8,
                        "direction": direction
                    })
            
            return detected_lines
            
        except Exception as e:
            logger.error(f"Line detection failed: {e}")
            return []
    
    def _create_cell_grid(self, 
                         horizontal_lines: List[Dict[str, Any]],
                         vertical_lines: List[Dict[str, Any]],
                         image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Create cell grid from detected lines.
        
        Args:
            horizontal_lines: Detected horizontal lines
            vertical_lines: Detected vertical lines
            image_size: Image size (height, width)
            
        Returns:
            List of cell regions
        """
        try:
            # Extract line positions
            h_positions = []
            for line in horizontal_lines:
                y = (line["bbox"][1] + line["bbox"][3]) / 2
                h_positions.append(y)
            
            v_positions = []
            for line in vertical_lines:
                x = (line["bbox"][0] + line["bbox"][2]) / 2
                v_positions.append(x)
            
            # Add image boundaries
            h_positions.extend([0, image_size[0]])
            v_positions.extend([0, image_size[1]])
            
            # Sort positions
            h_positions = sorted(set(h_positions))
            v_positions = sorted(set(v_positions))
            
            # Create cells
            cells = []
            
            for i in range(len(h_positions) - 1):
                for j in range(len(v_positions) - 1):
                    x1 = v_positions[j]
                    y1 = h_positions[i]
                    x2 = v_positions[j + 1]
                    y2 = h_positions[i + 1]
                    
                    # Skip very small cells
                    if (x2 - x1) < 20 or (y2 - y1) < 10:
                        continue
                    
                    cells.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": 0.7,
                        "row": i,
                        "col": j
                    })
            
            return cells
            
        except Exception as e:
            logger.error(f"Cell grid creation failed: {e}")
            return []
    
    async def _extract_table_text(self, 
                                table_image: np.ndarray,
                                structure_info: Dict[str, Any],
                                table_block: LayoutBlock) -> Optional[TableStructure]:
        """
        Extract text from table cells using OCR.
        
        Args:
            table_image: Table region image
            structure_info: Table structure information
            table_block: Original table block
            
        Returns:
            Complete table structure with text
        """
        try:
            cells_info = structure_info.get("cells", [])
            
            if not cells_info:
                logger.warning("No cells found in table structure")
                return None
            
            # Extract text from each cell
            table_cells = []
            
            for i, cell_info in enumerate(cells_info):
                cell_bbox = cell_info["bbox"]
                
                # Extract cell region
                cell_image = self._extract_cell_region(table_image, cell_bbox)
                
                if cell_image is not None:
                    # Use OCR to extract text
                    cell_text = await self.ocr_engine._recognize_text(cell_image)
                    
                    if not cell_text:
                        cell_text = ""
                    
                    # Determine cell type (simplified)
                    cell_type = self._classify_cell_type(cell_text, cell_bbox, i)
                    
                    # Create table cell
                    table_cell = TableCell(
                        row=cell_info.get("row", i // 5),  # Estimate row
                        col=cell_info.get("col", i % 5),   # Estimate col
                        bbox=cell_bbox,
                        text=cell_text.strip(),
                        cell_type=cell_type,
                        confidence=cell_info.get("confidence", 0.7)
                    )
                    
                    table_cells.append(table_cell)
            
            # Determine table dimensions
            if table_cells:
                max_row = max(cell.row for cell in table_cells) + 1
                max_col = max(cell.col for cell in table_cells) + 1
            else:
                max_row = max_col = 0
            
            # Create table structure
            table_structure = TableStructure(
                cells=table_cells,
                num_rows=max_row,
                num_cols=max_col,
                bbox=table_block.bbox,
                confidence=table_block.confidence,
                page_number=table_block.page_number,
                metadata={
                    "detection_method": structure_info.get("detection_method", "onnx"),
                    "total_cells": len(table_cells)
                }
            )
            
            return table_structure
            
        except Exception as e:
            logger.error(f"Table text extraction failed: {e}")
            return None
    
    def _extract_cell_region(self, 
                           table_image: np.ndarray, 
                           cell_bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """
        Extract cell region from table image.
        
        Args:
            table_image: Table image
            cell_bbox: Cell bounding box
            
        Returns:
            Cell region image or None if failed
        """
        try:
            x1, y1, x2, y2 = cell_bbox
            
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(table_image.shape[1], int(x2))
            y2 = min(table_image.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            cell_region = table_image[y1:y2, x1:x2]
            
            # Ensure minimum size
            if cell_region.shape[0] < 5 or cell_region.shape[1] < 5:
                return None
            
            return cell_region
            
        except Exception as e:
            logger.debug(f"Cell region extraction failed: {e}")
            return None
    
    def _classify_cell_type(self, 
                          text: str, 
                          bbox: Tuple[float, float, float, float], 
                          cell_index: int) -> TableCellType:
        """
        Classify cell type based on content and position.
        
        Args:
            text: Cell text content
            bbox: Cell bounding box
            cell_index: Cell index in table
            
        Returns:
            Classified cell type
        """
        try:
            if not text or not text.strip():
                return TableCellType.EMPTY
            
            # Simple heuristics for cell type classification
            text_lower = text.lower().strip()
            
            # Check if it's likely a header (first few cells, bold-like text)
            if cell_index < 10:  # First row likely headers
                return TableCellType.HEADER
            
            # Check for common header patterns
            header_indicators = ["total", "sum", "average", "count", "name", "type", "date", "id"]
            if any(indicator in text_lower for indicator in header_indicators):
                return TableCellType.HEADER
            
            return TableCellType.DATA
            
        except Exception as e:
            logger.debug(f"Cell type classification failed: {e}")
            return TableCellType.DATA
    
    async def process_table_blocks(self, 
                                 table_blocks: List[LayoutBlock],
                                 image: np.ndarray) -> List[TableStructure]:
        """
        Process multiple table blocks in an image.
        
        Args:
            table_blocks: List of table layout blocks
            image: Source image
            
        Returns:
            List of recognized table structures
        """
        try:
            recognized_tables = []
            
            for table_block in table_blocks:
                if table_block.block_type == LayoutBlockType.TABLE:
                    table_structure = await self.recognize_table_structure(table_block, image)
                    
                    if table_structure:
                        recognized_tables.append(table_structure)
            
            logger.info(f"Processed {len(table_blocks)} table blocks, recognized {len(recognized_tables)} tables")
            
            return recognized_tables
            
        except Exception as e:
            logger.error(f"Table blocks processing failed: {e}")
            return []
    
    async def get_table_stats(self) -> Dict[str, Any]:
        """Get table recognizer statistics."""
        try:
            model_info = self.vision_manager.get_model_info("table_structure")
            ocr_stats = await self.ocr_engine.get_ocr_stats()
            
            return {
                "initialized": self._initialized,
                "table_model_available": model_info.get("loaded", False),
                "ocr_available": ocr_stats.get("initialized", False),
                "confidence_threshold": self.confidence_threshold,
                "supported_formats": ["html", "markdown", "dataframe"],
                "fallback_available": True
            }
            
        except Exception as e:
            logger.error(f"Table stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
table_recognizer = TableRecognizer()