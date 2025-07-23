"""
Integrated DeepDoc processor for advanced document understanding.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import time

import trio

from app.services.document_processing.deepdoc.vision_models import vision_models_manager
from app.services.document_processing.deepdoc.layout_recognizer import layout_recognizer, LayoutBlock, LayoutBlockType
from app.services.document_processing.deepdoc.ocr_engine import ocr_engine, OCRResult
from app.services.document_processing.deepdoc.table_recognizer import table_recognizer, TableStructure
from app.core.config import settings

logger = logging.getLogger(__name__)


class DeepDocProcessor:
    """
    Integrated DeepDoc processor that combines layout recognition, OCR, and table processing.
    """
    
    def __init__(self):
        self.vision_manager = vision_models_manager
        self.layout_recognizer = layout_recognizer
        self.ocr_engine = ocr_engine
        self.table_recognizer = table_recognizer
        self._initialized = False
        
    async def initialize(self):
        """Initialize the DeepDoc processor."""
        if self._initialized:
            return
        
        try:
            # Initialize all components
            if not self.vision_manager._initialized:
                await self.vision_manager.initialize()
            
            if not self.layout_recognizer._initialized:
                await self.layout_recognizer.initialize()
            
            if not self.ocr_engine._initialized:
                await self.ocr_engine.initialize()
            
            if not self.table_recognizer._initialized:
                await self.table_recognizer.initialize()
            
            self._initialized = True
            logger.info("DeepDocProcessor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepDocProcessor: {e}")
            # Don't raise - allow system to work without DeepDoc
            self._initialized = True
    
    async def process_pdf_document(self, 
                                 pdf_bytes: bytes,
                                 document_id: str,
                                 page_range: Optional[Tuple[int, int]] = None,
                                 enable_ocr: bool = True,
                                 enable_tables: bool = True) -> Dict[str, Any]:
        """
        Process PDF document with full DeepDoc pipeline.
        
        Args:
            pdf_bytes: PDF file content
            document_id: Document identifier
            page_range: Optional page range to process
            enable_ocr: Whether to perform OCR
            enable_tables: Whether to process tables
            
        Returns:
            Complete processing results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"Starting DeepDoc processing for document {document_id}")
            
            # Check if DeepDoc is enabled
            if not settings.DEEPDOC_ENABLED:
                logger.info("DeepDoc is disabled, using fallback processing")
                return await self._fallback_processing(pdf_bytes, document_id)
            
            # Step 1: Layout Recognition
            logger.debug("Step 1: Layout recognition")
            layout_blocks = await self.layout_recognizer.analyze_pdf_layout(pdf_bytes, page_range)
            
            if not layout_blocks:
                logger.warning("No layout blocks detected, using fallback")
                return await self._fallback_processing(pdf_bytes, document_id)
            
            # Step 2: OCR Processing (if enabled)
            ocr_results = []
            if enable_ocr:
                logger.debug("Step 2: OCR processing")
                ocr_results = await self._process_ocr(layout_blocks, pdf_bytes)
            
            # Step 3: Table Processing (if enabled)
            table_structures = []
            if enable_tables:
                logger.debug("Step 3: Table processing")
                table_structures = await self._process_tables(layout_blocks, pdf_bytes)
            
            # Step 4: Combine Results
            logger.debug("Step 4: Combining results")
            combined_results = await self._combine_results(
                layout_blocks, 
                ocr_results, 
                table_structures,
                document_id
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"DeepDoc processing completed in {processing_time:.3f}s: "
                       f"{len(layout_blocks)} blocks, {len(table_structures)} tables")
            
            return {
                "success": True,
                "document_id": document_id,
                "processing_time": processing_time,
                "layout_blocks": [block.to_dict() for block in layout_blocks],
                "ocr_results": [result.to_dict() for result in ocr_results],
                "table_structures": [table.to_dict() for table in table_structures],
                "combined_content": combined_results,
                "stats": {
                    "total_blocks": len(layout_blocks),
                    "text_blocks": len([b for b in layout_blocks if b.block_type == LayoutBlockType.TEXT]),
                    "table_blocks": len([b for b in layout_blocks if b.block_type == LayoutBlockType.TABLE]),
                    "title_blocks": len([b for b in layout_blocks if b.block_type == LayoutBlockType.TITLE]),
                    "figure_blocks": len([b for b in layout_blocks if b.block_type == LayoutBlockType.FIGURE]),
                    "tables_processed": len(table_structures),
                    "ocr_pages": len(ocr_results)
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"DeepDoc processing failed: {e}")
            
            # Try fallback processing
            try:
                fallback_result = await self._fallback_processing(pdf_bytes, document_id)
                fallback_result["processing_time"] = processing_time
                fallback_result["error"] = str(e)
                fallback_result["fallback_used"] = True
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback processing also failed: {fallback_error}")
                
                return {
                    "success": False,
                    "document_id": document_id,
                    "processing_time": processing_time,
                    "error": str(e),
                    "fallback_error": str(fallback_error),
                    "layout_blocks": [],
                    "ocr_results": [],
                    "table_structures": [],
                    "combined_content": []
                }
    
    async def _process_ocr(self, 
                         layout_blocks: List[LayoutBlock],
                         pdf_bytes: bytes) -> List[OCRResult]:
        """
        Process OCR for layout blocks.
        
        Args:
            layout_blocks: Detected layout blocks
            pdf_bytes: Original PDF content
            
        Returns:
            List of OCR results
        """
        try:
            # Group blocks by page
            blocks_by_page = {}
            for block in layout_blocks:
                page_num = block.page_number
                if page_num not in blocks_by_page:
                    blocks_by_page[page_num] = []
                blocks_by_page[page_num].append(block)
            
            # Extract text using pdfplumber first (faster and more accurate for text)
            blocks_with_text = await self.layout_recognizer.extract_text_from_blocks(
                layout_blocks, 
                pdf_bytes
            )
            
            # For blocks without text or with low confidence, use OCR
            ocr_results = []
            
            # This is a simplified approach - in real implementation, you'd render PDF pages to images
            # and then apply OCR to specific regions
            for page_num in blocks_by_page.keys():
                page_blocks = blocks_by_page[page_num]
                
                # Create OCR result for page
                text_regions = []
                full_text_parts = []
                
                for block in page_blocks:
                    if block.text_content:
                        from app.services.document_processing.deepdoc.ocr_engine import TextRegion
                        
                        text_region = TextRegion(
                            bbox=block.bbox,
                            text=block.text_content,
                            confidence=block.confidence
                        )
                        text_regions.append(text_region)
                        full_text_parts.append(block.text_content)
                
                if text_regions:
                    ocr_result = OCRResult(
                        text_regions=text_regions,
                        full_text="\n".join(full_text_parts),
                        page_number=page_num,
                        metadata={"extraction_method": "pdfplumber"}
                    )
                    ocr_results.append(ocr_result)
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return []
    
    async def _process_tables(self, 
                            layout_blocks: List[LayoutBlock],
                            pdf_bytes: bytes) -> List[TableStructure]:
        """
        Process tables from layout blocks.
        
        Args:
            layout_blocks: Detected layout blocks
            pdf_bytes: Original PDF content
            
        Returns:
            List of table structures
        """
        try:
            # Find table blocks
            table_blocks = [block for block in layout_blocks if block.block_type == LayoutBlockType.TABLE]
            
            if not table_blocks:
                return []
            
            # This is a simplified approach - in real implementation, you'd need to:
            # 1. Render PDF pages to images
            # 2. Extract table regions
            # 3. Apply table recognition
            
            # For now, create basic table structures from layout blocks
            table_structures = []
            
            for table_block in table_blocks:
                # Create a basic table structure
                # In real implementation, this would use the table recognizer with actual images
                
                basic_table = TableStructure(
                    cells=[],  # Would be populated by table recognizer
                    num_rows=1,
                    num_cols=1,
                    bbox=table_block.bbox,
                    confidence=table_block.confidence,
                    page_number=table_block.page_number,
                    metadata={
                        "processing_method": "basic",
                        "text_content": table_block.text_content or "[TABLE]"
                    }
                )
                
                table_structures.append(basic_table)
            
            return table_structures
            
        except Exception as e:
            logger.error(f"Table processing failed: {e}")
            return []
    
    async def _combine_results(self, 
                             layout_blocks: List[LayoutBlock],
                             ocr_results: List[OCRResult],
                             table_structures: List[TableStructure],
                             document_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Combine all processing results into structured content.
        
        Args:
            layout_blocks: Layout blocks
            ocr_results: OCR results
            table_structures: Table structures
            document_id: Document ID
            
        Returns:
            List of (content, metadata) tuples suitable for chunking
        """
        try:
            combined_content = []
            
            # Sort blocks by reading order
            sorted_blocks = sorted(layout_blocks, key=lambda b: (b.page_number, b.bbox[1], b.bbox[0]))
            
            # Create content from blocks
            for block in sorted_blocks:
                content = ""
                metadata = {
                    "block_type": block.block_type.value,
                    "page_number": block.page_number,
                    "bbox": block.bbox,
                    "confidence": block.confidence,
                    "processing_method": "deepdoc"
                }
                
                if block.block_type == LayoutBlockType.TABLE:
                    # Find corresponding table structure
                    matching_table = None
                    for table in table_structures:
                        if (table.page_number == block.page_number and 
                            self._boxes_overlap(table.bbox, block.bbox)):
                            matching_table = table
                            break
                    
                    if matching_table:
                        # Use table content
                        content = matching_table.to_markdown()
                        metadata.update({
                            "table_rows": matching_table.num_rows,
                            "table_cols": matching_table.num_cols,
                            "table_cells": len(matching_table.cells)
                        })
                    else:
                        content = block.text_content or "[TABLE]"
                
                elif block.text_content:
                    content = block.text_content
                
                else:
                    # Use OCR content if available
                    for ocr_result in ocr_results:
                        if ocr_result.page_number == block.page_number:
                            # Find overlapping text regions
                            for text_region in ocr_result.text_regions:
                                if self._boxes_overlap(text_region.bbox, block.bbox):
                                    content = text_region.text
                                    break
                            if content:
                                break
                
                if content and content.strip():
                    combined_content.append((content.strip(), metadata))
            
            return combined_content
            
        except Exception as e:
            logger.error(f"Results combination failed: {e}")
            return []
    
    def _boxes_overlap(self, 
                      bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> bool:
        """
        Check if two bounding boxes overlap.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            True if boxes overlap
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Check for overlap
            return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
            
        except Exception:
            return False
    
    async def _fallback_processing(self, 
                                 pdf_bytes: bytes, 
                                 document_id: str) -> Dict[str, Any]:
        """
        Fallback processing using basic PDF parsing.
        
        Args:
            pdf_bytes: PDF content
            document_id: Document ID
            
        Returns:
            Basic processing results
        """
        try:
            logger.info("Using fallback PDF processing")
            
            # Use basic PDF parser
            from app.services.document_processing.parsers.pdf_parser import PDFParser
            
            pdf_parser = PDFParser()
            parsed_content = await pdf_parser(binary_content=pdf_bytes)
            
            # Convert to DeepDoc format
            combined_content = []
            
            for i, (content, metadata) in enumerate(parsed_content):
                if content and content.strip():
                    enhanced_metadata = {
                        "block_type": metadata.get("type", "text"),
                        "page_number": metadata.get("page_number", 1),
                        "processing_method": "fallback",
                        "confidence": 0.8
                    }
                    combined_content.append((content.strip(), enhanced_metadata))
            
            return {
                "success": True,
                "document_id": document_id,
                "layout_blocks": [],
                "ocr_results": [],
                "table_structures": [],
                "combined_content": combined_content,
                "fallback_used": True,
                "stats": {
                    "total_blocks": len(combined_content),
                    "processing_method": "fallback"
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            raise
    
    async def process_image_document(self, 
                                   image_bytes: bytes,
                                   document_id: str,
                                   image_format: str = "png") -> Dict[str, Any]:
        """
        Process image document using OCR.
        
        Args:
            image_bytes: Image file content
            document_id: Document identifier
            image_format: Image format (png, jpg, etc.)
            
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing image document {document_id}")
            
            # Convert image bytes to numpy array
            import cv2
            import numpy as np
            
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            ocr_result = await self.ocr_engine.extract_text_from_image(image, page_number=1)
            
            # Create combined content
            combined_content = []
            
            if ocr_result.full_text.strip():
                metadata = {
                    "block_type": "text",
                    "page_number": 1,
                    "processing_method": "ocr",
                    "confidence": 0.8,
                    "image_format": image_format
                }
                combined_content.append((ocr_result.full_text.strip(), metadata))
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "document_id": document_id,
                "processing_time": processing_time,
                "layout_blocks": [],
                "ocr_results": [ocr_result.to_dict()],
                "table_structures": [],
                "combined_content": combined_content,
                "stats": {
                    "total_blocks": len(combined_content),
                    "text_regions": len(ocr_result.text_regions),
                    "processing_method": "image_ocr"
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Image processing failed: {e}")
            
            return {
                "success": False,
                "document_id": document_id,
                "processing_time": processing_time,
                "error": str(e),
                "layout_blocks": [],
                "ocr_results": [],
                "table_structures": [],
                "combined_content": []
            }
    
    async def get_processor_stats(self) -> Dict[str, Any]:
        """Get DeepDoc processor statistics."""
        try:
            vision_stats = await self.vision_manager.health_check()
            layout_stats = await self.layout_recognizer.get_layout_stats()
            ocr_stats = await self.ocr_engine.get_ocr_stats()
            table_stats = await self.table_recognizer.get_table_stats()
            
            return {
                "initialized": self._initialized,
                "deepdoc_enabled": settings.DEEPDOC_ENABLED,
                "components": {
                    "vision_models": vision_stats,
                    "layout_recognizer": layout_stats,
                    "ocr_engine": ocr_stats,
                    "table_recognizer": table_stats
                },
                "supported_formats": ["pdf", "png", "jpg", "jpeg"],
                "fallback_available": True
            }
            
        except Exception as e:
            logger.error(f"Processor stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
deepdoc_processor = DeepDocProcessor()