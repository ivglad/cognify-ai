"""
PDF parser using unstructured library with trio support.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import trio
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Element

from .base_parser import BaseParser, DocumentStructure, ParsingError

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """
    PDF parser using unstructured library.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']
        self.parser_name = "PDFParser"
    
    async def parse(self, file_path: str, **kwargs) -> DocumentStructure:
        """
        Parse PDF document.
        
        Args:
            file_path: Path to PDF file
            **kwargs: Additional options:
                - strategy: "fast", "hi_res", "ocr_only"
                - extract_images: bool
                - infer_table_structure: bool
                
        Returns:
            DocumentStructure with parsed content
        """
        try:
            # Validate file
            if not await self.validate_file(file_path):
                raise ParsingError(f"Invalid PDF file: {file_path}", self.parser_name, file_path)
            
            # Parse options
            strategy = kwargs.get('strategy', 'fast')
            extract_images = kwargs.get('extract_images', False)
            infer_table_structure = kwargs.get('infer_table_structure', True)
            
            logger.info(f"Parsing PDF {file_path} with strategy: {strategy}")
            
            # Parse PDF using unstructured (run in thread to avoid blocking trio)
            elements = await trio.to_thread.run_sync(
                self._parse_pdf_sync,
                file_path,
                strategy,
                extract_images,
                infer_table_structure
            )
            
            # Process elements into DocumentStructure
            doc_structure = await self._process_elements(elements, file_path)
            
            logger.info(f"Successfully parsed PDF {file_path}: {len(doc_structure.text_blocks)} text blocks")
            
            return doc_structure
            
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise ParsingError(f"PDF parsing failed: {str(e)}", self.parser_name, file_path)
    
    def _parse_pdf_sync(self, file_path: str, strategy: str, extract_images: bool, infer_table_structure: bool) -> List[Element]:
        """
        Synchronous PDF parsing using unstructured.
        """
        return partition_pdf(
            filename=file_path,
            strategy=strategy,
            extract_images_in_pdf=extract_images,
            infer_table_structure=infer_table_structure,
            model_name="yolox" if strategy == "hi_res" else None
        )
    
    async def _process_elements(self, elements: List[Element], file_path: str) -> DocumentStructure:
        """
        Process unstructured elements into DocumentStructure.
        """
        doc_structure = DocumentStructure()
        doc_structure.metadata = self._extract_metadata(file_path)
        
        current_page = 1
        page_content = []
        
        for element in elements:
            element_type = str(type(element).__name__)
            element_text = str(element).strip()
            
            if not element_text:
                continue
            
            # Get element metadata
            metadata = {
                "element_type": element_type,
                "page_number": getattr(element, 'metadata', {}).get('page_number', current_page)
            }
            
            # Handle different element types
            if element_type == "Table":
                # Extract table
                table_data = {
                    "content": element_text,
                    "html_content": getattr(element, 'metadata', {}).get('text_as_html', ''),
                    "page_number": metadata["page_number"],
                    "element_type": "table"
                }
                doc_structure.tables.append(table_data)
                
                # Also add as text block
                doc_structure.text_blocks.append((element_text, metadata))
                
            elif element_type == "Image":
                # Extract image info
                image_data = {
                    "description": element_text,
                    "page_number": metadata["page_number"],
                    "element_type": "image"
                }
                doc_structure.images.append(image_data)
                
                # Add description as text block
                if element_text:
                    doc_structure.text_blocks.append((element_text, metadata))
                    
            else:
                # Regular text element
                doc_structure.text_blocks.append((element_text, metadata))
            
            # Track page content
            if metadata["page_number"] != current_page:
                if page_content:
                    doc_structure.pages.append({
                        "page_number": current_page,
                        "content": "\n".join(page_content)
                    })
                    page_content = []
                current_page = metadata["page_number"]
            
            page_content.append(element_text)
        
        # Add last page
        if page_content:
            doc_structure.pages.append({
                "page_number": current_page,
                "content": "\n".join(page_content)
            })
        
        # Update metadata
        doc_structure.metadata.update({
            "total_pages": len(doc_structure.pages),
            "total_tables": len(doc_structure.tables),
            "total_images": len(doc_structure.images),
            "total_text_blocks": len(doc_structure.text_blocks)
        })
        
        return doc_structure