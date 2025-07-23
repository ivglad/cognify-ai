"""
Parser manager for coordinating document parsing operations.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any

import trio

from app.services.document_processing.parsers.parser_factory import ParserFactory
from app.services.document_processing.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class ParserManager:
    """
    Manager for coordinating document parsing operations.
    """
    
    def __init__(self):
        self.factory = ParserFactory()
        
    async def parse_document(self, 
                           file_path: Optional[str] = None,
                           binary_content: Optional[bytes] = None,
                           content_type: Optional[str] = None,
                           **parser_kwargs) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Parse document using appropriate parser.
        
        Args:
            file_path: Path to the document file (optional)
            binary_content: Binary content of the document (optional)
            content_type: MIME type of the document (optional)
            **parser_kwargs: Additional parser-specific parameters
            
        Returns:
            List of tuples containing (text_content, metadata_dict)
            
        Raises:
            ValueError: If neither file_path nor binary_content is provided
        """
        if not file_path and not binary_content:
            raise ValueError("Either file_path or binary_content must be provided")
        
        try:
            # Get appropriate parser
            parser = self.factory.get_parser(file_path=file_path, content_type=content_type)
            
            logger.info(f"Using parser {parser.__class__.__name__} for file_path='{file_path}', content_type='{content_type}'")
            
            # Parse document
            result = await parser(
                file_path=file_path,
                binary_content=binary_content,
                **parser_kwargs
            )
            
            logger.info(f"Successfully parsed document. Extracted {len(result)} text parts.")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            # Return error information instead of raising
            return [(f"Error parsing document: {str(e)}", {
                "type": "error",
                "error": str(e),
                "parser_used": parser.__class__.__name__ if 'parser' in locals() else "unknown"
            })]
    
    async def batch_parse_documents(self, 
                                  documents: List[Dict[str, Any]],
                                  max_concurrent: int = 3) -> List[List[Tuple[str, Optional[Dict[str, Any]]]]]:
        """
        Parse multiple documents concurrently.
        
        Args:
            documents: List of document dictionaries with keys:
                      - file_path (optional)
                      - binary_content (optional)
                      - content_type (optional)
                      - parser_kwargs (optional)
            max_concurrent: Maximum number of concurrent parsing operations
            
        Returns:
            List of parsing results for each document
        """
        async def parse_single_document(doc_info: Dict[str, Any]) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
            """Parse a single document from the batch."""
            return await self.parse_document(
                file_path=doc_info.get('file_path'),
                binary_content=doc_info.get('binary_content'),
                content_type=doc_info.get('content_type'),
                **doc_info.get('parser_kwargs', {})
            )
        
        # Use trio semaphore to limit concurrent operations
        semaphore = trio.Semaphore(max_concurrent)
        
        async def limited_parse(doc_info: Dict[str, Any]) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
            """Parse with semaphore limiting."""
            async with semaphore:
                return await parse_single_document(doc_info)
        
        # Start all parsing tasks
        async with trio.open_nursery() as nursery:
            results = []
            
            for doc_info in documents:
                nursery.start_soon(limited_parse, doc_info)
        
        return results
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """
        Get information about supported file formats.
        
        Returns:
            Dictionary with supported extensions and MIME types
        """
        return {
            "extensions": self.factory.get_supported_extensions(),
            "mime_types": self.factory.get_supported_mime_types()
        }
    
    def is_supported_format(self, 
                          file_path: Optional[str] = None,
                          content_type: Optional[str] = None) -> bool:
        """
        Check if file format is supported.
        
        Args:
            file_path: Path to the file (optional)
            content_type: MIME type of the file (optional)
            
        Returns:
            True if format is supported, False otherwise
        """
        return self.factory.is_supported(file_path=file_path, content_type=content_type)
    
    async def validate_document(self, 
                              file_path: Optional[str] = None,
                              binary_content: Optional[bytes] = None,
                              content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate document before parsing.
        
        Args:
            file_path: Path to the document file (optional)
            binary_content: Binary content of the document (optional)
            content_type: MIME type of the document (optional)
            
        Returns:
            Validation result dictionary with keys:
            - is_valid: bool
            - format_supported: bool
            - estimated_size: int (if binary_content provided)
            - file_extension: str (if file_path provided)
            - detected_mime_type: str (if detectable)
            - warnings: List[str]
            - errors: List[str]
        """
        result = {
            "is_valid": True,
            "format_supported": False,
            "estimated_size": None,
            "file_extension": None,
            "detected_mime_type": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check if format is supported
            result["format_supported"] = self.is_supported_format(
                file_path=file_path, 
                content_type=content_type
            )
            
            if not result["format_supported"]:
                result["warnings"].append("File format may not be fully supported")
            
            # Get file extension if file_path provided
            if file_path:
                result["file_extension"] = self.factory._get_file_extension(file_path)
            
            # Get estimated size if binary_content provided
            if binary_content:
                result["estimated_size"] = len(binary_content)
                
                # Check for very large files
                if result["estimated_size"] > 100 * 1024 * 1024:  # 100MB
                    result["warnings"].append("Large file detected - parsing may take significant time")
                
                # Try to detect MIME type from binary content
                try:
                    import magic
                    result["detected_mime_type"] = magic.from_buffer(binary_content, mime=True)
                except ImportError:
                    result["warnings"].append("python-magic not available for MIME type detection")
                except Exception as e:
                    result["warnings"].append(f"Could not detect MIME type: {e}")
            
            # Basic validation checks
            if not file_path and not binary_content:
                result["errors"].append("Either file_path or binary_content must be provided")
                result["is_valid"] = False
            
            if binary_content and len(binary_content) == 0:
                result["errors"].append("Binary content is empty")
                result["is_valid"] = False
            
        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")
            result["is_valid"] = False
        
        return result