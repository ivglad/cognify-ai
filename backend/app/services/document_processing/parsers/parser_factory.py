"""
Parser factory for selecting appropriate document parser.
"""
import logging
import mimetypes
from typing import Optional, Dict, Any, Type

from app.services.document_processing.parsers.base_parser import BaseParser
from app.services.document_processing.parsers.text_parser import TextParser
from app.services.document_processing.parsers.pdf_parser import PDFParser
from app.services.document_processing.parsers.docx_parser import DocxParser
from app.services.document_processing.parsers.excel_parser import ExcelParser
from app.services.document_processing.parsers.html_parser import HTMLParser
from app.services.document_processing.parsers.xml_parser import XMLParser
from app.services.document_processing.parsers.json_parser import JSONParser

logger = logging.getLogger(__name__)


class ParserFactory:
    """
    Factory for creating document parsers based on file type.
    """
    
    # Map of file extensions to parser classes
    EXTENSION_PARSERS = {
        # Text formats
        'txt': TextParser,
        'md': TextParser,
        'csv': ExcelParser,
        
        # Structured formats
        'json': JSONParser,
        'html': HTMLParser,
        'htm': HTMLParser,
        'xml': XMLParser,
        'xhtml': HTMLParser,
        
        # Document formats
        'pdf': PDFParser,
        'docx': DocxParser,
        'doc': DocxParser,  # Note: Limited support for .doc
        
        # Spreadsheet formats
        'xlsx': ExcelParser,
        'xls': ExcelParser,
        'xlsm': ExcelParser,
        'ods': ExcelParser,
    }
    
    # Map of MIME types to parser classes
    MIME_PARSERS = {
        # Text formats
        'text/plain': TextParser,
        'text/markdown': TextParser,
        'text/csv': ExcelParser,
        
        # Structured formats
        'application/json': JSONParser,
        'text/html': HTMLParser,
        'application/xhtml+xml': HTMLParser,
        'text/xml': XMLParser,
        'application/xml': XMLParser,
        
        # Document formats
        'application/pdf': PDFParser,
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocxParser,
        'application/msword': DocxParser,
        
        # Spreadsheet formats
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ExcelParser,
        'application/vnd.ms-excel': ExcelParser,
        'application/vnd.ms-excel.sheet.macroEnabled.12': ExcelParser,
        'application/vnd.oasis.opendocument.spreadsheet': ExcelParser,
    }
    
    @classmethod
    def get_parser(cls, file_path: Optional[str] = None, 
                   content_type: Optional[str] = None) -> BaseParser:
        """
        Get appropriate parser for the given file.
        
        Args:
            file_path: Path to the file (optional)
            content_type: MIME type of the file (optional)
            
        Returns:
            Parser instance for the file type
            
        Raises:
            ValueError: If no suitable parser is found
        """
        parser_class = None
        
        # Try to determine parser by MIME type first
        if content_type:
            parser_class = cls.MIME_PARSERS.get(content_type)
            if parser_class:
                logger.debug(f"Selected parser {parser_class.__name__} based on MIME type: {content_type}")
                return parser_class()
        
        # Try to determine parser by file extension
        if file_path:
            extension = cls._get_file_extension(file_path)
            parser_class = cls.EXTENSION_PARSERS.get(extension)
            if parser_class:
                logger.debug(f"Selected parser {parser_class.__name__} based on extension: {extension}")
                return parser_class()
        
        # Try to guess MIME type from file path
        if file_path and not content_type:
            guessed_type, _ = mimetypes.guess_type(file_path)
            if guessed_type:
                parser_class = cls.MIME_PARSERS.get(guessed_type)
                if parser_class:
                    logger.debug(f"Selected parser {parser_class.__name__} based on guessed MIME type: {guessed_type}")
                    return parser_class()
        
        # Default to text parser if no specific parser found
        logger.warning(f"No specific parser found for file_path='{file_path}', content_type='{content_type}'. Using TextParser as fallback.")
        return TextParser()
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported file extensions
        """
        return list(cls.EXTENSION_PARSERS.keys())
    
    @classmethod
    def get_supported_mime_types(cls) -> list:
        """
        Get list of supported MIME types.
        
        Returns:
            List of supported MIME types
        """
        return list(cls.MIME_PARSERS.keys())
    
    @classmethod
    def is_supported(cls, file_path: Optional[str] = None, 
                     content_type: Optional[str] = None) -> bool:
        """
        Check if file type is supported.
        
        Args:
            file_path: Path to the file (optional)
            content_type: MIME type of the file (optional)
            
        Returns:
            True if file type is supported, False otherwise
        """
        # Check by MIME type
        if content_type and content_type in cls.MIME_PARSERS:
            return True
        
        # Check by file extension
        if file_path:
            extension = cls._get_file_extension(file_path)
            if extension in cls.EXTENSION_PARSERS:
                return True
        
        # Check by guessed MIME type
        if file_path and not content_type:
            guessed_type, _ = mimetypes.guess_type(file_path)
            if guessed_type and guessed_type in cls.MIME_PARSERS:
                return True
        
        return False
    
    @staticmethod
    def _get_file_extension(file_path: str) -> str:
        """
        Get file extension from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File extension without dot (lowercase)
        """
        return file_path.split('.')[-1].lower() if '.' in file_path else ''