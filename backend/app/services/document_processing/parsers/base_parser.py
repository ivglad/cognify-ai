"""
Base parser interface for document processing with trio support.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import trio

logger = logging.getLogger(__name__)


class DocumentStructure:
    """
    Document structure representation.
    """
    def __init__(self):
        self.pages: List[Dict[str, Any]] = []
        self.tables: List[Dict[str, Any]] = []
        self.images: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.text_blocks: List[Tuple[str, Optional[str]]] = []  # (content, metadata)


class BaseParser(ABC):
    """
    Abstract base class for document parsers.
    """
    
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.parser_name: str = self.__class__.__name__
    
    @abstractmethod
    async def parse(self, file_path: str, **kwargs) -> DocumentStructure:
        """
        Parse document and return structured content.
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional parsing options
            
        Returns:
            DocumentStructure with parsed content
        """
        pass
    
    async def parse_from_bytes(self, file_bytes: bytes, filename: str, **kwargs) -> DocumentStructure:
        """
        Parse document from bytes.
        
        Args:
            file_bytes: Document content as bytes
            filename: Original filename for format detection
            **kwargs: Additional parsing options
            
        Returns:
            DocumentStructure with parsed content
        """
        # Default implementation: save to temp file and parse
        import tempfile
        import os
        
        temp_dir = Path(tempfile.gettempdir())
        temp_file = temp_dir / f"temp_{trio.current_time()}_{filename}"
        
        try:
            # Write bytes to temp file
            async with trio.open_file(temp_file, 'wb') as f:
                await f.write(file_bytes)
            
            # Parse from temp file
            result = await self.parse(str(temp_file), **kwargs)
            
            return result
            
        finally:
            # Cleanup temp file
            try:
                if temp_file.exists():
                    await trio.to_thread.run_sync(temp_file.unlink)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    
    def supports_file(self, filename: str) -> bool:
        """
        Check if parser supports the given file.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if parser supports the file format
        """
        if not filename:
            return False
        
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.supported_extensions
    
    async def validate_file(self, file_path: str) -> bool:
        """
        Validate if file can be parsed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid for parsing
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                return False
            
            # Check if file is readable
            if not path.is_file():
                return False
            
            # Check file extension
            if not self.supports_file(path.name):
                return False
            
            # Check file size (basic validation)
            file_size = path.stat().st_size
            if file_size == 0:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {e}")
            return False
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract basic file metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata
        """
        try:
            path = Path(file_path)
            stat = path.stat()
            
            return {
                "filename": path.name,
                "file_size": stat.st_size,
                "file_extension": path.suffix.lower(),
                "parser_used": self.parser_name,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            return {
                "filename": Path(file_path).name,
                "parser_used": self.parser_name
            }


class ParsingError(Exception):
    """
    Exception raised when document parsing fails.
    """
    def __init__(self, message: str, parser_name: str, file_path: str = None):
        self.message = message
        self.parser_name = parser_name
        self.file_path = file_path
        super().__init__(f"[{parser_name}] {message}")


class UnsupportedFormatError(ParsingError):
    """
    Exception raised when document format is not supported.
    """
    pass