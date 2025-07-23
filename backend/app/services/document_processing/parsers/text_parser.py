"""
Text parser for plain text files with trio support.
"""
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import chardet

import trio

from .base_parser import BaseParser, DocumentStructure, ParsingError

logger = logging.getLogger(__name__)


class TextParser(BaseParser):
    """
    Parser for plain text files with encoding detection.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.txt', '.md', '.rst', '.log']
        self.parser_name = "TextParser"
    
    async def parse(self, file_path: str, **kwargs) -> DocumentStructure:
        """
        Parse text document.
        
        Args:
            file_path: Path to text file
            **kwargs: Additional options:
                - encoding: Specific encoding to use
                - chunk_by_lines: Number of lines per chunk
                - preserve_structure: bool
                
        Returns:
            DocumentStructure with parsed content
        """
        try:
            # Validate file
            if not await self.validate_file(file_path):
                raise ParsingError(f"Invalid text file: {file_path}", self.parser_name, file_path)
            
            # Parse options
            encoding = kwargs.get('encoding', None)
            chunk_by_lines = kwargs.get('chunk_by_lines', 0)
            preserve_structure = kwargs.get('preserve_structure', True)
            
            logger.info(f"Parsing text file {file_path}")
            
            # Parse text file (run in thread)
            doc_structure = await trio.to_thread.run_sync(
                self._parse_text_sync,
                file_path,
                encoding,
                chunk_by_lines,
                preserve_structure
            )
            
            logger.info(f"Successfully parsed text file {file_path}: {len(doc_structure.text_blocks)} text blocks")
            
            return doc_structure
            
        except Exception as e:
            logger.error(f"Failed to parse text file {file_path}: {e}")
            raise ParsingError(f"Text parsing failed: {str(e)}", self.parser_name, file_path)
    
    def _parse_text_sync(self, file_path: str, encoding: Optional[str], 
                        chunk_by_lines: int, preserve_structure: bool) -> DocumentStructure:
        """
        Synchronous text parsing.
        """
        doc_structure = DocumentStructure()
        doc_structure.metadata = self._extract_metadata(file_path)
        
        # Detect encoding if not provided
        if not encoding:
            encoding = self._detect_encoding(file_path)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # Process content
            if chunk_by_lines > 0:
                # Split by lines and chunk
                lines = content.split('\n')
                for i in range(0, len(lines), chunk_by_lines):
                    chunk_lines = lines[i:i + chunk_by_lines]
                    chunk_text = '\n'.join(chunk_lines).strip()
                    
                    if chunk_text:
                        metadata = {
                            "element_type": "text_chunk",
                            "line_start": i + 1,
                            "line_end": min(i + chunk_by_lines, len(lines)),
                            "chunk_index": i // chunk_by_lines
                        }
                        doc_structure.text_blocks.append((chunk_text, metadata))
            
            elif preserve_structure:
                # Split by paragraphs (double newlines)
                paragraphs = content.split('\n\n')
                
                for i, paragraph in enumerate(paragraphs):
                    paragraph = paragraph.strip()
                    if paragraph:
                        metadata = {
                            "element_type": "paragraph",
                            "paragraph_index": i
                        }
                        doc_structure.text_blocks.append((paragraph, metadata))
            
            else:
                # Treat entire content as single block
                if content.strip():
                    metadata = {
                        "element_type": "full_text",
                        "encoding": encoding
                    }
                    doc_structure.text_blocks.append((content.strip(), metadata))
            
            # Add as single page
            doc_structure.pages.append({
                "page_number": 1,
                "content": content,
                "encoding": encoding,
                "line_count": len(content.split('\n'))
            })
            
        except UnicodeDecodeError as e:
            raise ParsingError(f"Encoding error with {encoding}: {str(e)}", self.parser_name, file_path)
        except Exception as e:
            raise ParsingError(f"Failed to read file: {str(e)}", self.parser_name, file_path)
        
        # Update metadata
        doc_structure.metadata.update({
            "encoding": encoding,
            "total_lines": len(content.split('\n')),
            "total_characters": len(content),
            "total_text_blocks": len(doc_structure.text_blocks)
        })
        
        return doc_structure
    
    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet.
        """
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
            
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            logger.debug(f"Detected encoding {encoding} with confidence {confidence} for {file_path}")
            
            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                encoding = 'utf-8'
            
            return encoding
            
        except Exception as e:
            logger.warning(f"Failed to detect encoding for {file_path}: {e}")
            return 'utf-8'