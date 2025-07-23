"""
Base chunker interface for document chunking strategies.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

import trio
import tiktoken

from app.core.config import settings

logger = logging.getLogger(__name__)


class ChunkMetadata:
    """Metadata for a document chunk."""
    
    def __init__(self, 
                 chunk_id: str = None,
                 document_id: str = None,
                 chunk_type: str = "text",
                 chunk_strategy: str = "base",
                 page_number: Optional[int] = None,
                 position: Optional[int] = None,
                 token_count: int = 0,
                 char_count: int = 0,
                 confidence: float = 1.0,
                 **kwargs):
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.document_id = document_id
        self.chunk_type = chunk_type
        self.chunk_strategy = chunk_strategy
        self.page_number = page_number
        self.position = position
        self.token_count = token_count
        self.char_count = char_count
        self.confidence = confidence
        self.created_at = datetime.utcnow()
        self.extra_metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_type": self.chunk_type,
            "chunk_strategy": self.chunk_strategy,
            "page_number": self.page_number,
            "position": self.position,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            **self.extra_metadata
        }


class DocumentChunk:
    """A document chunk with content and metadata."""
    
    def __init__(self, content: str, metadata: ChunkMetadata):
        self.content = content
        self.metadata = metadata
        
        # Update token and character counts
        if not metadata.token_count:
            metadata.token_count = self._count_tokens(content)
        if not metadata.char_count:
            metadata.char_count = len(content)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using word count: {e}")
            return len(text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }


class BaseChunker(ABC):
    """
    Abstract base class for document chunking strategies.
    """
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 min_chunk_size: int = 50):
        self.chunk_size = chunk_size or settings.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.DEFAULT_CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size
        self.strategy_name = self.__class__.__name__.lower().replace('chunker', '')
        
    @abstractmethod
    async def chunk_document(self, 
                           document_parts: List[Tuple[str, Optional[Dict[str, Any]]]],
                           document_id: str) -> List[DocumentChunk]:
        """
        Chunk document parts into smaller pieces.
        
        Args:
            document_parts: List of (text, metadata) tuples from document parser
            document_id: Document identifier
            
        Returns:
            List of document chunks
        """
        pass
    
    def _split_text_by_tokens(self, 
                             text: str, 
                             max_tokens: int, 
                             overlap_tokens: int = 0) -> List[str]:
        """
        Split text by token count with optional overlap.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens
            
        Returns:
            List of text chunks
        """
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            
            if len(tokens) <= max_tokens:
                return [text]
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                
                # Move start position with overlap
                if end >= len(tokens):
                    break
                start = end - overlap_tokens
                
                # Ensure we make progress
                if start <= 0:
                    start = end
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Token-based splitting failed, using character-based: {e}")
            return self._split_text_by_chars(text, max_tokens * 4, overlap_tokens * 4)
    
    def _split_text_by_chars(self, 
                            text: str, 
                            max_chars: int, 
                            overlap_chars: int = 0) -> List[str]:
        """
        Split text by character count with optional overlap.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            overlap_chars: Number of overlapping characters
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Try to break at sentence or paragraph boundary
            chunk_text = text[start:end]
            
            # If not at the end, try to find a good break point
            if end < len(text):
                # Look for sentence endings
                for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 200), -1):
                    if chunk_text[i] in '.!?':
                        chunk_text = chunk_text[:i + 1]
                        end = start + i + 1
                        break
                else:
                    # Look for paragraph breaks
                    for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 200), -1):
                        if chunk_text[i] == '\n':
                            chunk_text = chunk_text[:i]
                            end = start + i
                            break
                    else:
                        # Look for word boundaries
                        for i in range(len(chunk_text) - 1, max(0, len(chunk_text) - 100), -1):
                            if chunk_text[i] == ' ':
                                chunk_text = chunk_text[:i]
                                end = start + i
                                break
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            # Move start position with overlap
            if end >= len(text):
                break
            start = max(start + 1, end - overlap_chars)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return len(text.split())
    
    def _create_chunk_metadata(self, 
                              document_id: str,
                              chunk_index: int,
                              source_metadata: Optional[Dict[str, Any]] = None) -> ChunkMetadata:
        """Create metadata for a chunk."""
        metadata = ChunkMetadata(
            document_id=document_id,
            chunk_strategy=self.strategy_name,
            position=chunk_index
        )
        
        # Copy relevant metadata from source
        if source_metadata:
            metadata.page_number = source_metadata.get('page_number')
            metadata.chunk_type = source_metadata.get('type', 'text')
            
            # Copy any additional metadata
            for key, value in source_metadata.items():
                if key not in ['type', 'page_number']:
                    metadata.extra_metadata[key] = value
        
        return metadata
    
    def _validate_chunk(self, chunk: DocumentChunk) -> bool:
        """Validate if chunk meets minimum requirements."""
        if not chunk.content or not chunk.content.strip():
            return False
        
        if len(chunk.content.strip()) < self.min_chunk_size:
            return False
        
        return True
    
    async def get_chunker_stats(self) -> Dict[str, Any]:
        """Get chunker statistics and configuration."""
        return {
            "strategy_name": self.strategy_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size
        }