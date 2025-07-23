"""
Naive merge chunking strategy - simple text merging until token limit.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

import trio

from app.services.chunking.strategies.base_chunker import BaseChunker, DocumentChunk, ChunkMetadata

logger = logging.getLogger(__name__)


class NaiveMergeChunker(BaseChunker):
    """
    Naive merge chunking strategy that simply merges text blocks until token limit.
    
    This is the simplest chunking strategy that concatenates text parts
    until reaching the maximum token limit, then starts a new chunk.
    """
    
    def __init__(self, 
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 min_chunk_size: int = 50,
                 merge_separator: str = "\n\n"):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.merge_separator = merge_separator
        self.strategy_name = "naive_merge"
    
    async def chunk_document(self, 
                           document_parts: List[Tuple[str, Optional[Dict[str, Any]]]],
                           document_id: str) -> List[DocumentChunk]:
        """
        Chunk document using naive merge strategy.
        
        Args:
            document_parts: List of (text, metadata) tuples from document parser
            document_id: Document identifier
            
        Returns:
            List of document chunks
        """
        if not document_parts:
            return []
        
        try:
            logger.debug(f"Starting naive merge chunking for document {document_id}")
            
            # Merge all text parts first
            merged_text_parts = await self._merge_text_parts(document_parts)
            
            # Split merged parts into chunks
            chunks = []
            chunk_index = 0
            
            for merged_part in merged_text_parts:
                text_content = merged_part["text"]
                source_metadata = merged_part["metadata"]
                
                # Split text if it's too long
                if self._count_tokens(text_content) > self.chunk_size:
                    text_chunks = self._split_text_by_tokens(
                        text_content, 
                        self.chunk_size, 
                        self.chunk_overlap
                    )
                else:
                    text_chunks = [text_content]
                
                # Create chunks
                for text_chunk in text_chunks:
                    if text_chunk.strip():
                        metadata = self._create_chunk_metadata(
                            document_id, 
                            chunk_index, 
                            source_metadata
                        )
                        
                        chunk = DocumentChunk(text_chunk, metadata)
                        
                        if self._validate_chunk(chunk):
                            chunks.append(chunk)
                            chunk_index += 1
            
            logger.info(f"Naive merge chunking completed: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            logger.error(f"Naive merge chunking failed: {e}")
            return []
    
    async def _merge_text_parts(self, 
                              document_parts: List[Tuple[str, Optional[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """
        Merge text parts until token limit is reached.
        
        Args:
            document_parts: List of (text, metadata) tuples
            
        Returns:
            List of merged text parts with combined metadata
        """
        if not document_parts:
            return []
        
        merged_parts = []
        current_text_parts = []
        current_token_count = 0
        current_metadata = None
        
        for text, metadata in document_parts:
            if not text or not text.strip():
                continue
            
            text = text.strip()
            text_tokens = self._count_tokens(text)
            
            # If adding this text would exceed the limit, finalize current merge
            if current_text_parts and (current_token_count + text_tokens > self.chunk_size):
                merged_text = self.merge_separator.join(current_text_parts)
                merged_parts.append({
                    "text": merged_text,
                    "metadata": current_metadata,
                    "token_count": current_token_count
                })
                
                # Start new merge
                current_text_parts = [text]
                current_token_count = text_tokens
                current_metadata = metadata
            else:
                # Add to current merge
                current_text_parts.append(text)
                current_token_count += text_tokens
                
                # Use metadata from first part or merge metadata
                if current_metadata is None:
                    current_metadata = metadata
                elif metadata:
                    current_metadata = self._merge_metadata(current_metadata, metadata)
        
        # Add final merge if exists
        if current_text_parts:
            merged_text = self.merge_separator.join(current_text_parts)
            merged_parts.append({
                "text": merged_text,
                "metadata": current_metadata,
                "token_count": current_token_count
            })
        
        return merged_parts
    
    def _merge_metadata(self, 
                       metadata1: Optional[Dict[str, Any]], 
                       metadata2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge metadata from two text parts.
        
        Args:
            metadata1: First metadata dict
            metadata2: Second metadata dict
            
        Returns:
            Merged metadata dict
        """
        if not metadata1:
            return metadata2 or {}
        if not metadata2:
            return metadata1
        
        merged = metadata1.copy()
        
        # Handle specific fields
        if "page_number" in metadata2:
            if "page_number" in merged:
                # Keep the first page number, but note the range
                if metadata2["page_number"] != merged["page_number"]:
                    merged["page_range"] = f"{merged['page_number']}-{metadata2['page_number']}"
            else:
                merged["page_number"] = metadata2["page_number"]
        
        # Merge types
        if "type" in metadata2:
            if "type" in merged and merged["type"] != metadata2["type"]:
                merged["type"] = "mixed"
            else:
                merged["type"] = metadata2["type"]
        
        # Add any additional fields from metadata2
        for key, value in metadata2.items():
            if key not in merged:
                merged[key] = value
        
        return merged
    
    async def chunk_text_directly(self, 
                                text: str, 
                                document_id: str,
                                source_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Chunk text directly without document parts.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            source_metadata: Optional metadata for the text
            
        Returns:
            List of document chunks
        """
        if not text or not text.strip():
            return []
        
        try:
            # Split text into chunks
            text_chunks = self._split_text_by_tokens(
                text.strip(), 
                self.chunk_size, 
                self.chunk_overlap
            )
            
            chunks = []
            for i, text_chunk in enumerate(text_chunks):
                if text_chunk.strip():
                    metadata = self._create_chunk_metadata(
                        document_id, 
                        i, 
                        source_metadata
                    )
                    
                    chunk = DocumentChunk(text_chunk, metadata)
                    
                    if self._validate_chunk(chunk):
                        chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Direct text chunking failed: {e}")
            return []
    
    async def get_chunker_stats(self) -> Dict[str, Any]:
        """Get naive merge chunker statistics."""
        base_stats = await super().get_chunker_stats()
        base_stats.update({
            "merge_separator": repr(self.merge_separator),
            "description": "Simple text merging until token limit"
        })
        return base_stats