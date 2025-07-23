"""
Chunking manager for coordinating different chunking strategies.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import re

import trio

from app.services.chunking.strategies.base_chunker import BaseChunker, DocumentChunk
from app.services.chunking.strategies.naive_merge import NaiveMergeChunker
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChunkingManager:
    """
    Manager for coordinating different chunking strategies.
    """
    
    def __init__(self):
        self.strategies: Dict[str, BaseChunker] = {}
        self.default_strategy = "naive_merge"
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize available chunking strategies."""
        # Import strategies
        from app.services.chunking.strategies.hierarchical_merge import HierarchicalMergeChunker
        from app.services.chunking.strategies.qa_chunking import QAChunkingStrategy
        from app.services.chunking.strategies.table_chunking import TableChunkingStrategy
        
        # Naive merge strategy
        self.strategies["naive_merge"] = NaiveMergeChunker(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
        
        # Hierarchical chunking strategy
        self.strategies["hierarchical"] = HierarchicalMergeChunker(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
        
        # QA chunking strategy
        self.strategies["qa_chunking"] = QAChunkingStrategy(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
        
        # Table chunking strategy
        self.strategies["table_chunking"] = TableChunkingStrategy(
            chunk_size=settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=settings.DEFAULT_CHUNK_OVERLAP
        )
        
        logger.info(f"Initialized {len(self.strategies)} chunking strategies")
    
    async def chunk_document(self, 
                           document_parts: List[Tuple[str, Optional[Dict[str, Any]]]],
                           document_id: str,
                           strategy: str = None,
                           strategy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Chunk document using specified strategy.
        
        Args:
            document_parts: List of (text, metadata) tuples from document parser
            document_id: Document identifier
            strategy: Chunking strategy name (optional, uses default or auto-selects)
            strategy_config: Strategy-specific configuration
            
        Returns:
            Chunking results with chunks and metadata
        """
        start_time = time.time()
        
        # Auto-select strategy if not specified or if "auto" is specified
        if strategy is None or strategy == "auto":
            strategy = await self._auto_select_strategy(document_parts)
            logger.info(f"Auto-selected strategy '{strategy}' for document {document_id}")
        else:
            strategy = strategy or self.default_strategy
        
        try:
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy '{strategy}', using default '{self.default_strategy}'")
                strategy = self.default_strategy
            
            chunker = self.strategies[strategy]
            
            # Apply strategy-specific configuration if provided
            if strategy_config:
                chunker = self._configure_chunker(chunker, strategy_config)
            
            logger.info(f"Starting chunking with strategy '{strategy}' for document {document_id}")
            
            # Perform chunking
            chunks = await chunker.chunk_document(document_parts, document_id)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            stats = await self._calculate_chunking_stats(chunks, document_parts, processing_time)
            
            logger.info(f"Chunking completed: {len(chunks)} chunks in {processing_time:.3f}s")
            
            return {
                "chunks": chunks,
                "strategy": strategy,
                "document_id": document_id,
                "stats": stats,
                "success": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Chunking failed for document {document_id}: {e}")
            
            return {
                "chunks": [],
                "strategy": strategy,
                "document_id": document_id,
                "stats": {
                    "chunk_count": 0,
                    "processing_time": processing_time,
                    "error": str(e)
                },
                "success": False,
                "error": str(e)
            }
    
    def _configure_chunker(self, 
                          chunker: BaseChunker, 
                          config: Dict[str, Any]) -> BaseChunker:
        """
        Configure chunker with provided settings.
        
        Args:
            chunker: Base chunker instance
            config: Configuration parameters
            
        Returns:
            Configured chunker (may be a new instance)
        """
        # Create new instance with custom configuration
        chunker_class = chunker.__class__
        
        # Extract relevant configuration
        chunk_size = config.get("chunk_size", chunker.chunk_size)
        chunk_overlap = config.get("chunk_overlap", chunker.chunk_overlap)
        min_chunk_size = config.get("min_chunk_size", chunker.min_chunk_size)
        
        # Create new configured instance
        configured_chunker = chunker_class(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        
        # Apply any strategy-specific configuration
        for key, value in config.items():
            if hasattr(configured_chunker, key) and key not in ["chunk_size", "chunk_overlap", "min_chunk_size"]:
                setattr(configured_chunker, key, value)
        
        return configured_chunker
    
    async def _calculate_chunking_stats(self, 
                                      chunks: List[DocumentChunk],
                                      document_parts: List[Tuple[str, Optional[Dict[str, Any]]]],
                                      processing_time: float) -> Dict[str, Any]:
        """Calculate chunking statistics."""
        if not chunks:
            return {
                "chunk_count": 0,
                "total_tokens": 0,
                "total_chars": 0,
                "avg_tokens_per_chunk": 0,
                "avg_chars_per_chunk": 0,
                "processing_time": processing_time,
                "original_parts_count": len(document_parts)
            }
        
        total_tokens = sum(chunk.metadata.token_count for chunk in chunks)
        total_chars = sum(chunk.metadata.char_count for chunk in chunks)
        
        return {
            "chunk_count": len(chunks),
            "total_tokens": total_tokens,
            "total_chars": total_chars,
            "avg_tokens_per_chunk": total_tokens / len(chunks),
            "avg_chars_per_chunk": total_chars / len(chunks),
            "min_tokens": min(chunk.metadata.token_count for chunk in chunks),
            "max_tokens": max(chunk.metadata.token_count for chunk in chunks),
            "processing_time": processing_time,
            "original_parts_count": len(document_parts),
            "compression_ratio": len(document_parts) / len(chunks) if chunks else 0
        }
    
    async def chunk_text_directly(self, 
                                text: str, 
                                document_id: str,
                                strategy: str = None,
                                source_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Chunk text directly without document parts.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            strategy: Chunking strategy name
            source_metadata: Optional metadata for the text
            
        Returns:
            Chunking results
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        try:
            if strategy not in self.strategies:
                logger.warning(f"Unknown strategy '{strategy}', using default '{self.default_strategy}'")
                strategy = self.default_strategy
            
            chunker = self.strategies[strategy]
            
            # Use direct chunking if available, otherwise convert to document parts
            if hasattr(chunker, 'chunk_text_directly'):
                chunks = await chunker.chunk_text_directly(text, document_id, source_metadata)
            else:
                # Convert text to document parts format
                document_parts = [(text, source_metadata)]
                chunks = await chunker.chunk_document(document_parts, document_id)
            
            processing_time = time.time() - start_time
            stats = await self._calculate_chunking_stats(chunks, [(text, source_metadata)], processing_time)
            
            return {
                "chunks": chunks,
                "strategy": strategy,
                "document_id": document_id,
                "stats": stats,
                "success": True
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Direct text chunking failed: {e}")
            
            return {
                "chunks": [],
                "strategy": strategy,
                "document_id": document_id,
                "stats": {
                    "chunk_count": 0,
                    "processing_time": processing_time,
                    "error": str(e)
                },
                "success": False,
                "error": str(e)
            }
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available chunking strategies."""
        return list(self.strategies.keys())
    
    async def get_strategy_info(self, strategy: str = None) -> Dict[str, Any]:
        """
        Get information about a specific strategy or all strategies.
        
        Args:
            strategy: Strategy name (optional, returns all if not specified)
            
        Returns:
            Strategy information
        """
        if strategy:
            if strategy in self.strategies:
                chunker = self.strategies[strategy]
                return await chunker.get_chunker_stats()
            else:
                return {"error": f"Unknown strategy: {strategy}"}
        else:
            # Return info for all strategies
            strategies_info = {}
            for name, chunker in self.strategies.items():
                strategies_info[name] = await chunker.get_chunker_stats()
            
            return {
                "available_strategies": strategies_info,
                "default_strategy": self.default_strategy,
                "total_strategies": len(self.strategies)
            }
    
    async def validate_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Validate chunks for quality and consistency.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            Validation results
        """
        if not chunks:
            return {
                "valid": True,
                "issues": [],
                "stats": {"total_chunks": 0}
            }
        
        issues = []
        
        # Check for empty chunks
        empty_chunks = [i for i, chunk in enumerate(chunks) if not chunk.content.strip()]
        if empty_chunks:
            issues.append(f"Found {len(empty_chunks)} empty chunks at positions: {empty_chunks[:10]}")
        
        # Check for very small chunks
        min_size = 20  # Minimum reasonable chunk size
        small_chunks = [i for i, chunk in enumerate(chunks) if len(chunk.content.strip()) < min_size]
        if small_chunks:
            issues.append(f"Found {len(small_chunks)} very small chunks (< {min_size} chars)")
        
        # Check for very large chunks
        max_reasonable_tokens = settings.DEFAULT_CHUNK_SIZE * 2
        large_chunks = [i for i, chunk in enumerate(chunks) if chunk.metadata.token_count > max_reasonable_tokens]
        if large_chunks:
            issues.append(f"Found {len(large_chunks)} very large chunks (> {max_reasonable_tokens} tokens)")
        
        # Check for missing metadata
        missing_metadata = [i for i, chunk in enumerate(chunks) if not chunk.metadata.document_id]
        if missing_metadata:
            issues.append(f"Found {len(missing_metadata)} chunks with missing document_id")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "total_chunks": len(chunks),
                "empty_chunks": len(empty_chunks),
                "small_chunks": len(small_chunks),
                "large_chunks": len(large_chunks),
                "missing_metadata": len(missing_metadata)
            }
        }
    
    async def _auto_select_strategy(self, document_parts: List[Tuple[str, Optional[Dict[str, Any]]]]) -> str:
        """
        Automatically select the best chunking strategy based on document analysis.
        
        Args:
            document_parts: List of (text, metadata) tuples from document parser
            
        Returns:
            Selected strategy name
        """
        try:
            # Combine all text for analysis
            full_text = "\n\n".join([part[0] for part in document_parts if part[0]])
            
            if not full_text.strip():
                return self.default_strategy
            
            # Analyze document characteristics
            analysis = await self._analyze_document_content(full_text)
            
            # Select strategy based on analysis
            strategy = self._select_strategy_from_analysis(analysis)
            
            logger.debug(f"Document analysis: {analysis}")
            logger.debug(f"Selected strategy: {strategy}")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Auto strategy selection failed: {e}")
            return self.default_strategy
    
    async def _analyze_document_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze document content to determine characteristics.
        
        Args:
            text: Full document text
            
        Returns:
            Analysis results with scores for different content types
        """
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'structure_score': 0.0,
            'qa_score': 0.0,
            'table_score': 0.0
        }
        
        try:
            # Structure analysis patterns
            structure_patterns = [
                # Markdown headings
                re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
                # HTML headings
                re.compile(r'<h[1-6][^>]*>.+?</h[1-6]>', re.IGNORECASE),
                # Numbered sections
                re.compile(r'^\d+(?:\.\d+)*\.?\s+.+$', re.MULTILINE),
                # All caps headings
                re.compile(r'^[A-ZА-Я][A-ZА-Я\s]{5,}$', re.MULTILINE)
            ]
            
            # QA analysis patterns
            qa_patterns = [
                # Q: A: format
                re.compile(r'(?:Q|Question|Вопрос)[:.]?\s*.+?(?:A|Answer|Ответ)[:.]?\s*.+', re.IGNORECASE),
                # FAQ format
                re.compile(r'\d+\.?\s*.+\?\s*\n.+', re.MULTILINE),
                # Interview format
                re.compile(r'(?:Interviewer|Интервьюер)[:.]?\s*.+', re.IGNORECASE)
            ]
            
            # Table analysis patterns
            table_patterns = [
                # Markdown tables
                re.compile(r'\|.+\|\s*\n\s*\|[-:\s|]+\|\s*\n(?:\|.+\|\s*\n?)+'),
                # HTML tables
                re.compile(r'<table[^>]*>.*?</table>', re.IGNORECASE | re.DOTALL),
                # CSV-like
                re.compile(r'(?:[^,\n]+,){2,}[^,\n]*\n(?:[^,\n]+,){2,}[^,\n]*'),
                # Tab-separated
                re.compile(r'(?:[^\t\n]+\t){2,}[^\t\n]*\n(?:[^\t\n]+\t){2,}[^\t\n]*')
            ]
            
            # Count structure indicators
            structure_matches = 0
            for pattern in structure_patterns:
                structure_matches += len(pattern.findall(text))
            
            # Normalize structure score
            analysis['structure_score'] = min(1.0, structure_matches / max(1, analysis['line_count'] / 10))
            
            # Count QA indicators
            qa_matches = 0
            for pattern in qa_patterns:
                qa_matches += len(pattern.findall(text))
            
            # Normalize QA score
            analysis['qa_score'] = min(1.0, qa_matches / max(1, analysis['word_count'] / 100))
            
            # Count table indicators
            table_matches = 0
            for pattern in table_patterns:
                table_matches += len(pattern.findall(text))
            
            # Normalize table score
            analysis['table_score'] = min(1.0, table_matches / max(1, analysis['line_count'] / 20))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Document content analysis failed: {e}")
            return analysis
    
    def _select_strategy_from_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Select chunking strategy based on document analysis.
        
        Args:
            analysis: Document analysis results
            
        Returns:
            Selected strategy name
        """
        try:
            structure_score = analysis['structure_score']
            qa_score = analysis['qa_score']
            table_score = analysis['table_score']
            
            # Thresholds for strategy selection
            high_threshold = 0.3
            medium_threshold = 0.1
            
            # Strategy selection logic
            if table_score > high_threshold:
                return "table_chunking"
            elif qa_score > high_threshold:
                return "qa_chunking"
            elif structure_score > high_threshold:
                return "hierarchical"
            elif structure_score > medium_threshold:
                # Some structure detected, use hierarchical
                return "hierarchical"
            else:
                # Plain text, use naive merge
                return "naive_merge"
                
        except Exception as e:
            logger.error(f"Strategy selection from analysis failed: {e}")
            return self.default_strategy
    
    async def analyze_document_for_strategy(self, 
                                         document_parts: List[Tuple[str, Optional[Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Analyze document and return analysis results with recommended strategy.
        
        Args:
            document_parts: List of (text, metadata) tuples from document parser
            
        Returns:
            Analysis results with recommended strategy
        """
        try:
            # Combine all text for analysis
            full_text = "\n\n".join([part[0] for part in document_parts if part[0]])
            
            if not full_text.strip():
                return {
                    'analysis': {'error': 'No text content found'},
                    'recommended_strategy': self.default_strategy,
                    'available_strategies': self.get_available_strategies(),
                    'confidence': 0.0
                }
            
            # Analyze document
            analysis = await self._analyze_document_content(full_text)
            recommended_strategy = self._select_strategy_from_analysis(analysis)
            
            # Calculate confidence
            confidence = self._calculate_strategy_confidence(analysis)
            
            return {
                'analysis': analysis,
                'recommended_strategy': recommended_strategy,
                'available_strategies': self.get_available_strategies(),
                'confidence': confidence,
                'document_stats': {
                    'total_parts': len(document_parts),
                    'total_length': len(full_text),
                    'word_count': analysis['word_count'],
                    'line_count': analysis['line_count']
                }
            }
            
        except Exception as e:
            logger.error(f"Document analysis for strategy failed: {e}")
            return {
                'error': str(e),
                'recommended_strategy': self.default_strategy,
                'available_strategies': self.get_available_strategies(),
                'confidence': 0.0
            }
    
    def _calculate_strategy_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in strategy selection."""
        try:
            scores = [
                analysis['structure_score'],
                analysis['qa_score'],
                analysis['table_score']
            ]
            
            # Confidence is higher when one score is clearly dominant
            max_score = max(scores)
            
            if max_score > 0.5:
                return 0.9
            elif max_score > 0.3:
                return 0.7
            elif max_score > 0.1:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.0


# Global instance
chunking_manager = ChunkingManager()