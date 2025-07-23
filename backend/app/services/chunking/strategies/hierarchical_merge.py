"""
Hierarchical chunking strategy for structured documents.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import trio

from app.services.chunking.strategies.base_chunker import BaseChunker
from app.services.nlp.rag_tokenizer import rag_tokenizer

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Represents a section in a hierarchical document."""
    level: int
    title: str
    content: str
    start_pos: int
    end_pos: int
    parent: Optional['DocumentSection'] = None
    children: List['DocumentSection'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class HierarchicalMergeChunker(BaseChunker):
    """
    Hierarchical chunking strategy that preserves document structure.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100,
                 max_heading_level: int = 6):
        """
        Initialize hierarchical chunker.
        
        Args:
            chunk_size: Target size for chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_chunk_size: Minimum chunk size in tokens
            max_heading_level: Maximum heading level to consider (1-6)
        """
        super().__init__(chunk_size, chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self.max_heading_level = max_heading_level
        self.tokenizer = rag_tokenizer
        
        # Heading patterns for different formats
        self.heading_patterns = [
            # Markdown headings
            (re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE), 'markdown'),
            # HTML headings
            (re.compile(r'<h([1-6])[^>]*>(.+?)</h[1-6]>', re.IGNORECASE | re.DOTALL), 'html'),
            # Text headings (all caps, underlined, etc.)
            (re.compile(r'^([A-ZА-Я][A-ZА-Я\s]{2,})$', re.MULTILINE), 'text_caps'),
            # Numbered headings
            (re.compile(r'^(\d+(?:\.\d+)*\.?)\s+(.+)$', re.MULTILINE), 'numbered'),
        ]
    
    async def chunk_text(self, 
                        text: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using hierarchical structure.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata for the text
            
        Returns:
            List of chunk dictionaries with hierarchical metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize tokenizer if needed
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Detect document structure
            sections = await self._detect_document_structure(text)
            
            if not sections:
                # Fallback to simple chunking if no structure detected
                return await self._simple_hierarchical_chunk(text, metadata)
            
            # Create hierarchical chunks
            chunks = await self._create_hierarchical_chunks(sections, text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Hierarchical chunking failed: {e}")
            # Fallback to simple chunking
            return await self._simple_hierarchical_chunk(text, metadata)
    
    async def _detect_document_structure(self, text: str) -> List[DocumentSection]:
        """
        Detect hierarchical structure in document.
        
        Args:
            text: Input text
            
        Returns:
            List of document sections
        """
        sections = []
        
        # Try each heading pattern
        for pattern, pattern_type in self.heading_patterns:
            detected_sections = await self._extract_sections_by_pattern(text, pattern, pattern_type)
            
            if detected_sections:
                # Use the first pattern that finds sections
                sections = detected_sections
                logger.debug(f"Detected {len(sections)} sections using {pattern_type} pattern")
                break
        
        # If no headings found, try to detect by paragraph structure
        if not sections:
            sections = await self._detect_paragraph_structure(text)
        
        # Build hierarchy
        if sections:
            sections = self._build_section_hierarchy(sections)
        
        return sections
    
    async def _extract_sections_by_pattern(self, 
                                         text: str, 
                                         pattern: re.Pattern, 
                                         pattern_type: str) -> List[DocumentSection]:
        """Extract sections using a specific pattern."""
        sections = []
        
        try:
            matches = list(pattern.finditer(text))
            
            if not matches:
                return sections
            
            for i, match in enumerate(matches):
                # Determine heading level and title
                if pattern_type == 'markdown':
                    level = len(match.group(1))  # Number of # symbols
                    title = match.group(2).strip()
                elif pattern_type == 'html':
                    level = int(match.group(1))  # h1, h2, etc.
                    title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                elif pattern_type == 'text_caps':
                    level = 1  # Assume top level for caps headings
                    title = match.group(1).strip()
                elif pattern_type == 'numbered':
                    # Count dots to determine level
                    level = match.group(1).count('.') + 1
                    title = match.group(2).strip()
                else:
                    level = 1
                    title = match.group(0).strip()
                
                # Skip if level is too deep
                if level > self.max_heading_level:
                    continue
                
                # Determine content boundaries
                start_pos = match.end()
                
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(text)
                
                # Extract content
                content = text[start_pos:end_pos].strip()
                
                # Create section
                section = DocumentSection(
                    level=level,
                    title=title,
                    content=content,
                    start_pos=match.start(),
                    end_pos=end_pos
                )
                
                sections.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Section extraction failed for pattern {pattern_type}: {e}")
            return []
    
    async def _detect_paragraph_structure(self, text: str) -> List[DocumentSection]:
        """Detect structure based on paragraph breaks."""
        try:
            # Split by double newlines (paragraph breaks)
            paragraphs = re.split(r'\n\s*\n', text)
            
            if len(paragraphs) < 2:
                return []
            
            sections = []
            current_pos = 0
            
            for i, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    current_pos += len(paragraph) + 2  # +2 for \n\n
                    continue
                
                # Use first sentence as title if it's short enough
                sentences = re.split(r'[.!?]+', paragraph.strip())
                first_sentence = sentences[0].strip() if sentences else paragraph[:50]
                
                if len(first_sentence) > 100:
                    title = f"Section {i + 1}"
                else:
                    title = first_sentence
                
                section = DocumentSection(
                    level=1,
                    title=title,
                    content=paragraph.strip(),
                    start_pos=current_pos,
                    end_pos=current_pos + len(paragraph)
                )
                
                sections.append(section)
                current_pos += len(paragraph) + 2
            
            return sections
            
        except Exception as e:
            logger.error(f"Paragraph structure detection failed: {e}")
            return []
    
    def _build_section_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Build parent-child relationships between sections."""
        if not sections:
            return sections
        
        try:
            # Stack to track parent sections at each level
            parent_stack = []
            
            for section in sections:
                # Remove parents that are at same or deeper level
                while parent_stack and parent_stack[-1].level >= section.level:
                    parent_stack.pop()
                
                # Set parent if available
                if parent_stack:
                    section.parent = parent_stack[-1]
                    parent_stack[-1].children.append(section)
                
                # Add current section to stack
                parent_stack.append(section)
            
            return sections
            
        except Exception as e:
            logger.error(f"Hierarchy building failed: {e}")
            return sections
    
    async def _create_hierarchical_chunks(self, 
                                        sections: List[DocumentSection], 
                                        full_text: str,
                                        metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from hierarchical sections."""
        chunks = []
        
        try:
            for section in sections:
                section_chunks = await self._chunk_section(section, metadata)
                chunks.extend(section_chunks)
            
            # Add sequence numbers and relationships
            for i, chunk in enumerate(chunks):
                chunk['chunk_index'] = i
                chunk['total_chunks'] = len(chunks)
                
                # Add navigation metadata
                if i > 0:
                    chunk['previous_chunk'] = chunks[i - 1]['chunk_id']
                if i < len(chunks) - 1:
                    chunk['next_chunk'] = chunks[i + 1]['chunk_id']
            
            return chunks
            
        except Exception as e:
            logger.error(f"Hierarchical chunk creation failed: {e}")
            return []
    
    async def _chunk_section(self, 
                           section: DocumentSection, 
                           base_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a single section."""
        chunks = []
        
        try:
            # Prepare section text with title
            section_text = f"{section.title}\n\n{section.content}"
            
            # Tokenize section text
            tokens = await self.tokenizer.tokenize_text(section_text, remove_stopwords=False, stem_words=False)
            
            if len(tokens) <= self.chunk_size:
                # Section fits in one chunk
                chunk = await self._create_chunk(
                    text=section_text,
                    section=section,
                    chunk_index=0,
                    is_complete_section=True,
                    base_metadata=base_metadata
                )
                chunks.append(chunk)
            else:
                # Split section into multiple chunks
                section_chunks = await self._split_section(section, section_text, tokens, base_metadata)
                chunks.extend(section_chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Section chunking failed: {e}")
            return []
    
    async def _split_section(self, 
                           section: DocumentSection, 
                           section_text: str,
                           tokens: List[str],
                           base_metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split a large section into multiple chunks."""
        chunks = []
        
        try:
            # Calculate chunk boundaries
            chunk_starts = []
            current_pos = 0
            
            while current_pos < len(tokens):
                chunk_starts.append(current_pos)
                current_pos += self.chunk_size - self.chunk_overlap
            
            # Create chunks
            for i, start_pos in enumerate(chunk_starts):
                end_pos = min(start_pos + self.chunk_size, len(tokens))
                
                # Get chunk tokens
                chunk_tokens = tokens[start_pos:end_pos]
                
                # Reconstruct text from tokens (approximate)
                chunk_text = ' '.join(chunk_tokens)
                
                # Add section title to first chunk
                if i == 0:
                    chunk_text = f"{section.title}\n\n{chunk_text}"
                
                # Create chunk
                chunk = await self._create_chunk(
                    text=chunk_text,
                    section=section,
                    chunk_index=i,
                    is_complete_section=(len(chunk_starts) == 1),
                    base_metadata=base_metadata
                )
                
                # Add split metadata
                chunk['is_section_split'] = True
                chunk['section_chunk_index'] = i
                chunk['section_total_chunks'] = len(chunk_starts)
                
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Section splitting failed: {e}")
            return []
    
    async def _create_chunk(self, 
                          text: str,
                          section: DocumentSection,
                          chunk_index: int,
                          is_complete_section: bool,
                          base_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a chunk dictionary with hierarchical metadata."""
        try:
            # Calculate token count
            tokens = await self.tokenizer.tokenize_text(text, remove_stopwords=False, stem_words=False)
            token_count = len(tokens)
            
            # Create chunk metadata
            chunk_metadata = {
                'type': 'hierarchical',
                'section_title': section.title,
                'section_level': section.level,
                'section_start_pos': section.start_pos,
                'section_end_pos': section.end_pos,
                'chunk_index': chunk_index,
                'is_complete_section': is_complete_section,
                'token_count': token_count,
                'char_count': len(text)
            }
            
            # Add parent section information
            if section.parent:
                chunk_metadata['parent_section'] = {
                    'title': section.parent.title,
                    'level': section.parent.level
                }
                
                # Build breadcrumb trail
                breadcrumb = []
                current = section.parent
                while current:
                    breadcrumb.insert(0, current.title)
                    current = current.parent
                
                chunk_metadata['breadcrumb'] = breadcrumb
            
            # Add child sections information
            if section.children:
                chunk_metadata['child_sections'] = [
                    {'title': child.title, 'level': child.level}
                    for child in section.children
                ]
            
            # Merge with base metadata
            if base_metadata:
                chunk_metadata.update(base_metadata)
            
            # Create chunk
            chunk = {
                'chunk_id': f"hierarchical_{section.level}_{chunk_index}_{hash(text) % 10000}",
                'text': text,
                'metadata': chunk_metadata
            }
            
            return chunk
            
        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            return {
                'chunk_id': f"error_{chunk_index}",
                'text': text,
                'metadata': {'type': 'error', 'error': str(e)}
            }
    
    async def _simple_hierarchical_chunk(self, 
                                       text: str, 
                                       metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to simple chunking when no structure is detected."""
        try:
            # Use base chunker for simple splitting
            simple_chunks = await super().chunk_text(text, metadata)
            
            # Add hierarchical metadata
            for i, chunk in enumerate(simple_chunks):
                chunk['metadata']['type'] = 'hierarchical_fallback'
                chunk['metadata']['section_title'] = f"Section {i + 1}"
                chunk['metadata']['section_level'] = 1
                chunk['metadata']['is_complete_section'] = False
            
            return simple_chunks
            
        except Exception as e:
            logger.error(f"Simple hierarchical chunking failed: {e}")
            return []
    
    async def get_chunker_stats(self) -> Dict[str, Any]:
        """Get hierarchical chunker statistics."""
        base_stats = await super().get_chunker_stats()
        
        hierarchical_stats = {
            'min_chunk_size': self.min_chunk_size,
            'max_heading_level': self.max_heading_level,
            'supported_patterns': len(self.heading_patterns),
            'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
        }
        
        base_stats.update(hierarchical_stats)
        return base_stats