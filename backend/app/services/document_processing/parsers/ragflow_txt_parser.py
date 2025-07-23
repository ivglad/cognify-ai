"""
RAGFlow-style text file parser with intelligent chunking and structure detection.
"""
import re
import chardet
import trio
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import io

from app.core.logging_config import get_logger
from app.services.nlp.rag_tokenizer import rag_tokenizer

logger = get_logger(__name__)


class TextStructureType(str, Enum):
    """Text structure type enumeration."""
    PLAIN = "plain"
    BULLETED = "bulleted"
    NUMBERED = "numbered"
    HIERARCHICAL = "hierarchical"
    DIALOGUE = "dialogue"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class TextChunkingOptions:
    """Text chunking configuration options."""
    chunk_token_num: int = 128
    delimiter: str = "\n!?;。；！？"
    preserve_structure: bool = True
    detect_bullets: bool = True
    detect_hierarchy: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 1000
    overlap_size: int = 20
    encoding_detection: bool = True
    fallback_encoding: str = "utf-8"
    streaming_enabled: bool = True
    context_overlap: bool = True


@dataclass
class TextStructure:
    """Text structure analysis result."""
    structure_type: TextStructureType
    sections: List[Dict[str, Any]]
    bullet_patterns: List[str]
    hierarchy_levels: List[int]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ChunkContext:
    """Context information for chunk processing."""
    previous_chunk: Optional[str] = None
    next_chunk: Optional[str] = None
    section_info: Optional[Dict[str, Any]] = None
    file_position: Optional[Tuple[int, int]] = None
    overlap_content: Optional[str] = None


class RAGFlowTxtParser:
    """Advanced text file parser with intelligent chunking and structure detection."""
    
    def __init__(self, options: Optional[TextChunkingOptions] = None):
        self.options = options or TextChunkingOptions()
        self.tokenizer = rag_tokenizer
        
        # Large file processing settings
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.streaming_threshold = 10 * 1024 * 1024  # 10MB
        self.chunk_buffer_size = 1024 * 1024  # 1MB        

        # Bullet point patterns (from RAGFlow)
        self.bullet_patterns = [
            r'^\s*[\d]+\.',  # 1. 2. 3.
            r'^\s*[a-zA-Z]\.',  # a. b. c.
            r'^\s*[ivxlcdm]+\.',  # i. ii. iii.
            r'^\s*•',  # •
            r'^\s*-',  # -
            r'^\s*\*',  # *
            r'^\s*\+',  # +
            r'^\s*→',  # →
            r'^\s*►',  # ►
            r'^\s*▪',  # ▪
            r'^\s*▫',  # ▫
        ]
        
        # Hierarchical patterns
        self.hierarchy_patterns = [
            r'^\s*第[一二三四五六七八九十]+章',  # Chinese chapters
            r'^\s*Chapter\s+\d+',  # English chapters
            r'^\s*CHAPTER\s+\d+',  # English chapters (caps)
            r'^\s*第\d+章',  # Mixed Chinese chapters
            r'^\s*\d+\.\d+',  # 1.1, 1.2, etc.
            r'^\s*\d+\.\d+\.\d+',  # 1.1.1, 1.1.2, etc.
            r'^\s*[A-Z]\.',  # A. B. C.
            r'^\s*[IVX]+\.',  # I. II. III.
        ]
        
        # Dialogue patterns
        self.dialogue_patterns = [
            r'^\s*["""].*["""]$',  # Quoted text
            r'^\s*—\s*',  # Em dash dialogue
            r'^\s*-\s*[A-Za-zА-Яа-я]',  # Dash dialogue
            r'^\s*\w+:\s*',  # Name: dialogue
        ]
        
        # Code patterns
        self.code_patterns = [
            r'^\s*def\s+\w+',  # Python functions
            r'^\s*class\s+\w+',  # Python classes
            r'^\s*function\s+\w+',  # JavaScript functions
            r'^\s*var\s+\w+',  # JavaScript variables
            r'^\s*const\s+\w+',  # JavaScript constants
            r'^\s*let\s+\w+',  # JavaScript let
            r'^\s*#include',  # C/C++ includes
            r'^\s*import\s+',  # Python/Java imports
            r'^\s*from\s+\w+\s+import',  # Python imports
        ]
    
    async def __call__(self, 
                      filename: str, 
                      binary: Optional[bytes] = None, 
                      chunk_token_num: Optional[int] = None,
                      delimiter: Optional[str] = None) -> List[str]:
        """
        Parse text file with intelligent chunking.
        
        Args:
            filename: Name of the file
            binary: File content as bytes
            chunk_token_num: Override chunk token number
            delimiter: Override delimiter string
            
        Returns:
            List of text chunks
        """
        try:
            # Override options if provided
            if chunk_token_num:
                self.options.chunk_token_num = chunk_token_num
            if delimiter:
                self.options.delimiter = delimiter
            
            # Check file size for streaming decision
            file_size = len(binary) if binary else self._get_file_size(filename)
            
            if file_size > self.streaming_threshold and self.options.streaming_enabled:
                # Use streaming processing for large files
                return await self._process_large_file_streaming(filename, binary)
            else:
                # Use standard processing for smaller files
                return await self._process_standard_file(filename, binary)
                
        except Exception as e:
            logger.error(f"Text parsing failed for {filename}: {e}")
            return []
    
    async def _process_standard_file(self, filename: str, binary: Optional[bytes] = None) -> List[str]:
        """Process file using standard in-memory approach."""
        try:
            # Get text content
            text = await self._get_text_content(filename, binary)
            
            if not text.strip():
                return []
            
            # Analyze text structure
            structure = await self._analyze_text_structure(text)
            
            # Parse text based on structure
            chunks = await self._parse_text_with_structure(text, structure)
            
            # Add context overlap if enabled
            if self.options.context_overlap:
                chunks = await self._add_context_overlap(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Standard file processing failed: {e}")
            return []
    
    def _get_file_size(self, filename: str) -> int:
        """Get file size."""
        try:
            import os
            return os.path.getsize(filename)
        except Exception:
            return 0
    
    async def _get_text_content(self, filename: str, binary: Optional[bytes] = None) -> str:
        """Get text content with encoding detection."""
        try:
            if binary is None:
                # Read file
                with open(filename, 'rb') as f:
                    binary = f.read()
            
            if not binary:
                return ""
            
            # Check file size limit
            if len(binary) > self.max_file_size:
                logger.warning(f"File size {len(binary)} exceeds limit {self.max_file_size}")
                # Truncate file
                binary = binary[:self.max_file_size]
            
            # Detect encoding if enabled
            if self.options.encoding_detection:
                detected = chardet.detect(binary)
                encoding = detected.get('encoding', self.options.fallback_encoding)
                confidence = detected.get('confidence', 0.0)
                
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                
                # Use detected encoding if confidence is high enough
                if confidence > 0.7 and encoding:
                    try:
                        return binary.decode(encoding)
                    except UnicodeDecodeError:
                        logger.warning(f"Failed to decode with detected encoding {encoding}, using fallback")
            
            # Fallback to default encoding
            try:
                return binary.decode(self.options.fallback_encoding)
            except UnicodeDecodeError:
                # Last resort: decode with errors='replace'
                return binary.decode(self.options.fallback_encoding, errors='replace')
                
        except Exception as e:
            logger.error(f"Text content extraction failed: {e}")
            return ""    
    a
sync def _process_large_file_streaming(self, filename: str, binary: Optional[bytes] = None) -> List[str]:
        """Process large file using streaming approach."""
        try:
            chunks = []
            
            if binary:
                # Process from bytes
                text_stream = io.StringIO(binary.decode(self.options.fallback_encoding, errors='replace'))
            else:
                # Process from file
                text_stream = open(filename, 'r', encoding=self.options.fallback_encoding, errors='replace')
            
            try:
                async for chunk_batch in self._stream_process_text(text_stream):
                    chunks.extend(chunk_batch)
            finally:
                if not binary:
                    text_stream.close()
            
            return chunks
            
        except Exception as e:
            logger.error(f"Streaming file processing failed: {e}")
            # Fallback to standard processing
            return await self._process_standard_file(filename, binary)
    
    async def _stream_process_text(self, text_stream) -> AsyncGenerator[List[str], None]:
        """Stream process text in chunks."""
        try:
            buffer = ""
            chunk_buffer = []
            
            while True:
                # Read chunk from stream
                chunk = text_stream.read(self.chunk_buffer_size)
                if not chunk:
                    break
                
                buffer += chunk
                
                # Process complete lines
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    chunk_buffer.append(line)
                    
                    # Check if we have enough content to process
                    if len(chunk_buffer) >= 100:  # Process every 100 lines
                        text_chunk = '\n'.join(chunk_buffer)
                        processed_chunks = await self._process_text_chunk(text_chunk)
                        yield processed_chunks
                        chunk_buffer = []
                
                # Yield control to event loop
                await asyncio.sleep(0)
            
            # Process remaining buffer
            if buffer or chunk_buffer:
                if buffer:
                    chunk_buffer.append(buffer)
                text_chunk = '\n'.join(chunk_buffer)
                processed_chunks = await self._process_text_chunk(text_chunk)
                yield processed_chunks
                
        except Exception as e:
            logger.error(f"Stream text processing failed: {e}")
    
    async def _process_text_chunk(self, text_chunk: str) -> List[str]:
        """Process a chunk of text."""
        try:
            # Simple chunking for streaming
            return self.parser_txt(text_chunk, self.options.chunk_token_num, self.options.delimiter)
            
        except Exception as e:
            logger.error(f"Text chunk processing failed: {e}")
            return [text_chunk] if text_chunk.strip() else []
    
    async def _analyze_text_structure(self, text: str) -> TextStructure:
        """Analyze text structure to determine optimal parsing strategy."""
        try:
            lines = text.split('\n')
            
            # Count different structure types
            bullet_count = 0
            hierarchy_count = 0
            dialogue_count = 0
            code_count = 0
            
            bullet_patterns_found = []
            hierarchy_levels = []
            
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                # Check for bullet points
                for pattern in self.bullet_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        bullet_count += 1
                        if pattern not in bullet_patterns_found:
                            bullet_patterns_found.append(pattern)
                        break
                
                # Check for hierarchy
                for i, pattern in enumerate(self.hierarchy_patterns):
                    if re.match(pattern, line, re.IGNORECASE):
                        hierarchy_count += 1
                        hierarchy_levels.append(i)
                        break
                
                # Check for dialogue
                for pattern in self.dialogue_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        dialogue_count += 1
                        break
                
                # Check for code
                for pattern in self.code_patterns:
                    if re.match(pattern, line, re.IGNORECASE):
                        code_count += 1
                        break
            
            total_lines = len([line for line in lines if line.strip()])
            
            # Determine structure type
            structure_type = TextStructureType.PLAIN
            confidence = 0.5
            
            if code_count > total_lines * 0.1:
                structure_type = TextStructureType.CODE
                confidence = 0.9
            elif hierarchy_count > total_lines * 0.05:
                structure_type = TextStructureType.HIERARCHICAL
                confidence = 0.8
            elif bullet_count > total_lines * 0.1:
                if len(set(hierarchy_levels)) > 1:
                    structure_type = TextStructureType.HIERARCHICAL
                else:
                    structure_type = TextStructureType.BULLETED
                confidence = 0.7
            elif dialogue_count > total_lines * 0.2:
                structure_type = TextStructureType.DIALOGUE
                confidence = 0.6
            
            # Check for mixed structure
            structure_counts = [bullet_count, hierarchy_count, dialogue_count, code_count]
            if sum(1 for count in structure_counts if count > total_lines * 0.05) > 1:
                structure_type = TextStructureType.MIXED
                confidence = 0.6
            
            # Create sections based on structure
            sections = await self._create_sections(lines, structure_type)
            
            return TextStructure(
                structure_type=structure_type,
                sections=sections,
                bullet_patterns=bullet_patterns_found,
                hierarchy_levels=list(set(hierarchy_levels)),
                confidence=confidence,
                metadata={
                    'total_lines': total_lines,
                    'bullet_count': bullet_count,
                    'hierarchy_count': hierarchy_count,
                    'dialogue_count': dialogue_count,
                    'code_count': code_count
                }
            )
            
        except Exception as e:
            logger.error(f"Text structure analysis failed: {e}")
            return TextStructure(
                structure_type=TextStructureType.PLAIN,
                sections=[],
                bullet_patterns=[],
                hierarchy_levels=[],
                confidence=0.5,
                metadata={}
            )   
 
    async def _create_sections(self, lines: List[str], structure_type: TextStructureType) -> List[Dict[str, Any]]:
        """Create sections based on detected structure type."""
        try:
            sections = []
            current_section = []
            section_id = 0
            
            if structure_type == TextStructureType.HIERARCHICAL:
                # Group by hierarchical levels
                current_level = -1
                
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        if current_section:
                            current_section.append(line)
                        continue
                    
                    # Check if this is a new hierarchical level
                    new_level = -1
                    for i, pattern in enumerate(self.hierarchy_patterns):
                        if re.match(pattern, line, re.IGNORECASE):
                            new_level = i
                            break
                    
                    if new_level != -1 and new_level <= current_level and current_section:
                        # Start new section
                        sections.append({
                            'id': section_id,
                            'type': 'hierarchical',
                            'level': current_level,
                            'lines': current_section.copy(),
                            'content': '\n'.join(current_section)
                        })
                        section_id += 1
                        current_section = [line]
                        current_level = new_level
                    else:
                        current_section.append(line)
                        if new_level != -1:
                            current_level = new_level
                
            elif structure_type == TextStructureType.BULLETED:
                # Group by bullet points
                for line in lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        if current_section:
                            current_section.append(line)
                        continue
                    
                    # Check if this is a new bullet point
                    is_bullet = False
                    for pattern in self.bullet_patterns:
                        if re.match(pattern, line, re.IGNORECASE):
                            is_bullet = True
                            break
                    
                    if is_bullet and current_section:
                        # Start new section
                        sections.append({
                            'id': section_id,
                            'type': 'bullet',
                            'lines': current_section.copy(),
                            'content': '\n'.join(current_section)
                        })
                        section_id += 1
                        current_section = [line]
                    else:
                        current_section.append(line)
            
            else:
                # For other types, create sections by paragraphs
                for line in lines:
                    if line.strip():
                        current_section.append(line)
                    else:
                        if current_section:
                            sections.append({
                                'id': section_id,
                                'type': 'paragraph',
                                'lines': current_section.copy(),
                                'content': '\n'.join(current_section)
                            })
                            section_id += 1
                            current_section = []
            
            # Add final section
            if current_section:
                sections.append({
                    'id': section_id,
                    'type': 'final',
                    'lines': current_section.copy(),
                    'content': '\n'.join(current_section)
                })
            
            return sections
            
        except Exception as e:
            logger.error(f"Section creation failed: {e}")
            return []
    
    async def _parse_text_with_structure(self, text: str, structure: TextStructure) -> List[str]:
        """Parse text into chunks based on detected structure."""
        try:
            if structure.structure_type == TextStructureType.PLAIN:
                return await self._parse_plain_text(text)
            elif structure.structure_type == TextStructureType.HIERARCHICAL:
                return await self._parse_hierarchical_text(structure.sections)
            elif structure.structure_type == TextStructureType.BULLETED:
                return await self._parse_bulleted_text(structure.sections)
            elif structure.structure_type == TextStructureType.CODE:
                return await self._parse_code_text(text)
            elif structure.structure_type == TextStructureType.DIALOGUE:
                return await self._parse_dialogue_text(text)
            else:  # MIXED
                return await self._parse_mixed_text(structure.sections)
                
        except Exception as e:
            logger.error(f"Structured text parsing failed: {e}")
            return await self._parse_plain_text(text)
    
    async def _parse_plain_text(self, text: str) -> List[str]:
        """Parse plain text using delimiter-based chunking."""
        try:
            return self.parser_txt(text, self.options.chunk_token_num, self.options.delimiter)
            
        except Exception as e:
            logger.error(f"Plain text parsing failed: {e}")
            return [text]
    
    async def _parse_hierarchical_text(self, sections: List[Dict[str, Any]]) -> List[str]:
        """Parse hierarchical text preserving structure."""
        try:
            chunks = []
            current_chunk = ""
            current_tokens = 0
            
            for section in sections:
                section_content = section['content']
                section_tokens = self._count_tokens(section_content)
                
                # If section is too large, split it
                if section_tokens > self.options.max_chunk_size:
                    # Add current chunk if not empty
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        current_tokens = 0
                    
                    # Split large section
                    section_chunks = await self._split_large_section(section_content)
                    chunks.extend(section_chunks)
                    
                elif current_tokens + section_tokens > self.options.chunk_token_num:
                    # Start new chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = section_content
                    current_tokens = section_tokens
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + section_content
                    else:
                        current_chunk = section_content
                    current_tokens += section_tokens
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"Hierarchical text parsing failed: {e}")
            return ['\n'.join(section['content'] for section in sections)] 
   
    async def _parse_bulleted_text(self, sections: List[Dict[str, Any]]) -> List[str]:
        """Parse bulleted text preserving bullet structure."""
        try:
            chunks = []
            current_chunk = ""
            current_tokens = 0
            
            for section in sections:
                section_content = section['content']
                section_tokens = self._count_tokens(section_content)
                
                if current_tokens + section_tokens > self.options.chunk_token_num and current_chunk:
                    # Start new chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = section_content
                    current_tokens = section_tokens
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n" + section_content
                    else:
                        current_chunk = section_content
                    current_tokens += section_tokens
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"Bulleted text parsing failed: {e}")
            return ['\n'.join(section['content'] for section in sections)]
    
    async def _parse_code_text(self, text: str) -> List[str]:
        """Parse code text preserving code structure."""
        try:
            lines = text.split('\n')
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            i = 0
            while i < len(lines):
                line = lines[i]
                line_tokens = self._count_tokens(line)
                
                # Check if this starts a function/class/block
                is_block_start = any(re.match(pattern, line, re.IGNORECASE) for pattern in self.code_patterns)
                
                if is_block_start:
                    # Find the end of this block
                    block_lines = [line]
                    indent_level = len(line) - len(line.lstrip())
                    i += 1
                    
                    while i < len(lines):
                        next_line = lines[i]
                        if next_line.strip():  # Non-empty line
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= indent_level and not next_line.lstrip().startswith(('#', '//', '/*')):
                                # End of block
                                break
                        block_lines.append(next_line)
                        i += 1
                    
                    block_content = '\n'.join(block_lines)
                    block_tokens = self._count_tokens(block_content)
                    
                    # Check if we need to start a new chunk
                    if current_tokens + block_tokens > self.options.chunk_token_num and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = block_lines
                        current_tokens = block_tokens
                    else:
                        current_chunk.extend(block_lines)
                        current_tokens += block_tokens
                else:
                    # Regular line
                    if current_tokens + line_tokens > self.options.chunk_token_num and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = [line]
                        current_tokens = line_tokens
                    else:
                        current_chunk.append(line)
                        current_tokens += line_tokens
                    i += 1
            
            # Add final chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Code text parsing failed: {e}")
            return await self._parse_plain_text(text)
    
    async def _parse_dialogue_text(self, text: str) -> List[str]:
        """Parse dialogue text preserving conversation structure."""
        try:
            lines = text.split('\n')
            chunks = []
            current_chunk = []
            current_tokens = 0
            
            for line in lines:
                line_tokens = self._count_tokens(line)
                
                # Check if this is a new speaker
                is_new_speaker = any(re.match(pattern, line, re.IGNORECASE) for pattern in self.dialogue_patterns)
                
                if is_new_speaker and current_tokens + line_tokens > self.options.chunk_token_num and current_chunk:
                    # Start new chunk at speaker change
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_tokens = line_tokens
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens
            
            # Add final chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            
            return chunks
            
        except Exception as e:
            logger.error(f"Dialogue text parsing failed: {e}")
            return await self._parse_plain_text(text)
    
    async def _parse_mixed_text(self, sections: List[Dict[str, Any]]) -> List[str]:
        """Parse mixed structure text."""
        try:
            chunks = []
            current_chunk = ""
            current_tokens = 0
            
            for section in sections:
                section_content = section['content']
                section_tokens = self._count_tokens(section_content)
                
                if current_tokens + section_tokens > self.options.chunk_token_num and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = section_content
                    current_tokens = section_tokens
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + section_content
                    else:
                        current_chunk = section_content
                    current_tokens += section_tokens
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            logger.error(f"Mixed text parsing failed: {e}")
            return ['\n'.join(section['content'] for section in sections)]
    
    async def _split_large_section(self, content: str) -> List[str]:
        """Split a large section into smaller chunks."""
        try:
            # Use plain text parsing for large sections
            return self.parser_txt(content, self.options.chunk_token_num, self.options.delimiter)
            
        except Exception as e:
            logger.error(f"Large section splitting failed: {e}")
            return [content]
    
    async def _add_context_overlap(self, chunks: List[str]) -> List[str]:
        """Add context overlap between chunks."""
        try:
            if len(chunks) <= 1 or not self.options.context_overlap:
                return chunks
            
            overlapped_chunks = []
            
            for i, chunk in enumerate(chunks):
                enhanced_chunk = chunk
                
                # Add overlap from previous chunk
                if i > 0 and self.options.overlap_size > 0:
                    prev_chunk = chunks[i - 1]
                    prev_words = prev_chunk.split()
                    if len(prev_words) > self.options.overlap_size:
                        overlap = ' '.join(prev_words[-self.options.overlap_size:])
                        enhanced_chunk = f"[...{overlap}]\n\n{enhanced_chunk}"
                
                # Add overlap from next chunk
                if i < len(chunks) - 1 and self.options.overlap_size > 0:
                    next_chunk = chunks[i + 1]
                    next_words = next_chunk.split()
                    if len(next_words) > self.options.overlap_size:
                        overlap = ' '.join(next_words[:self.options.overlap_size])
                        enhanced_chunk = f"{enhanced_chunk}\n\n[{overlap}...]"
                
                overlapped_chunks.append(enhanced_chunk)
            
            return overlapped_chunks
            
        except Exception as e:
            logger.error(f"Context overlap addition failed: {e}")
            return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            if hasattr(self.tokenizer, 'count_tokens'):
                return self.tokenizer.count_tokens(text)
            else:
                # Fallback: approximate token count
                return len(text.split())
                
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text.split())
    
    @classmethod
    def parser_txt(cls, 
                  txt: str, 
                  chunk_token_num: int = 128, 
                  delimiter: str = "\n!?;。；！？") -> List[str]:
        """
        Parse text with configurable delimiters and chunking (RAGFlow style).
        
        Args:
            txt: Input text
            chunk_token_num: Maximum tokens per chunk
            delimiter: Delimiter characters
            
        Returns:
            List of text chunks
        """
        try:
            if not txt.strip():
                return []
            
            chunks = [""]
            token_nums = [0]
            
            # Convert delimiter escape sequences
            delimiter = delimiter.encode('utf-8').decode('unicode_escape').encode('latin1').decode('utf-8')
            
            def add_chunk(t: str):
                """Add text to current chunk or create new chunk."""
                tnum = cls._num_tokens_from_string(t)
                if token_nums[-1] + tnum <= chunk_token_num:
                    chunks[-1] += t
                    token_nums[-1] += tnum
                else:
                    chunks.append(t)
                    token_nums.append(tnum)
            
            # Split text by delimiters while preserving structure
            dels = [d for d in delimiter if d]
            dels = [re.escape(d) for d in dels if d]
            dels_pattern = "|".join(dels)
            
            if dels_pattern:
                sections = re.split(r"(%s)" % dels_pattern, txt)
            else:
                sections = [txt]
            
            for section in sections:
                if section.strip():
                    add_chunk(section)
            
            return [chunk for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Text parsing failed: {e}")
            return [txt] if txt.strip() else []
    
    @staticmethod
    def _num_tokens_from_string(string: str) -> int:
        """Estimate number of tokens in a string."""
        try:
            # Simple estimation: ~4 characters per token for mixed languages
            return max(1, len(string) // 4)
            
        except Exception:
            return len(string.split())
    
    async def handle_processing_failures(self, filename: str, error: Exception) -> List[str]:
        """Handle processing failures with fallback mechanisms."""
        try:
            logger.warning(f"Processing failed for {filename}, attempting fallback: {error}")
            
            # Try basic file reading
            try:
                with open(filename, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Simple chunking as last resort
                if content:
                    return self.parser_txt(content, self.options.chunk_token_num, self.options.delimiter)
                    
            except Exception as fallback_error:
                logger.error(f"Fallback processing also failed: {fallback_error}")
            
            return []
            
        except Exception as e:
            logger.error(f"Failure handling failed: {e}")
            return []


# Global instance
ragflow_txt_parser = RAGFlowTxtParser()