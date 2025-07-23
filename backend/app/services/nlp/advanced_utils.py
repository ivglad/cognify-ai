"""
Advanced NLP utilities for text processing, structure analysis, and content manipulation.
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from app.core.logging_config import get_logger
from app.services.nlp.rag_tokenizer import rag_tokenizer

logger = get_logger(__name__)


class TextStructureType(str, Enum):
    """Text structure types."""
    PLAIN = "plain"
    BULLETED = "bulleted"
    NUMBERED = "numbered"
    HIERARCHICAL = "hierarchical"
    TABLE_OF_CONTENTS = "table_of_contents"
    MIXED = "mixed"


class LanguageType(str, Enum):
    """Language types for detection."""
    ENGLISH = "english"
    CHINESE = "chinese"
    RUSSIAN = "russian"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class TextSection:
    """Text section with structure information."""
    content: str
    section_type: str
    level: int
    bullet_type: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


class AdvancedNLPUtils:
    """Advanced NLP utilities for text processing and analysis."""
    
    # Bullet patterns from RAGFlow
    BULLET_PATTERNS = [
        [r'^\s*[\d]+\.', r'^\s*[a-zA-Z]\.', r'^\s*[ivxlcdm]+\.'],  # Numbered
        [r'^\s*•', r'^\s*-', r'^\s*\*', r'^\s*\+'],  # Bulleted
        [r'^\s*第[一二三四五六七八九十]+章', r'^\s*Chapter\s+\d+', r'^\s*CHAPTER\s+\d+'],  # Chapters
        [r'^\s*\d+\.\d+', r'^\s*\d+\.\d+\.\d+'],  # Hierarchical numbering
    ]
    
    # Language detection patterns
    CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]')
    ENGLISH_PATTERN = re.compile(r'[a-zA-Z0-9\s.,\':;/\"?<>!\(\)\-]')
    RUSSIAN_PATTERN = re.compile(r'[а-яё]', re.IGNORECASE)
    
    # Table of contents patterns
    TOC_PATTERNS = [
        r'^\s*(table\s+of\s+contents|contents|目录|目錄|内容|оглавление|содержание)\s*$',
        r'^\s*\d+\.?\s+[^\n]{1,50}\s*\.{2,}\s*\d+\s*$',  # Page number references
        r'^\s*[IVX]+\.\s+[^\n]{1,50}\s*\.{2,}\s*\d+\s*$',  # Roman numerals with page numbers
    ]
    
    @staticmethod
    def bullets_category(sections: List[str]) -> int:
        """
        Categorize sections by bullet/numbering level.
        
        Args:
            sections: List of text sections
            
        Returns:
            Category index (0-3) representing the dominant bullet pattern
        """
        try:
            if not sections:
                return -1
            
            hits = [0] * len(AdvancedNLPUtils.BULLET_PATTERNS)
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                for i, patterns in enumerate(AdvancedNLPUtils.BULLET_PATTERNS):
                    for pattern in patterns:
                        if re.match(pattern, section, re.IGNORECASE):
                            hits[i] += 1
                            break
            
            # Return the pattern index with most hits
            return hits.index(max(hits)) if max(hits) > 0 else -1
            
        except Exception as e:
            logger.error(f"Bullet categorization failed: {e}")
            return -1
    
    @staticmethod
    def hierarchical_merge(bullet_levels: List[int], 
                          sections: List[str], 
                          depth: int = 5) -> List[str]:
        """
        Merge sections hierarchically based on bullet levels.
        
        Args:
            bullet_levels: List of bullet level indicators
            sections: List of text sections
            depth: Maximum hierarchy depth
            
        Returns:
            List of merged sections
        """
        try:
            if not sections or not bullet_levels:
                return sections
            
            if len(bullet_levels) != len(sections):
                logger.warning("Bullet levels and sections length mismatch")
                return sections
            
            merged = []
            current_group = []
            current_level = bullet_levels[0] if bullet_levels else 0
            
            for i, (section, level) in enumerate(zip(sections, bullet_levels)):
                section = section.strip()
                if not section:
                    if current_group:
                        current_group.append(section)
                    continue
                
                # Check if we should start a new group
                if level <= current_level and current_group and level >= 0:
                    # Finish current group
                    merged.append('\n'.join(current_group))
                    current_group = [section]
                    current_level = level
                else:
                    current_group.append(section)
                    if level >= 0:
                        current_level = level
            
            # Add final group
            if current_group:
                merged.append('\n'.join(current_group))
            
            return merged
            
        except Exception as e:
            logger.error(f"Hierarchical merge failed: {e}")
            return sections
    
    @staticmethod
    def naive_merge(sections: List[str], 
                   chunk_token_num: int = 128, 
                   delimiter: str = "\n。；！？") -> List[str]:
        """
        Simple text merging with token limits.
        
        Args:
            sections: List of text sections
            chunk_token_num: Maximum tokens per chunk
            delimiter: Delimiter characters for splitting
            
        Returns:
            List of merged chunks
        """
        try:
            if not sections:
                return []
            
            chunks = [""]
            token_counts = [0]
            
            def add_chunk(text: str):
                """Add text to current chunk or create new chunk."""
                tokens = AdvancedNLPUtils._count_tokens(text)
                if token_counts[-1] + tokens <= chunk_token_num:
                    if chunks[-1]:
                        chunks[-1] += "\n" + text
                    else:
                        chunks[-1] = text
                    token_counts[-1] += tokens
                else:
                    chunks.append(text)
                    token_counts.append(tokens)
            
            for section in sections:
                section = section.strip()
                if section:
                    add_chunk(section)
            
            return [chunk for chunk in chunks if chunk.strip()]
            
        except Exception as e:
            logger.error(f"Naive merge failed: {e}")
            return sections
    
    @staticmethod
    def tokenize_table(tables: List[Tuple[Any, Dict[str, Any]]], 
                      doc: Dict[str, Any], 
                      is_english: bool) -> List[Dict[str, Any]]:
        """
        Tokenize table content for indexing.
        
        Args:
            tables: List of (table_content, table_metadata) tuples
            doc: Document metadata
            is_english: Whether content is in English
            
        Returns:
            List of tokenized table documents
        """
        try:
            results = []
            delimiter = "; " if is_english else "； "
            
            for table_content, table_metadata in tables:
                if isinstance(table_content, list):
                    # Process table rows in batches
                    batch_size = 10
                    for i in range(0, len(table_content), batch_size):
                        batch = table_content[i:i + batch_size]
                        content = delimiter.join(str(item) for item in batch if item)
                        
                        if content.strip():
                            table_doc = doc.copy()
                            table_doc.update({
                                "content_with_weight": content,
                                "content_ltks": rag_tokenizer.tokenize(content),
                                "table_index": i // batch_size,
                                "table_metadata": table_metadata,
                                "is_table_content": True
                            })
                            results.append(table_doc)
                elif isinstance(table_content, str):
                    # Process string table content
                    if table_content.strip():
                        table_doc = doc.copy()
                        table_doc.update({
                            "content_with_weight": table_content,
                            "content_ltks": rag_tokenizer.tokenize(table_content),
                            "table_metadata": table_metadata,
                            "is_table_content": True
                        })
                        results.append(table_doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Table tokenization failed: {e}")
            return []    

    @staticmethod
    def is_english(texts: Union[str, List[str]]) -> bool:
        """
        Detect if text is primarily English.
        
        Args:
            texts: Text string or list of strings
            
        Returns:
            True if text is primarily English
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            if not texts:
                return False
            
            english_count = 0
            total_count = 0
            
            for text in texts:
                if isinstance(text, str) and text.strip():
                    total_count += len(text)
                    english_matches = AdvancedNLPUtils.ENGLISH_PATTERN.findall(text)
                    english_count += len(''.join(english_matches))
            
            if total_count == 0:
                return False
            
            return (english_count / total_count) > 0.7
            
        except Exception as e:
            logger.error(f"English detection failed: {e}")
            return False
    
    @staticmethod
    def is_chinese(text: str) -> bool:
        """
        Detect if text contains Chinese characters.
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Chinese characters
        """
        try:
            if not text:
                return False
            
            chinese_chars = AdvancedNLPUtils.CHINESE_PATTERN.findall(text)
            return len(chinese_chars) > 0
            
        except Exception as e:
            logger.error(f"Chinese detection failed: {e}")
            return False
    
    @staticmethod
    def is_russian(text: str) -> bool:
        """
        Detect if text contains Russian characters.
        
        Args:
            text: Input text
            
        Returns:
            True if text contains Russian characters
        """
        try:
            if not text:
                return False
            
            russian_chars = AdvancedNLPUtils.RUSSIAN_PATTERN.findall(text)
            return len(russian_chars) > 0
            
        except Exception as e:
            logger.error(f"Russian detection failed: {e}")
            return False
    
    @staticmethod
    def detect_language(text: str) -> LanguageType:
        """
        Detect the primary language of text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language type
        """
        try:
            if not text or not text.strip():
                return LanguageType.UNKNOWN
            
            is_eng = AdvancedNLPUtils.is_english(text)
            is_chi = AdvancedNLPUtils.is_chinese(text)
            is_rus = AdvancedNLPUtils.is_russian(text)
            
            # Count language indicators
            language_count = sum([is_eng, is_chi, is_rus])
            
            if language_count > 1:
                return LanguageType.MIXED
            elif is_eng:
                return LanguageType.ENGLISH
            elif is_chi:
                return LanguageType.CHINESE
            elif is_rus:
                return LanguageType.RUSSIAN
            else:
                return LanguageType.UNKNOWN
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return LanguageType.UNKNOWN
    
    @staticmethod
    def remove_contents_table(sections: List[str], is_english: bool = False) -> List[str]:
        """
        Remove table of contents from document sections.
        
        Args:
            sections: List of text sections
            is_english: Whether content is in English
            
        Returns:
            Filtered sections without table of contents
        """
        try:
            if not sections:
                return sections
            
            filtered_sections = []
            skip_next = 0
            
            for i, section in enumerate(sections):
                if skip_next > 0:
                    skip_next -= 1
                    continue
                
                section_lower = section.lower().strip()
                
                # Check for table of contents patterns
                is_toc = False
                for pattern in AdvancedNLPUtils.TOC_PATTERNS:
                    if re.match(pattern, section_lower, re.IGNORECASE):
                        is_toc = True
                        break
                
                if is_toc:
                    # Skip this section and potentially next few sections
                    skip_next = min(3, len(sections) - i - 1)  # Skip up to 3 following sections
                    continue
                
                # Check for numbered lists that might be TOC
                if re.match(r'^\s*\d+\.?\s+[^\n]{1,50}\s*\.{2,}\s*\d+\s*$', section):
                    continue
                
                # Check for chapter/section references
                if re.match(r'^\s*(chapter|section|part)\s+\d+', section_lower):
                    # Only skip if it's very short (likely a reference)
                    if len(section.strip()) < 100:
                        continue
                
                filtered_sections.append(section)
            
            return filtered_sections
            
        except Exception as e:
            logger.error(f"Contents table removal failed: {e}")
            return sections
    
    @staticmethod
    def detect_title_and_headers(sections: List[str]) -> List[TextSection]:
        """
        Detect titles and headers in text sections.
        
        Args:
            sections: List of text sections
            
        Returns:
            List of TextSection objects with structure information
        """
        try:
            if not sections:
                return []
            
            structured_sections = []
            
            for i, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue
                
                # Analyze section structure
                section_type = "text"
                level = 0
                confidence = 0.5
                bullet_type = None
                
                # Check for titles (usually short, capitalized, at beginning)
                if i == 0 and len(section) < 100 and section.isupper():
                    section_type = "title"
                    confidence = 0.9
                
                # Check for headers
                elif re.match(r'^\s*[A-Z][^.!?]*$', section) and len(section) < 200:
                    section_type = "header"
                    confidence = 0.7
                
                # Check for bullet points
                for j, patterns in enumerate(AdvancedNLPUtils.BULLET_PATTERNS):
                    for pattern in patterns:
                        if re.match(pattern, section, re.IGNORECASE):
                            section_type = "bullet"
                            level = j
                            bullet_type = pattern
                            confidence = 0.8
                            break
                    if section_type == "bullet":
                        break
                
                # Check for numbered sections
                if re.match(r'^\s*\d+\.\s+', section):
                    section_type = "numbered"
                    level = 1
                    confidence = 0.8
                elif re.match(r'^\s*\d+\.\d+\s+', section):
                    section_type = "numbered"
                    level = 2
                    confidence = 0.8
                elif re.match(r'^\s*\d+\.\d+\.\d+\s+', section):
                    section_type = "numbered"
                    level = 3
                    confidence = 0.8
                
                structured_section = TextSection(
                    content=section,
                    section_type=section_type,
                    level=level,
                    bullet_type=bullet_type,
                    confidence=confidence,
                    metadata={
                        'position': i,
                        'length': len(section),
                        'word_count': len(section.split())
                    }
                )
                
                structured_sections.append(structured_section)
            
            return structured_sections
            
        except Exception as e:
            logger.error(f"Title and header detection failed: {e}")
            return []
    
    @staticmethod
    def analyze_section_structure(sections: List[str]) -> Dict[str, Any]:
        """
        Analyze the overall structure of text sections.
        
        Args:
            sections: List of text sections
            
        Returns:
            Structure analysis results
        """
        try:
            if not sections:
                return {'structure_type': TextStructureType.PLAIN, 'confidence': 0.0}
            
            # Detect structured sections
            structured_sections = AdvancedNLPUtils.detect_title_and_headers(sections)
            
            # Count different structure types
            type_counts = {}
            for section in structured_sections:
                section_type = section.section_type
                type_counts[section_type] = type_counts.get(section_type, 0) + 1
            
            total_sections = len(structured_sections)
            
            # Determine overall structure type
            structure_type = TextStructureType.PLAIN
            confidence = 0.5
            
            # Check for hierarchical structure
            hierarchical_count = type_counts.get('numbered', 0) + type_counts.get('bullet', 0)
            if hierarchical_count > total_sections * 0.3:
                structure_type = TextStructureType.HIERARCHICAL
                confidence = 0.8
            
            # Check for bulleted structure
            elif type_counts.get('bullet', 0) > total_sections * 0.4:
                structure_type = TextStructureType.BULLETED
                confidence = 0.7
            
            # Check for numbered structure
            elif type_counts.get('numbered', 0) > total_sections * 0.4:
                structure_type = TextStructureType.NUMBERED
                confidence = 0.7
            
            # Check for mixed structure
            elif len([t for t, c in type_counts.items() if c > total_sections * 0.1]) > 2:
                structure_type = TextStructureType.MIXED
                confidence = 0.6
            
            return {
                'structure_type': structure_type,
                'confidence': confidence,
                'type_counts': type_counts,
                'total_sections': total_sections,
                'structured_sections': structured_sections,
                'language': AdvancedNLPUtils.detect_language(' '.join(sections[:5]))  # Sample first 5 sections
            }
            
        except Exception as e:
            logger.error(f"Section structure analysis failed: {e}")
            return {'structure_type': TextStructureType.PLAIN, 'confidence': 0.0}
    
    @staticmethod
    def extract_key_phrases(text: str, max_phrases: int = 10) -> List[Tuple[str, float]]:
        """
        Extract key phrases from text using simple heuristics.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of (phrase, score) tuples
        """
        try:
            if not text or not text.strip():
                return []
            
            # Simple key phrase extraction using frequency and position
            sentences = re.split(r'[.!?]+', text)
            words = re.findall(r'\b[a-zA-Zа-яё]{3,}\b', text.lower())
            
            # Count word frequencies
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Extract noun phrases (simple pattern matching)
            phrases = []
            
            # Look for capitalized words (potential proper nouns)
            proper_nouns = re.findall(r'\b[A-ZА-Я][a-zа-я]+\b', text)
            for noun in proper_nouns:
                if len(noun) > 2:
                    score = word_freq.get(noun.lower(), 1) * 2  # Boost proper nouns
                    phrases.append((noun, score))
            
            # Look for repeated important words
            for word, freq in word_freq.items():
                if freq > 1 and len(word) > 3:
                    phrases.append((word, freq))
            
            # Sort by score and return top phrases
            phrases.sort(key=lambda x: x[1], reverse=True)
            return phrases[:max_phrases]
            
        except Exception as e:
            logger.error(f"Key phrase extraction failed: {e}")
            return []
    
    @staticmethod
    def _count_tokens(text: str) -> int:
        """Count tokens in text using the tokenizer."""
        try:
            if hasattr(rag_tokenizer, 'count_tokens'):
                return rag_tokenizer.count_tokens(text)
            else:
                # Fallback: approximate token count
                return len(text.split())
                
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text.split())
    
    @staticmethod
    def clean_text(text: str, 
                  remove_extra_whitespace: bool = True,
                  remove_special_chars: bool = False,
                  normalize_quotes: bool = True) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            remove_extra_whitespace: Remove extra whitespace
            remove_special_chars: Remove special characters
            normalize_quotes: Normalize quote characters
            
        Returns:
            Cleaned text
        """
        try:
            if not text:
                return text
            
            cleaned = text
            
            # Remove extra whitespace
            if remove_extra_whitespace:
                cleaned = re.sub(r'\s+', ' ', cleaned)
                cleaned = cleaned.strip()
            
            # Normalize quotes
            if normalize_quotes:
                cleaned = re.sub(r'["""]', '"', cleaned)
                cleaned = re.sub(r'[''']', "'", cleaned)
            
            # Remove special characters (keep basic punctuation)
            if remove_special_chars:
                cleaned = re.sub(r'[^\w\s.,!?;:()\-"\']', '', cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return text


# Global utilities instance
advanced_nlp_utils = AdvancedNLPUtils()