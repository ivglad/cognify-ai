"""
RAG tokenizer for Russian and English text processing with frequency analysis.
"""
import logging
import re
from typing import List, Dict, Tuple, Optional, Set
import unicodedata
from collections import defaultdict

import trio
import datrie

logger = logging.getLogger(__name__)


class RagTokenizer:
    """
    Custom tokenizer for Russian and English with frequency analysis and trie-based fast tokenization.
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.dictionary: Dict[str, int] = {}  # Token frequency dictionary
        self.trie: Optional[datrie.Trie] = None  # Trie for fast tokenization
        self.pos_tags: Dict[str, str] = {}  # Part-of-speech tags
        self._initialized = False
        
        # Language detection patterns
        self.cyrillic_pattern = re.compile(r'[а-яё]', re.IGNORECASE)
        self.latin_pattern = re.compile(r'[a-z]', re.IGNORECASE)
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        # Tokenization patterns
        self.word_pattern = re.compile(r'\b\w+\b')
        self.sentence_pattern = re.compile(r'[.!?]+')
        
        # Character normalization mappings
        self.q2b_mapping = self._create_q2b_mapping()
        
    def _create_q2b_mapping(self) -> Dict[str, str]:
        """Create full-width to half-width character mapping."""
        mapping = {}
        
        # Full-width ASCII to half-width
        for i in range(0xFF01, 0xFF5F):  # Full-width ASCII range
            full_char = chr(i)
            half_char = chr(i - 0xFEE0)
            mapping[full_char] = half_char
        
        # Full-width space
        mapping['\u3000'] = ' '
        
        return mapping
    
    async def initialize(self):
        """Initialize the tokenizer with dictionaries and trie."""
        if self._initialized:
            return
        
        try:
            # Load built-in frequency dictionary
            await self._load_frequency_dictionary()
            
            # Build trie for fast tokenization
            await self._build_trie()
            
            # Load POS tags if available
            await self._load_pos_tags()
            
            self._initialized = True
            logger.info("RagTokenizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RagTokenizer: {e}")
            # Continue without full initialization
            self._initialized = True
    
    async def _load_frequency_dictionary(self):
        """Load token frequency dictionary."""
        try:
            # Built-in frequency data for common Russian and English words
            russian_freq = {
                "и": 100000, "в": 95000, "не": 90000, "на": 85000, "я": 80000,
                "быть": 75000, "он": 70000, "с": 65000, "что": 60000, "а": 55000,
                "по": 50000, "это": 45000, "она": 40000, "к": 35000, "но": 30000,
                "они": 25000, "мы": 20000, "как": 18000, "из": 16000, "у": 14000,
                "который": 12000, "то": 10000, "за": 9000, "свой": 8000, "ее": 7000,
                "так": 6000, "его": 5500, "со": 5000, "для": 4500, "уже": 4000,
                "или": 3500, "да": 3000, "от": 2800, "все": 2600, "еще": 2400,
                "бы": 2200, "же": 2000, "до": 1800, "вы": 1600, "при": 1400,
                "документ": 1200, "система": 1100, "данные": 1000, "поиск": 900,
                "текст": 800, "файл": 700, "информация": 600, "результат": 500
            }
            
            english_freq = {
                "the": 100000, "be": 95000, "to": 90000, "of": 85000, "and": 80000,
                "a": 75000, "in": 70000, "that": 65000, "have": 60000, "i": 55000,
                "it": 50000, "for": 45000, "not": 40000, "on": 35000, "with": 30000,
                "he": 25000, "as": 20000, "you": 18000, "do": 16000, "at": 14000,
                "this": 12000, "but": 10000, "his": 9000, "by": 8000, "from": 7000,
                "they": 6000, "we": 5500, "say": 5000, "her": 4500, "she": 4000,
                "or": 3500, "an": 3000, "will": 2800, "my": 2600, "one": 2400,
                "all": 2200, "would": 2000, "there": 1800, "their": 1600, "what": 1400,
                "document": 1200, "system": 1100, "data": 1000, "search": 900,
                "text": 800, "file": 700, "information": 600, "result": 500
            }
            
            # Combine dictionaries
            self.dictionary.update(russian_freq)
            self.dictionary.update(english_freq)
            
            logger.info(f"Loaded {len(self.dictionary)} tokens in frequency dictionary")
            
        except Exception as e:
            logger.error(f"Failed to load frequency dictionary: {e}")
            self.dictionary = {}
    
    async def _build_trie(self):
        """Build trie for fast tokenization."""
        try:
            if not self.dictionary:
                logger.warning("No dictionary available for trie building")
                return
            
            # Create trie with all dictionary keys
            alphabet = set()
            for word in self.dictionary.keys():
                alphabet.update(word.lower())
            
            # Convert to sorted string for datrie
            alphabet_str = ''.join(sorted(alphabet))
            
            self.trie = datrie.Trie(alphabet_str)
            
            # Add all dictionary words to trie
            for word, freq in self.dictionary.items():
                try:
                    self.trie[word.lower()] = freq
                except Exception as e:
                    if self.debug:
                        logger.debug(f"Failed to add word '{word}' to trie: {e}")
            
            logger.info(f"Built trie with {len(self.trie)} entries")
            
        except Exception as e:
            logger.error(f"Failed to build trie: {e}")
            self.trie = None
    
    async def _load_pos_tags(self):
        """Load part-of-speech tags."""
        try:
            # Basic POS tags for common words
            basic_pos = {
                # Russian
                "и": "CONJ", "в": "PREP", "не": "PART", "на": "PREP", "я": "PRON",
                "быть": "VERB", "он": "PRON", "с": "PREP", "что": "PRON", "а": "CONJ",
                "документ": "NOUN", "система": "NOUN", "данные": "NOUN", "поиск": "NOUN",
                
                # English
                "the": "DET", "be": "VERB", "to": "PREP", "of": "PREP", "and": "CONJ",
                "a": "DET", "in": "PREP", "that": "PRON", "have": "VERB", "i": "PRON",
                "document": "NOUN", "system": "NOUN", "data": "NOUN", "search": "NOUN"
            }
            
            self.pos_tags.update(basic_pos)
            
        except Exception as e:
            logger.error(f"Failed to load POS tags: {e}")
            self.pos_tags = {}
    
    async def tokenize(self, text: str) -> str:
        """
        Main tokenization method with language detection.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            Tokenized text as string
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not text or not text.strip():
                return ""
            
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Split by language segments
            language_segments = self._split_by_lang(normalized_text)
            
            # Tokenize each segment
            tokenized_segments = []
            
            for segment_text, is_chinese in language_segments:
                if is_chinese:
                    # Chinese tokenization (simplified)
                    tokens = self._tokenize_chinese(segment_text)
                else:
                    # Russian/English tokenization
                    tokens = self._tokenize_western(segment_text)
                
                tokenized_segments.append(' '.join(tokens))
            
            return ' '.join(tokenized_segments)
            
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return text  # Return original text on failure
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text with character conversion and cleaning."""
        try:
            # Full-width to half-width conversion
            normalized = self._strQ2B(text)
            
            # Traditional to simplified Chinese (if needed)
            normalized = self._tradi2simp(normalized)
            
            # Unicode normalization
            normalized = unicodedata.normalize('NFKC', normalized)
            
            # Clean up whitespace
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            return normalized
            
        except Exception as e:
            logger.debug(f"Text normalization failed: {e}")
            return text
    
    def _strQ2B(self, text: str) -> str:
        """Convert full-width to half-width characters."""
        try:
            result = []
            for char in text:
                if char in self.q2b_mapping:
                    result.append(self.q2b_mapping[char])
                else:
                    result.append(char)
            return ''.join(result)
        except Exception as e:
            logger.debug(f"Q2B conversion failed: {e}")
            return text
    
    def _tradi2simp(self, text: str) -> str:
        """Convert traditional Chinese to simplified (placeholder)."""
        try:
            # This is a placeholder - in real implementation you'd use hanziconv
            # For now, just return the text as-is
            return text
        except Exception as e:
            logger.debug(f"Traditional to simplified conversion failed: {e}")
            return text
    
    def _split_by_lang(self, text: str) -> List[Tuple[str, bool]]:
        """
        Split text by language (Chinese/non-Chinese detection).
        
        Args:
            text: Input text
            
        Returns:
            List of (text_segment, is_chinese) tuples
        """
        try:
            segments = []
            current_segment = ""
            current_is_chinese = None
            
            for char in text:
                is_chinese = bool(self.chinese_pattern.match(char))
                
                if current_is_chinese is None:
                    current_is_chinese = is_chinese
                    current_segment = char
                elif current_is_chinese == is_chinese:
                    current_segment += char
                else:
                    # Language change
                    if current_segment.strip():
                        segments.append((current_segment.strip(), current_is_chinese))
                    current_segment = char
                    current_is_chinese = is_chinese
            
            # Add final segment
            if current_segment.strip():
                segments.append((current_segment.strip(), current_is_chinese))
            
            return segments
            
        except Exception as e:
            logger.debug(f"Language splitting failed: {e}")
            return [(text, False)]  # Treat as non-Chinese
    
    def _tokenize_chinese(self, text: str) -> List[str]:
        """Tokenize Chinese text (simplified approach)."""
        try:
            # Simple character-based tokenization for Chinese
            # In real implementation, you'd use jieba or similar
            tokens = []
            
            for char in text:
                if char.strip() and not char.isspace():
                    tokens.append(char)
            
            return tokens
            
        except Exception as e:
            logger.debug(f"Chinese tokenization failed: {e}")
            return [text]
    
    def _tokenize_western(self, text: str) -> List[str]:
        """Tokenize Western (Russian/English) text."""
        try:
            # Use trie-based tokenization if available
            if self.trie:
                return self._trie_tokenize(text)
            else:
                return self._simple_tokenize(text)
                
        except Exception as e:
            logger.debug(f"Western tokenization failed: {e}")
            return self._simple_tokenize(text)
    
    def _trie_tokenize(self, text: str) -> List[str]:
        """Fast trie-based tokenization."""
        try:
            tokens = []
            text_lower = text.lower()
            i = 0
            
            while i < len(text_lower):
                # Find longest match in trie
                longest_match = ""
                
                for j in range(i + 1, min(i + 20, len(text_lower) + 1)):  # Limit search length
                    candidate = text_lower[i:j]
                    if candidate in self.trie:
                        longest_match = candidate
                
                if longest_match:
                    tokens.append(text[i:i + len(longest_match)])  # Preserve original case
                    i += len(longest_match)
                else:
                    # No match found, use simple word boundary
                    match = self.word_pattern.match(text, i)
                    if match:
                        tokens.append(match.group())
                        i = match.end()
                    else:
                        i += 1
            
            return [token for token in tokens if token.strip()]
            
        except Exception as e:
            logger.debug(f"Trie tokenization failed: {e}")
            return self._simple_tokenize(text)
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple regex-based tokenization."""
        try:
            # Extract words using regex
            tokens = self.word_pattern.findall(text)
            return [token for token in tokens if token.strip()]
            
        except Exception as e:
            logger.debug(f"Simple tokenization failed: {e}")
            return text.split()
    
    async def fine_grained_tokenize(self, tokens: str) -> str:
        """
        Fine-grained tokenization for better accuracy.
        
        Args:
            tokens: Pre-tokenized text
            
        Returns:
            Fine-grained tokenized text
        """
        try:
            if not tokens or not tokens.strip():
                return ""
            
            # Split into individual tokens
            token_list = tokens.split()
            
            # Apply fine-grained processing to each token
            fine_tokens = []
            
            for token in token_list:
                # Check if token can be further split
                sub_tokens = self._split_compound_token(token)
                fine_tokens.extend(sub_tokens)
            
            return ' '.join(fine_tokens)
            
        except Exception as e:
            logger.error(f"Fine-grained tokenization failed: {e}")
            return tokens
    
    def _split_compound_token(self, token: str) -> List[str]:
        """Split compound tokens into sub-tokens."""
        try:
            # Simple heuristics for compound word splitting
            if len(token) < 6:  # Too short to be compound
                return [token]
            
            # Check for common compound patterns
            # This is simplified - real implementation would be more sophisticated
            
            # Camel case splitting
            if re.search(r'[a-z][A-Z]', token):
                parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
                if len(parts) > 1:
                    return parts
            
            # Underscore splitting
            if '_' in token:
                return token.split('_')
            
            # Hyphen splitting
            if '-' in token:
                return token.split('-')
            
            return [token]
            
        except Exception as e:
            logger.debug(f"Compound token splitting failed: {e}")
            return [token]
    
    def freq(self, token: str) -> int:
        """
        Get token frequency from built-in dictionary.
        
        Args:
            token: Token to look up
            
        Returns:
            Frequency count (0 if not found)
        """
        try:
            return self.dictionary.get(token.lower(), 0)
        except Exception as e:
            logger.debug(f"Frequency lookup failed for '{token}': {e}")
            return 0
    
    def tag(self, token: str) -> str:
        """
        Get part-of-speech tag for token.
        
        Args:
            token: Token to tag
            
        Returns:
            POS tag or 'UNK' if unknown
        """
        try:
            return self.pos_tags.get(token.lower(), 'UNK')
        except Exception as e:
            logger.debug(f"POS tagging failed for '{token}': {e}")
            return 'UNK'
    
    def detect_language(self, text: str) -> str:
        """
        Detect primary language of text.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('ru', 'en', 'zh', 'mixed', 'unknown')
        """
        try:
            if not text or not text.strip():
                return 'unknown'
            
            # Count characters by language
            cyrillic_count = len(self.cyrillic_pattern.findall(text))
            latin_count = len(self.latin_pattern.findall(text))
            chinese_count = len(self.chinese_pattern.findall(text))
            
            total_chars = cyrillic_count + latin_count + chinese_count
            
            if total_chars == 0:
                return 'unknown'
            
            # Calculate percentages
            cyrillic_pct = cyrillic_count / total_chars
            latin_pct = latin_count / total_chars
            chinese_pct = chinese_count / total_chars
            
            # Determine primary language
            if chinese_pct > 0.3:
                return 'zh'
            elif cyrillic_pct > 0.5:
                return 'ru'
            elif latin_pct > 0.5:
                return 'en'
            elif cyrillic_pct > 0.2 and latin_pct > 0.2:
                return 'mixed'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return 'unknown'
    
    async def get_tokenizer_stats(self) -> Dict[str, Any]:
        """Get tokenizer statistics."""
        try:
            return {
                "initialized": self._initialized,
                "dictionary_size": len(self.dictionary),
                "trie_available": self.trie is not None,
                "trie_size": len(self.trie) if self.trie else 0,
                "pos_tags_count": len(self.pos_tags),
                "supported_languages": ["ru", "en", "zh"],
                "debug_mode": self.debug
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
rag_tokenizer = RagTokenizer()