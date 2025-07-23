"""
QA chunking strategy for question-answer format documents.
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
class QAPair:
    """Represents a question-answer pair."""
    question: str
    answer: str
    start_pos: int
    end_pos: int
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QAChunkingStrategy(BaseChunker):
    """
    Chunking strategy for question-answer format documents.
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 min_qa_confidence: float = 0.6,
                 max_answer_length: int = 2000):
        """
        Initialize QA chunker.
        
        Args:
            chunk_size: Target size for chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            min_qa_confidence: Minimum confidence for QA pair detection
            max_answer_length: Maximum length for answers in characters
        """
        super().__init__(chunk_size, chunk_overlap)
        self.min_qa_confidence = min_qa_confidence
        self.max_answer_length = max_answer_length
        self.tokenizer = rag_tokenizer
        
        # QA patterns for different formats
        self.qa_patterns = [
            # Q: ... A: ... format
            {
                'pattern': re.compile(r'(?:^|\n)\s*(?:Q|Question|Вопрос)[:.]?\s*(.+?)\s*(?:A|Answer|Ответ)[:.]?\s*(.+?)(?=(?:\n\s*(?:Q|Question|Вопрос)[:.])|$)', 
                                    re.IGNORECASE | re.DOTALL | re.MULTILINE),
                'type': 'qa_colon',
                'confidence': 0.9
            },
            # FAQ format with numbers
            {
                'pattern': re.compile(r'(?:^|\n)\s*(\d+\.?\s*.+?\?)\s*\n\s*(.+?)(?=(?:\n\s*\d+\.)|$)', 
                                    re.DOTALL | re.MULTILINE),
                'type': 'numbered_faq',
                'confidence': 0.8
            },
            # Question followed by answer paragraph
            {
                'pattern': re.compile(r'(?:^|\n)\s*(.+\?)\s*\n\s*(.+?)(?=(?:\n\s*.+\?)|$)', 
                                    re.DOTALL | re.MULTILINE),
                'type': 'question_paragraph',
                'confidence': 0.7
            },
            # Interview format
            {
                'pattern': re.compile(r'(?:^|\n)\s*(?:Interviewer|Интервьюер)[:.]?\s*(.+?)\s*(?:Interviewee|Респондент|Response|Ответ)[:.]?\s*(.+?)(?=(?:\n\s*(?:Interviewer|Интервьюер))|$)', 
                                    re.IGNORECASE | re.DOTALL | re.MULTILINE),
                'type': 'interview',
                'confidence': 0.8
            },
            # Dialogue format
            {
                'pattern': re.compile(r'(?:^|\n)\s*([А-ЯA-Z][а-яa-z\s]+)[:.]?\s*(.+?)(?=(?:\n\s*[А-ЯA-Z][а-яa-z\s]+[:.])|$)', 
                                    re.DOTALL | re.MULTILINE),
                'type': 'dialogue',
                'confidence': 0.6
            }
        ]
        
        # Question indicators
        self.question_indicators = [
            # Russian
            r'\b(?:что|как|где|когда|почему|зачем|какой|какая|какие|кто|чем|откуда|куда)\b',
            # English
            r'\b(?:what|how|where|when|why|who|which|whose|whom)\b',
            # Question marks
            r'\?'
        ]
        
        self.question_pattern = re.compile('|'.join(self.question_indicators), re.IGNORECASE)
    
    async def chunk_text(self, 
                        text: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using QA detection.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata for the text
            
        Returns:
            List of chunk dictionaries with QA metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize tokenizer if needed
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Detect QA pairs
            qa_pairs = await self._detect_qa_pairs(text)
            
            if not qa_pairs:
                # Fallback to simple chunking if no QA pairs detected
                return await self._simple_qa_chunk(text, metadata)
            
            # Create QA chunks
            chunks = await self._create_qa_chunks(qa_pairs, text, metadata)
            
            return chunks
            
        except Exception as e:
            logger.error(f"QA chunking failed: {e}")
            # Fallback to simple chunking
            return await self._simple_qa_chunk(text, metadata)
    
    async def _detect_qa_pairs(self, text: str) -> List[QAPair]:
        """
        Detect question-answer pairs in text.
        
        Args:
            text: Input text
            
        Returns:
            List of QA pairs
        """
        qa_pairs = []
        
        try:
            # Try each QA pattern
            for pattern_info in self.qa_patterns:
                pattern = pattern_info['pattern']
                pattern_type = pattern_info['type']
                base_confidence = pattern_info['confidence']
                
                matches = list(pattern.finditer(text))
                
                for match in matches:
                    question = match.group(1).strip()
                    answer = match.group(2).strip()
                    
                    # Validate QA pair
                    confidence = await self._validate_qa_pair(question, answer, base_confidence)
                    
                    if confidence >= self.min_qa_confidence:
                        qa_pair = QAPair(
                            question=question,
                            answer=answer,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            metadata={
                                'pattern_type': pattern_type,
                                'question_length': len(question),
                                'answer_length': len(answer)
                            }
                        )
                        
                        qa_pairs.append(qa_pair)
                
                # If we found high-confidence pairs, use them
                if qa_pairs and max(pair.confidence for pair in qa_pairs) > 0.8:
                    break
            
            # Remove overlapping pairs (keep highest confidence)
            qa_pairs = self._remove_overlapping_pairs(qa_pairs)
            
            # Sort by position
            qa_pairs.sort(key=lambda x: x.start_pos)
            
            logger.debug(f"Detected {len(qa_pairs)} QA pairs")
            
            return qa_pairs
            
        except Exception as e:
            logger.error(f"QA pair detection failed: {e}")
            return []
    
    async def _validate_qa_pair(self, question: str, answer: str, base_confidence: float) -> float:
        """
        Validate and score a QA pair.
        
        Args:
            question: Question text
            answer: Answer text
            base_confidence: Base confidence from pattern matching
            
        Returns:
            Confidence score (0-1)
        """
        try:
            confidence = base_confidence
            
            # Check question indicators
            if self.question_pattern.search(question):
                confidence += 0.1
            
            # Check question mark
            if question.endswith('?'):
                confidence += 0.1
            
            # Penalize very short questions
            if len(question.split()) < 3:
                confidence -= 0.2
            
            # Penalize very long questions (likely not questions)
            if len(question.split()) > 50:
                confidence -= 0.3
            
            # Check answer length
            if len(answer) < 10:
                confidence -= 0.2
            elif len(answer) > self.max_answer_length:
                confidence -= 0.1
            
            # Check if answer looks like a proper answer
            if self._looks_like_answer(answer):
                confidence += 0.1
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"QA pair validation failed: {e}")
            return 0.0
    
    def _looks_like_answer(self, answer: str) -> bool:
        """Check if text looks like a proper answer."""
        try:
            # Check for answer indicators
            answer_indicators = [
                # Russian
                r'\b(?:да|нет|это|потому что|поскольку|так как|следовательно|таким образом)\b',
                # English
                r'\b(?:yes|no|because|since|therefore|thus|this|that|it)\b'
            ]
            
            answer_pattern = re.compile('|'.join(answer_indicators), re.IGNORECASE)
            
            if answer_pattern.search(answer):
                return True
            
            # Check if it's a complete sentence
            if answer.strip().endswith('.') or answer.strip().endswith('!'):
                return True
            
            # Check if it has reasonable length
            words = answer.split()
            if 5 <= len(words) <= 200:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _remove_overlapping_pairs(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Remove overlapping QA pairs, keeping highest confidence ones."""
        if not qa_pairs:
            return qa_pairs
        
        try:
            # Sort by confidence (descending)
            sorted_pairs = sorted(qa_pairs, key=lambda x: x.confidence, reverse=True)
            
            non_overlapping = []
            
            for pair in sorted_pairs:
                # Check if this pair overlaps with any already selected pair
                overlaps = False
                
                for selected_pair in non_overlapping:
                    if (pair.start_pos < selected_pair.end_pos and 
                        pair.end_pos > selected_pair.start_pos):
                        overlaps = True
                        break
                
                if not overlaps:
                    non_overlapping.append(pair)
            
            return non_overlapping
            
        except Exception as e:
            logger.error(f"Overlap removal failed: {e}")
            return qa_pairs
    
    async def _create_qa_chunks(self, 
                              qa_pairs: List[QAPair], 
                              full_text: str,
                              metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks from QA pairs."""
        chunks = []
        
        try:
            for i, qa_pair in enumerate(qa_pairs):
                # Create chunk for this QA pair
                chunk = await self._create_qa_chunk(qa_pair, i, metadata)
                chunks.append(chunk)
                
                # Check if we need to combine with adjacent pairs
                combined_chunk = await self._try_combine_with_next(qa_pairs, i, metadata)
                if combined_chunk:
                    chunks.append(combined_chunk)
            
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
            logger.error(f"QA chunk creation failed: {e}")
            return []
    
    async def _create_qa_chunk(self, 
                             qa_pair: QAPair, 
                             index: int,
                             base_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a chunk from a QA pair."""
        try:
            # Format QA text
            qa_text = f"Q: {qa_pair.question}\n\nA: {qa_pair.answer}"
            
            # Calculate token count
            tokens = await self.tokenizer.tokenize_text(qa_text, remove_stopwords=False, stem_words=False)
            token_count = len(tokens)
            
            # Create chunk metadata
            chunk_metadata = {
                'type': 'qa_pair',
                'question': qa_pair.question,
                'answer': qa_pair.answer,
                'qa_confidence': qa_pair.confidence,
                'pattern_type': qa_pair.metadata.get('pattern_type', 'unknown'),
                'question_length': qa_pair.metadata.get('question_length', len(qa_pair.question)),
                'answer_length': qa_pair.metadata.get('answer_length', len(qa_pair.answer)),
                'token_count': token_count,
                'char_count': len(qa_text),
                'start_pos': qa_pair.start_pos,
                'end_pos': qa_pair.end_pos
            }
            
            # Merge with base metadata
            if base_metadata:
                chunk_metadata.update(base_metadata)
            
            # Create chunk
            chunk = {
                'chunk_id': f"qa_{index}_{hash(qa_text) % 10000}",
                'text': qa_text,
                'metadata': chunk_metadata
            }
            
            return chunk
            
        except Exception as e:
            logger.error(f"QA chunk creation failed: {e}")
            return {
                'chunk_id': f"qa_error_{index}",
                'text': f"Q: {qa_pair.question}\n\nA: {qa_pair.answer}",
                'metadata': {'type': 'qa_error', 'error': str(e)}
            }
    
    async def _try_combine_with_next(self, 
                                   qa_pairs: List[QAPair], 
                                   current_index: int,
                                   base_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Try to combine current QA pair with next one if they fit in chunk size."""
        if current_index >= len(qa_pairs) - 1:
            return None
        
        try:
            current_pair = qa_pairs[current_index]
            next_pair = qa_pairs[current_index + 1]
            
            # Create combined text
            combined_text = (f"Q: {current_pair.question}\n\nA: {current_pair.answer}\n\n"
                           f"Q: {next_pair.question}\n\nA: {next_pair.answer}")
            
            # Check if combined text fits in chunk size
            tokens = await self.tokenizer.tokenize_text(combined_text, remove_stopwords=False, stem_words=False)
            
            if len(tokens) <= self.chunk_size:
                # Create combined chunk
                chunk_metadata = {
                    'type': 'qa_combined',
                    'qa_count': 2,
                    'questions': [current_pair.question, next_pair.question],
                    'answers': [current_pair.answer, next_pair.answer],
                    'qa_confidences': [current_pair.confidence, next_pair.confidence],
                    'token_count': len(tokens),
                    'char_count': len(combined_text),
                    'start_pos': current_pair.start_pos,
                    'end_pos': next_pair.end_pos
                }
                
                # Merge with base metadata
                if base_metadata:
                    chunk_metadata.update(base_metadata)
                
                chunk = {
                    'chunk_id': f"qa_combined_{current_index}_{hash(combined_text) % 10000}",
                    'text': combined_text,
                    'metadata': chunk_metadata
                }
                
                return chunk
            
            return None
            
        except Exception as e:
            logger.error(f"QA combination failed: {e}")
            return None
    
    async def _simple_qa_chunk(self, 
                             text: str, 
                             metadata: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback to simple chunking when no QA pairs are detected."""
        try:
            # Use base chunker for simple splitting
            simple_chunks = await super().chunk_text(text, metadata)
            
            # Add QA metadata
            for i, chunk in enumerate(simple_chunks):
                chunk['metadata']['type'] = 'qa_fallback'
                chunk['metadata']['qa_detection_failed'] = True
                
                # Try to detect questions in the chunk
                questions = self._find_questions_in_text(chunk['text'])
                if questions:
                    chunk['metadata']['detected_questions'] = questions
                    chunk['metadata']['question_count'] = len(questions)
            
            return simple_chunks
            
        except Exception as e:
            logger.error(f"Simple QA chunking failed: {e}")
            return []
    
    def _find_questions_in_text(self, text: str) -> List[str]:
        """Find potential questions in text."""
        try:
            # Split by sentences and find questions
            sentences = re.split(r'[.!?]+', text)
            questions = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if it looks like a question
                if (sentence.endswith('?') or 
                    self.question_pattern.search(sentence) or
                    any(sentence.lower().startswith(word) for word in ['что', 'как', 'где', 'когда', 'почему', 'what', 'how', 'where', 'when', 'why'])):
                    questions.append(sentence)
            
            return questions
            
        except Exception:
            return []
    
    async def get_chunker_stats(self) -> Dict[str, Any]:
        """Get QA chunker statistics."""
        base_stats = await super().get_chunker_stats()
        
        qa_stats = {
            'min_qa_confidence': self.min_qa_confidence,
            'max_answer_length': self.max_answer_length,
            'supported_patterns': len(self.qa_patterns),
            'question_indicators': len(self.question_indicators),
            'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
        }
        
        base_stats.update(qa_stats)
        return base_stats