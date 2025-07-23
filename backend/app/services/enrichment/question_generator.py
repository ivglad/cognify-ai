"""
Question generation service for content enrichment.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import Counter

import trio

from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.services.enrichment.keyword_extractor import keyword_extractor
from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Question generation service for creating questions from content.
    """
    
    def __init__(self,
                 max_questions: int = 10,
                 min_question_quality: float = 0.6,
                 question_types: Optional[List[str]] = None):
        """
        Initialize question generator.
        
        Args:
            max_questions: Maximum number of questions to generate
            min_question_quality: Minimum quality threshold for questions
            question_types: Types of questions to generate
        """
        self.max_questions = max_questions
        self.min_question_quality = min_question_quality
        self.question_types = question_types or [
            'what', 'how', 'why', 'when', 'where', 'who', 'which'
        ]
        
        self.tokenizer = rag_tokenizer
        self.keyword_extractor = keyword_extractor
        self.cache = cache_manager
        
        # Question templates for different types
        self.question_templates = {
            'what': [
                "Что такое {keyword}?",
                "Что представляет собой {keyword}?",
                "Что означает {keyword}?",
                "What is {keyword}?",
                "What does {keyword} mean?",
                "What represents {keyword}?"
            ],
            'how': [
                "Как работает {keyword}?",
                "Как использовать {keyword}?",
                "Как применяется {keyword}?",
                "How does {keyword} work?",
                "How to use {keyword}?",
                "How is {keyword} applied?"
            ],
            'why': [
                "Почему важен {keyword}?",
                "Почему используется {keyword}?",
                "Зачем нужен {keyword}?",
                "Why is {keyword} important?",
                "Why is {keyword} used?",
                "Why do we need {keyword}?"
            ],
            'when': [
                "Когда используется {keyword}?",
                "Когда применяется {keyword}?",
                "В каких случаях нужен {keyword}?",
                "When is {keyword} used?",
                "When do we apply {keyword}?",
                "In what cases is {keyword} needed?"
            ],
            'where': [
                "Где используется {keyword}?",
                "Где применяется {keyword}?",
                "В каких областях важен {keyword}?",
                "Where is {keyword} used?",
                "Where do we apply {keyword}?",
                "In which areas is {keyword} important?"
            ],
            'who': [
                "Кто использует {keyword}?",
                "Кто работает с {keyword}?",
                "Для кого предназначен {keyword}?",
                "Who uses {keyword}?",
                "Who works with {keyword}?",
                "Who is {keyword} intended for?"
            ],
            'which': [
                "Какие виды {keyword} существуют?",
                "Какие типы {keyword} бывают?",
                "Какие варианты {keyword} доступны?",
                "Which types of {keyword} exist?",
                "Which kinds of {keyword} are there?",
                "Which variants of {keyword} are available?"
            ]
        }
        
        # Patterns for extracting factual statements
        self.fact_patterns = [
            # Russian patterns
            re.compile(r'([А-ЯЁ][а-яё\s]+)\s+(?:является|представляет собой|означает)\s+(.+?)\.', re.IGNORECASE),
            re.compile(r'([А-ЯЁ][а-яё\s]+)\s*[-–—]\s*(.+?)\.', re.IGNORECASE),
            re.compile(r'(.+?)\s+называется\s+([а-яё\s]+)\.', re.IGNORECASE),
            # English patterns
            re.compile(r'([A-Z][a-z\s]+)\s+(?:is|are|means|represents)\s+(.+?)\.', re.IGNORECASE),
            re.compile(r'([A-Z][a-z\s]+)\s*[-–—]\s*(.+?)\.', re.IGNORECASE),
            re.compile(r'(.+?)\s+(?:is called|are called)\s+([a-z\s]+)\.', re.IGNORECASE)
        ]
    
    async def generate_questions(self, 
                               text: str,
                               document_id: Optional[str] = None,
                               use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Generate questions from text content.
        
        Args:
            text: Input text
            document_id: Optional document ID for caching
            use_cache: Whether to use caching
            
        Returns:
            List of generated questions with metadata
        """
        if not text or not text.strip():
            return []
        
        try:
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            # Check cache
            if use_cache and document_id:
                cache_key = f"questions:{document_id}"
                cached_questions = await self.cache.get(cache_key)
                if cached_questions:
                    logger.debug(f"Using cached questions for document {document_id}")
                    return cached_questions
            
            logger.debug(f"Generating questions from text ({len(text)} chars)")
            
            # Generate questions using multiple methods
            questions = await self._generate_multi_method(text)
            
            # Validate and filter questions
            validated_questions = await self._validate_questions(questions, text)
            
            # Cache results
            if use_cache and document_id and validated_questions:
                await self.cache.set(f"questions:{document_id}", validated_questions, ttl=86400)  # 24 hours
            
            logger.debug(f"Generated {len(validated_questions)} validated questions")
            
            return validated_questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return []
    
    async def _generate_multi_method(self, text: str) -> List[Dict[str, Any]]:
        """Generate questions using multiple methods."""
        try:
            # Method 1: Keyword-based questions
            keyword_questions = await self._generate_keyword_questions(text)
            
            # Method 2: Fact-based questions
            fact_questions = await self._generate_fact_questions(text)
            
            # Method 3: Structure-based questions
            structure_questions = await self._generate_structure_questions(text)
            
            # Method 4: Template-based questions
            template_questions = await self._generate_template_questions(text)
            
            # Combine all questions
            all_questions = []
            all_questions.extend(keyword_questions)
            all_questions.extend(fact_questions)
            all_questions.extend(structure_questions)
            all_questions.extend(template_questions)
            
            # Remove duplicates and rank
            unique_questions = await self._deduplicate_questions(all_questions)
            
            return unique_questions
            
        except Exception as e:
            logger.error(f"Multi-method question generation failed: {e}")
            return []
    
    async def _generate_keyword_questions(self, text: str) -> List[Dict[str, Any]]:
        """Generate questions based on extracted keywords."""
        try:
            # Extract keywords
            keywords = await self.keyword_extractor.extract_keywords(text)
            
            if not keywords:
                return []
            
            questions = []
            
            # Generate questions for top keywords
            for kw in keywords[:min(5, len(keywords))]:
                keyword = kw['keyword']
                
                # Determine language
                language = self._detect_language(keyword)
                
                # Generate questions for each type
                for question_type in self.question_types:
                    templates = self.question_templates.get(question_type, [])
                    
                    # Filter templates by language
                    filtered_templates = [
                        t for t in templates 
                        if (language == 'russian' and any(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in t.lower())) or
                           (language == 'english' and not any(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' for c in t.lower()))
                    ]
                    
                    if filtered_templates:
                        template = filtered_templates[0]  # Use first matching template
                        question_text = template.format(keyword=keyword)
                        
                        questions.append({
                            'question': question_text,
                            'type': question_type,
                            'method': 'keyword',
                            'source_keyword': keyword,
                            'keyword_score': kw.get('score', 0),
                            'language': language,
                            'quality_score': self._calculate_initial_quality(question_text, question_type)
                        })
            
            return questions
            
        except Exception as e:
            logger.error(f"Keyword-based question generation failed: {e}")
            return []
    
    async def _generate_fact_questions(self, text: str) -> List[Dict[str, Any]]:
        """Generate questions based on factual statements in text."""
        try:
            questions = []
            
            # Extract factual statements using patterns
            for pattern in self.fact_patterns:
                matches = pattern.findall(text)
                
                for match in matches:
                    if len(match) >= 2:
                        subject = match[0].strip()
                        description = match[1].strip()
                        
                        # Generate questions about the fact
                        fact_questions = self._create_fact_questions(subject, description)
                        questions.extend(fact_questions)
            
            return questions
            
        except Exception as e:
            logger.error(f"Fact-based question generation failed: {e}")
            return []
    
    def _create_fact_questions(self, subject: str, description: str) -> List[Dict[str, Any]]:
        """Create questions from a factual statement."""
        try:
            questions = []
            language = self._detect_language(subject + " " + description)
            
            if language == 'russian':
                # What question
                questions.append({
                    'question': f"Что такое {subject}?",
                    'type': 'what',
                    'method': 'fact',
                    'source_fact': f"{subject} - {description}",
                    'expected_answer': description,
                    'language': language,
                    'quality_score': 0.8
                })
                
                # How question (if description suggests a process)
                if any(word in description.lower() for word in ['процесс', 'метод', 'способ', 'технология']):
                    questions.append({
                        'question': f"Как работает {subject}?",
                        'type': 'how',
                        'method': 'fact',
                        'source_fact': f"{subject} - {description}",
                        'expected_answer': description,
                        'language': language,
                        'quality_score': 0.7
                    })
            
            else:  # English
                # What question
                questions.append({
                    'question': f"What is {subject}?",
                    'type': 'what',
                    'method': 'fact',
                    'source_fact': f"{subject} - {description}",
                    'expected_answer': description,
                    'language': language,
                    'quality_score': 0.8
                })
                
                # How question (if description suggests a process)
                if any(word in description.lower() for word in ['process', 'method', 'way', 'technology']):
                    questions.append({
                        'question': f"How does {subject} work?",
                        'type': 'how',
                        'method': 'fact',
                        'source_fact': f"{subject} - {description}",
                        'expected_answer': description,
                        'language': language,
                        'quality_score': 0.7
                    })
            
            return questions
            
        except Exception as e:
            logger.error(f"Fact question creation failed: {e}")
            return []
    
    async def _generate_structure_questions(self, text: str) -> List[Dict[str, Any]]:
        """Generate questions based on text structure (headings, lists, etc.)."""
        try:
            questions = []
            
            # Find headings
            heading_patterns = [
                re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),  # Markdown headings
                re.compile(r'^([А-ЯЁA-Z][А-ЯЁA-Z\s]{5,})$', re.MULTILINE),  # All caps headings
                re.compile(r'^\d+\.?\s+(.+)$', re.MULTILINE)  # Numbered headings
            ]
            
            for pattern in heading_patterns:
                matches = pattern.findall(text)
                
                for heading in matches:
                    heading = heading.strip()
                    if len(heading) > 5:  # Skip very short headings
                        language = self._detect_language(heading)
                        
                        if language == 'russian':
                            question_text = f"О чём рассказывается в разделе '{heading}'?"
                        else:
                            question_text = f"What is discussed in the section '{heading}'?"
                        
                        questions.append({
                            'question': question_text,
                            'type': 'what',
                            'method': 'structure',
                            'source_heading': heading,
                            'language': language,
                            'quality_score': 0.6
                        })
            
            # Find lists
            list_pattern = re.compile(r'(?:^|\n)\s*[-*•]\s+(.+)', re.MULTILINE)
            list_items = list_pattern.findall(text)
            
            if len(list_items) >= 3:  # Only if there are multiple list items
                language = self._detect_language(' '.join(list_items[:3]))
                
                if language == 'russian':
                    question_text = "Какие основные пункты перечислены в тексте?"
                else:
                    question_text = "What are the main points listed in the text?"
                
                questions.append({
                    'question': question_text,
                    'type': 'which',
                    'method': 'structure',
                    'source_items': list_items[:5],  # First 5 items
                    'language': language,
                    'quality_score': 0.7
                })
            
            return questions
            
        except Exception as e:
            logger.error(f"Structure-based question generation failed: {e}")
            return []
    
    async def _generate_template_questions(self, text: str) -> List[Dict[str, Any]]:
        """Generate questions using generic templates."""
        try:
            questions = []
            language = self._detect_language(text)
            
            # Generic questions based on text content
            if language == 'russian':
                generic_questions = [
                    "О чём говорится в данном тексте?",
                    "Какая основная идея представлена в тексте?",
                    "Какие ключевые моменты освещаются в материале?",
                    "Что можно узнать из данного текста?"
                ]
            else:
                generic_questions = [
                    "What is this text about?",
                    "What is the main idea presented in the text?",
                    "What key points are covered in the material?",
                    "What can be learned from this text?"
                ]
            
            for i, question_text in enumerate(generic_questions):
                questions.append({
                    'question': question_text,
                    'type': 'what',
                    'method': 'template',
                    'template_id': i,
                    'language': language,
                    'quality_score': 0.5  # Lower score for generic questions
                })
            
            return questions
            
        except Exception as e:
            logger.error(f"Template-based question generation failed: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text (Russian or English)."""
        try:
            if not text:
                return 'english'
            
            # Count Cyrillic vs Latin characters
            cyrillic_count = len(re.findall(r'[а-яёА-ЯЁ]', text))
            latin_count = len(re.findall(r'[a-zA-Z]', text))
            
            if cyrillic_count > latin_count:
                return 'russian'
            else:
                return 'english'
                
        except Exception:
            return 'english'
    
    def _calculate_initial_quality(self, question: str, question_type: str) -> float:
        """Calculate initial quality score for a question."""
        try:
            score = 0.5  # Base score
            
            # Length bonus
            if 10 <= len(question) <= 100:
                score += 0.2
            
            # Question type bonus
            type_bonuses = {
                'what': 0.2,
                'how': 0.3,
                'why': 0.3,
                'when': 0.1,
                'where': 0.1,
                'who': 0.1,
                'which': 0.2
            }
            score += type_bonuses.get(question_type, 0)
            
            # Grammar check (basic)
            if question.endswith('?'):
                score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    async def _deduplicate_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate questions and rank by quality."""
        try:
            if not questions:
                return []
            
            # Group similar questions
            unique_questions = {}
            
            for question in questions:
                question_text = question['question'].lower().strip()
                
                # Simple similarity check
                found_similar = False
                for existing_text in unique_questions.keys():
                    if self._questions_similar(question_text, existing_text):
                        # Keep the one with higher quality
                        if question.get('quality_score', 0) > unique_questions[existing_text].get('quality_score', 0):
                            del unique_questions[existing_text]
                            unique_questions[question_text] = question
                        found_similar = True
                        break
                
                if not found_similar:
                    unique_questions[question_text] = question
            
            # Convert back to list and sort by quality
            result = list(unique_questions.values())
            result.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            return result[:self.max_questions]
            
        except Exception as e:
            logger.error(f"Question deduplication failed: {e}")
            return questions[:self.max_questions]
    
    def _questions_similar(self, q1: str, q2: str) -> bool:
        """Check if two questions are similar."""
        try:
            # Simple similarity check based on common words
            words1 = set(q1.split())
            words2 = set(q2.split())
            
            if not words1 or not words2:
                return False
            
            intersection = words1 & words2
            union = words1 | words2
            
            similarity = len(intersection) / len(union)
            
            return similarity > 0.7
            
        except Exception:
            return False
    
    async def _validate_questions(self, 
                                questions: List[Dict[str, Any]], 
                                source_text: str) -> List[Dict[str, Any]]:
        """Validate and filter questions based on quality."""
        try:
            validated_questions = []
            
            for question in questions:
                # Calculate comprehensive quality score
                quality_score = await self._calculate_comprehensive_quality(question, source_text)
                question['final_quality_score'] = quality_score
                
                # Filter by minimum quality
                if quality_score >= self.min_question_quality:
                    validated_questions.append(question)
            
            # Sort by quality and return top questions
            validated_questions.sort(key=lambda x: x['final_quality_score'], reverse=True)
            
            return validated_questions[:self.max_questions]
            
        except Exception as e:
            logger.error(f"Question validation failed: {e}")
            return questions[:self.max_questions]
    
    async def _calculate_comprehensive_quality(self, 
                                             question: Dict[str, Any], 
                                             source_text: str) -> float:
        """Calculate comprehensive quality score for a question."""
        try:
            base_score = question.get('quality_score', 0.5)
            
            # Method bonus
            method_bonuses = {
                'fact': 0.3,
                'keyword': 0.2,
                'structure': 0.1,
                'template': 0.0
            }
            method_bonus = method_bonuses.get(question.get('method', 'template'), 0)
            
            # Relevance to source text
            relevance_score = await self._calculate_relevance(question, source_text)
            
            # Combine scores
            final_score = (base_score * 0.4 + 
                          method_bonus * 0.3 + 
                          relevance_score * 0.3)
            
            return min(1.0, final_score)
            
        except Exception as e:
            logger.error(f"Comprehensive quality calculation failed: {e}")
            return question.get('quality_score', 0.5)
    
    async def _calculate_relevance(self, 
                                 question: Dict[str, Any], 
                                 source_text: str) -> float:
        """Calculate relevance of question to source text."""
        try:
            question_text = question['question'].lower()
            source_text_lower = source_text.lower()
            
            # Extract key terms from question
            question_words = set(re.findall(r'\b\w+\b', question_text))
            source_words = set(re.findall(r'\b\w+\b', source_text_lower))
            
            # Remove common words
            common_words = {'что', 'как', 'где', 'когда', 'почему', 'кто', 'какой', 'какая', 'какие',
                           'what', 'how', 'where', 'when', 'why', 'who', 'which', 'is', 'are', 'the', 'a', 'an'}
            
            question_words -= common_words
            
            if not question_words:
                return 0.5
            
            # Calculate overlap
            overlap = question_words & source_words
            relevance = len(overlap) / len(question_words)
            
            return relevance
            
        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 0.5
    
    async def generate_questions_batch(self, 
                                     texts: List[str],
                                     document_ids: Optional[List[str]] = None) -> List[List[Dict[str, Any]]]:
        """Generate questions from multiple texts in batch."""
        try:
            if not texts:
                return []
            
            results = []
            
            # Process in parallel with limited concurrency
            async with trio.open_nursery() as nursery:
                semaphore = trio.Semaphore(3)  # Limit concurrent generations
                
                async def generate_single(text: str, doc_id: Optional[str]):
                    async with semaphore:
                        questions = await self.generate_questions(text, doc_id)
                        results.append(questions)
                
                for i, text in enumerate(texts):
                    doc_id = document_ids[i] if document_ids and i < len(document_ids) else None
                    nursery.start_soon(generate_single, text, doc_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch question generation failed: {e}")
            return [[] for _ in texts]
    
    async def get_generator_stats(self) -> Dict[str, Any]:
        """Get question generator statistics."""
        try:
            return {
                'max_questions': self.max_questions,
                'min_question_quality': self.min_question_quality,
                'question_types': self.question_types,
                'template_count': sum(len(templates) for templates in self.question_templates.values()),
                'fact_patterns_count': len(self.fact_patterns),
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False
            }
            
        except Exception as e:
            logger.error(f"Generator stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
question_generator = QuestionGenerator()