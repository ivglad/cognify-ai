"""
Synonym expansion service for query enhancement.
"""
import logging
from typing import List, Dict, Set, Optional
import json

import trio
import nltk
from nltk.corpus import wordnet

from app.core.cache import cache_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class SynonymExpander:
    """
    Synonym expansion service using WordNet and custom dictionaries.
    """
    
    def __init__(self):
        self.cache_manager = cache_manager
        self.custom_synonyms: Dict[str, List[str]] = {}
        self.cache_prefix = "synonyms:"
        self._initialized = False
        
    async def initialize(self):
        """Initialize the synonym expander."""
        if self._initialized:
            return
            
        try:
            # Download required NLTK data
            await trio.to_thread.run_sync(self._download_nltk_data)
            
            # Load custom synonyms
            await self._load_custom_synonyms()
            
            self._initialized = True
            logger.info("SynonymExpander initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize SynonymExpander: {e}")
            raise
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    async def _load_custom_synonyms(self):
        """Load custom synonym dictionaries."""
        # Russian synonyms
        russian_synonyms = {
            "документ": ["файл", "бумага", "текст", "материал"],
            "поиск": ["поисковый", "найти", "искать", "обнаружить"],
            "система": ["платформа", "сервис", "приложение", "программа"],
            "данные": ["информация", "сведения", "материалы", "контент"],
            "анализ": ["исследование", "изучение", "разбор", "обработка"],
            "результат": ["итог", "вывод", "заключение", "следствие"],
            "процесс": ["операция", "действие", "процедура", "метод"],
            "пользователь": ["клиент", "юзер", "человек", "персона"],
            "компания": ["организация", "предприятие", "фирма", "корпорация"],
            "проект": ["задача", "работа", "дело", "инициатива"]
        }
        
        # English synonyms
        english_synonyms = {
            "document": ["file", "paper", "text", "material", "record"],
            "search": ["find", "look", "seek", "discover", "locate"],
            "system": ["platform", "service", "application", "program"],
            "data": ["information", "content", "material", "details"],
            "analysis": ["research", "study", "examination", "processing"],
            "result": ["outcome", "conclusion", "finding", "output"],
            "process": ["operation", "action", "procedure", "method"],
            "user": ["client", "person", "individual", "customer"],
            "company": ["organization", "business", "firm", "corporation"],
            "project": ["task", "work", "job", "initiative"]
        }
        
        # Combine dictionaries
        self.custom_synonyms.update(russian_synonyms)
        self.custom_synonyms.update(english_synonyms)
        
        logger.info(f"Loaded {len(self.custom_synonyms)} custom synonym entries")
    
    async def expand_query(self, 
                          query: str, 
                          max_synonyms: int = 3,
                          use_wordnet: bool = True,
                          use_custom: bool = True) -> Dict[str, Any]:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            max_synonyms: Maximum synonyms per term
            use_wordnet: Use WordNet for synonyms
            use_custom: Use custom synonym dictionary
            
        Returns:
            Expanded query information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check cache first
            cache_key = f"{self.cache_prefix}{query}:{max_synonyms}:{use_wordnet}:{use_custom}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Tokenize query
            tokens = await trio.to_thread.run_sync(self._tokenize_query, query)
            
            # Expand each token
            expanded_terms = {}
            all_synonyms = set()
            
            for token in tokens:
                synonyms = await self._get_synonyms(
                    token, 
                    max_synonyms=max_synonyms,
                    use_wordnet=use_wordnet,
                    use_custom=use_custom
                )
                
                if synonyms:
                    expanded_terms[token] = synonyms
                    all_synonyms.update(synonyms)
            
            # Build expanded query
            expanded_query = self._build_expanded_query(query, expanded_terms)
            
            result = {
                "original_query": query,
                "expanded_query": expanded_query,
                "expanded_terms": expanded_terms,
                "all_synonyms": list(all_synonyms),
                "expansion_count": len(all_synonyms)
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return {
                "original_query": query,
                "expanded_query": query,
                "expanded_terms": {},
                "all_synonyms": [],
                "expansion_count": 0,
                "error": str(e)
            }
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize query into words."""
        try:
            # Simple tokenization - can be enhanced with spaCy
            tokens = nltk.word_tokenize(query.lower())
            
            # Filter out punctuation and short words
            filtered_tokens = []
            for token in tokens:
                if token.isalpha() and len(token) > 2:
                    filtered_tokens.append(token)
            
            return filtered_tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed, using simple split: {e}")
            return [word.lower().strip() for word in query.split() if len(word.strip()) > 2]
    
    async def _get_synonyms(self, 
                           word: str, 
                           max_synonyms: int = 3,
                           use_wordnet: bool = True,
                           use_custom: bool = True) -> List[str]:
        """Get synonyms for a word."""
        synonyms = set()
        
        # Get custom synonyms
        if use_custom and word in self.custom_synonyms:
            synonyms.update(self.custom_synonyms[word][:max_synonyms])
        
        # Get WordNet synonyms
        if use_wordnet and len(synonyms) < max_synonyms:
            wordnet_synonyms = await trio.to_thread.run_sync(
                self._get_wordnet_synonyms, 
                word, 
                max_synonyms - len(synonyms)
            )
            synonyms.update(wordnet_synonyms)
        
        # Remove the original word and return
        synonyms.discard(word)
        return list(synonyms)[:max_synonyms]
    
    def _get_wordnet_synonyms(self, word: str, max_count: int) -> List[str]:
        """Get synonyms from WordNet."""
        try:
            synonyms = set()
            
            # Get synsets for the word
            synsets = wordnet.synsets(word)
            
            for synset in synsets:
                # Get lemmas (word forms) from synset
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    if synonym != word and len(synonym) > 2:
                        synonyms.add(synonym)
                        
                        if len(synonyms) >= max_count:
                            break
                
                if len(synonyms) >= max_count:
                    break
            
            return list(synonyms)[:max_count]
            
        except Exception as e:
            logger.warning(f"WordNet synonym lookup failed for '{word}': {e}")
            return []
    
    def _build_expanded_query(self, 
                             original_query: str, 
                             expanded_terms: Dict[str, List[str]]) -> str:
        """Build expanded query string."""
        if not expanded_terms:
            return original_query
        
        # Simple expansion - add synonyms with OR operator
        expanded_parts = []
        
        for word in original_query.split():
            clean_word = word.lower().strip('.,!?;:')
            
            if clean_word in expanded_terms:
                synonyms = expanded_terms[clean_word]
                # Create OR group: (original OR synonym1 OR synonym2)
                or_group = f"({clean_word} OR {' OR '.join(synonyms)})"
                expanded_parts.append(or_group)
            else:
                expanded_parts.append(word)
        
        return ' '.join(expanded_parts)
    
    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache."""
        try:
            if self.cache_manager:
                cached_data = await self.cache_manager.get(cache_key)
                if cached_data:
                    return cached_data
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result."""
        try:
            if self.cache_manager:
                await self.cache_manager.set(
                    cache_key,
                    result,
                    settings.SEARCH_CACHE_TTL
                )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    async def add_custom_synonyms(self, synonyms_dict: Dict[str, List[str]]):
        """Add custom synonyms to the dictionary."""
        self.custom_synonyms.update(synonyms_dict)
        logger.info(f"Added {len(synonyms_dict)} custom synonym entries")
    
    async def get_synonym_stats(self) -> Dict[str, Any]:
        """Get synonym expansion statistics."""
        return {
            "custom_synonyms_count": len(self.custom_synonyms),
            "wordnet_available": self._is_wordnet_available(),
            "cache_prefix": self.cache_prefix,
            "initialized": self._initialized
        }
    
    def _is_wordnet_available(self) -> bool:
        """Check if WordNet is available."""
        try:
            wordnet.synsets("test")
            return True
        except Exception:
            return False