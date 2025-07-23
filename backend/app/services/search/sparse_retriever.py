"""
Sparse retrieval system using Elasticsearch with custom analyzers.
"""
import logging
from typing import List, Dict, Any, Optional, Set
import json

import trio
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError

from app.services.nlp.rag_tokenizer import rag_tokenizer
from app.services.nlp.term_weighting import term_weighting_system
from app.core.elasticsearch_client import elasticsearch_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class SparseRetriever:
    """
    Sparse retrieval system using Elasticsearch with BM25 and custom analyzers.
    """
    
    def __init__(self,
                 index_name: str = "documents",
                 max_results: int = 100,
                 min_score: float = 0.1):
        """
        Initialize sparse retriever.
        
        Args:
            index_name: Elasticsearch index name
            max_results: Maximum number of results to return
            min_score: Minimum relevance score threshold
        """
        self.index_name = index_name
        self.max_results = max_results
        self.min_score = min_score
        
        self.es_client = elasticsearch_client
        self.tokenizer = rag_tokenizer
        self.term_weighting = term_weighting_system
        
        # Synonym dictionaries
        self.synonyms = {
            'english': [
                "ai,artificial intelligence,machine learning,ml",
                "api,application programming interface",
                "db,database,data base",
                "ui,user interface,gui,graphical user interface",
                "cpu,processor,central processing unit",
                "gpu,graphics processing unit,graphics card",
                "ram,memory,random access memory",
                "ssd,solid state drive,flash storage",
                "http,hypertext transfer protocol",
                "html,hypertext markup language",
                "css,cascading style sheets",
                "js,javascript,ecmascript"
            ],
            'russian': [
                "ии,искусственный интеллект,машинное обучение,мо",
                "апи,программный интерфейс приложения",
                "бд,база данных,базы данных",
                "пи,пользовательский интерфейс,интерфейс пользователя",
                "цп,центральный процессор,процессор",
                "гп,графический процессор,видеокарта",
                "озу,оперативная память,память",
                "ссд,твердотельный накопитель,флеш накопитель"
            ]
        }
    
    async def initialize_index(self) -> bool:
        """Initialize Elasticsearch index with custom settings and mappings."""
        try:
            # Check if index exists
            if await self.es_client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return True
            
            # Create index with custom settings
            index_settings = await self._get_index_settings()
            index_mappings = await self._get_index_mappings()
            
            await self.es_client.indices.create(
                index=self.index_name,
                body={
                    "settings": index_settings,
                    "mappings": index_mappings
                }
            )
            
            logger.info(f"Created Elasticsearch index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Index initialization failed: {e}")
            return False
    
    async def _get_index_settings(self) -> Dict[str, Any]:
        """Get index settings with custom analyzers."""
        return {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "filter": {
                    "english_synonyms": {
                        "type": "synonym",
                        "synonyms": self.synonyms['english']
                    },
                    "russian_synonyms": {
                        "type": "synonym", 
                        "synonyms": self.synonyms['russian']
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    },
                    "russian_stemmer": {
                        "type": "stemmer",
                        "language": "russian"
                    },
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english_"
                    },
                    "russian_stop": {
                        "type": "stop",
                        "stopwords": "_russian_"
                    }
                },
                "analyzer": {
                    "english_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "english_synonyms",
                            "english_stemmer"
                        ]
                    },
                    "russian_analyzer": {
                        "type": "custom", 
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "russian_stop",
                            "russian_synonyms", 
                            "russian_stemmer"
                        ]
                    },
                    "multilingual_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "russian_stop",
                            "english_synonyms",
                            "russian_synonyms"
                        ]
                    }
                }
            }
        }
    
    async def _get_index_mappings(self) -> Dict[str, Any]:
        """Get index mappings for document fields."""
        return {
            "properties": {
                "document_id": {
                    "type": "keyword"
                },
                "title": {
                    "type": "text",
                    "analyzer": "multilingual_analyzer",
                    "fields": {
                        "english": {
                            "type": "text",
                            "analyzer": "english_analyzer"
                        },
                        "russian": {
                            "type": "text", 
                            "analyzer": "russian_analyzer"
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "multilingual_analyzer",
                    "fields": {
                        "english": {
                            "type": "text",
                            "analyzer": "english_analyzer"
                        },
                        "russian": {
                            "type": "text",
                            "analyzer": "russian_analyzer"
                        }
                    }
                },
                "chunk_id": {
                    "type": "keyword"
                },
                "chunk_index": {
                    "type": "integer"
                },
                "keywords": {
                    "type": "text",
                    "analyzer": "keyword"
                },
                "entities": {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "confidence": {"type": "float"}
                    }
                },
                "tags": {
                    "type": "keyword"
                },
                "language": {
                    "type": "keyword"
                },
                "created_at": {
                    "type": "date"
                },
                "updated_at": {
                    "type": "date"
                }
            }
        }
    
    async def index_document(self, 
                           document_id: str,
                           chunks: List[Dict[str, Any]],
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index document chunks in Elasticsearch.
        
        Args:
            document_id: Document identifier
            chunks: List of document chunks
            metadata: Additional document metadata
            
        Returns:
            True if successful
        """
        try:
            if not chunks:
                logger.warning(f"No chunks to index for document {document_id}")
                return False
            
            # Initialize index if needed
            await self.initialize_index()
            
            # Prepare bulk indexing operations
            bulk_operations = []
            
            for i, chunk in enumerate(chunks):
                # Detect language
                chunk_text = chunk.get('text', '')
                language = await self._detect_language(chunk_text)
                
                # Prepare document for indexing
                doc = {
                    "document_id": document_id,
                    "chunk_id": chunk.get('chunk_id', f"{document_id}_chunk_{i}"),
                    "chunk_index": i,
                    "title": chunk.get('title', ''),
                    "content": chunk_text,
                    "language": language,
                    "keywords": chunk.get('keywords', []),
                    "entities": chunk.get('entities', []),
                    "tags": chunk.get('tags', []),
                    "created_at": chunk.get('created_at'),
                    "updated_at": chunk.get('updated_at')
                }
                
                # Add metadata
                if metadata:
                    doc.update(metadata)
                
                # Add to bulk operations
                bulk_operations.extend([
                    {"index": {"_index": self.index_name, "_id": doc["chunk_id"]}},
                    doc
                ])
            
            # Execute bulk indexing
            if bulk_operations:
                response = await self.es_client.bulk(body=bulk_operations)
                
                # Check for errors
                if response.get('errors'):
                    error_count = sum(1 for item in response['items'] if 'error' in item.get('index', {}))
                    logger.warning(f"Bulk indexing had {error_count} errors for document {document_id}")
                
                logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Document indexing failed for {document_id}: {e}")
            return False
    
    async def search(self, 
                    query: str,
                    filters: Optional[Dict[str, Any]] = None,
                    boost_terms: Optional[List[Dict[str, Any]]] = None,
                    size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search documents using sparse retrieval.
        
        Args:
            query: Search query
            filters: Optional filters to apply
            boost_terms: Optional terms to boost
            size: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            if not query or not query.strip():
                return []
            
            # Initialize services
            if not self.tokenizer._initialized:
                await self.tokenizer.initialize()
            
            size = size or self.max_results
            
            # Build Elasticsearch query
            es_query = await self._build_search_query(query, filters, boost_terms)
            
            # Execute search
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "query": es_query,
                    "size": size,
                    "min_score": self.min_score,
                    "_source": ["document_id", "chunk_id", "chunk_index", "title", "content", 
                               "keywords", "entities", "tags", "language"]
                }
            )
            
            # Process results
            results = await self._process_search_results(response, query)
            
            logger.debug(f"Sparse search returned {len(results)} results for query: {query}")
            
            return results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return [] 
   
    async def _build_search_query(self, 
                                query: str,
                                filters: Optional[Dict[str, Any]] = None,
                                boost_terms: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Build Elasticsearch query with BM25 and custom scoring."""
        try:
            # Detect query language
            language = await self._detect_language(query)
            
            # Build multi-match query
            multi_match_query = {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",  # Boost title matches
                        "content",
                        f"title.{language}^2",  # Language-specific fields
                        f"content.{language}"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "operator": "or"
                }
            }
            
            # Build should clauses for boosting
            should_clauses = [multi_match_query]
            
            # Add keyword matching
            should_clauses.append({
                "match": {
                    "keywords": {
                        "query": query,
                        "boost": 2.0
                    }
                }
            })
            
            # Add entity matching
            should_clauses.append({
                "nested": {
                    "path": "entities",
                    "query": {
                        "match": {
                            "entities.text": {
                                "query": query,
                                "boost": 1.5
                            }
                        }
                    }
                }
            })
            
            # Add boost terms if provided
            if boost_terms:
                for boost_term in boost_terms:
                    term_text = boost_term.get('term', '')
                    boost_value = boost_term.get('boost', 1.0)
                    
                    should_clauses.append({
                        "multi_match": {
                            "query": term_text,
                            "fields": ["title", "content"],
                            "boost": boost_value
                        }
                    })
            
            # Build main query
            main_query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
            
            # Add filters if provided
            if filters:
                filter_clauses = []
                
                # Document ID filter
                if 'document_ids' in filters:
                    filter_clauses.append({
                        "terms": {"document_id": filters['document_ids']}
                    })
                
                # Language filter
                if 'language' in filters:
                    filter_clauses.append({
                        "term": {"language": filters['language']}
                    })
                
                # Tag filter
                if 'tags' in filters:
                    filter_clauses.append({
                        "terms": {"tags": filters['tags']}
                    })
                
                # Date range filter
                if 'date_range' in filters:
                    date_range = filters['date_range']
                    filter_clauses.append({
                        "range": {
                            "created_at": {
                                "gte": date_range.get('from'),
                                "lte": date_range.get('to')
                            }
                        }
                    })
                
                if filter_clauses:
                    main_query = {
                        "bool": {
                            "must": [main_query],
                            "filter": filter_clauses
                        }
                    }
            
            return main_query
            
        except Exception as e:
            logger.error(f"Search query building failed: {e}")
            return {"match_all": {}}
    
    async def _process_search_results(self, 
                                    response: Dict[str, Any], 
                                    original_query: str) -> List[Dict[str, Any]]:
        """Process Elasticsearch search results."""
        try:
            results = []
            hits = response.get('hits', {}).get('hits', [])
            
            for hit in hits:
                source = hit.get('_source', {})
                score = hit.get('_score', 0.0)
                
                # Extract highlights if available
                highlights = hit.get('highlight', {})
                
                result = {
                    'document_id': source.get('document_id'),
                    'chunk_id': source.get('chunk_id'),
                    'chunk_index': source.get('chunk_index', 0),
                    'title': source.get('title', ''),
                    'content': source.get('content', ''),
                    'score': score,
                    'method': 'sparse',
                    'language': source.get('language', 'unknown'),
                    'keywords': source.get('keywords', []),
                    'entities': source.get('entities', []),
                    'tags': source.get('tags', []),
                    'highlights': highlights
                }
                
                # Calculate additional relevance metrics
                result['relevance_metrics'] = await self._calculate_relevance_metrics(
                    result, original_query
                )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search result processing failed: {e}")
            return []
    
    async def _calculate_relevance_metrics(self, 
                                         result: Dict[str, Any], 
                                         query: str) -> Dict[str, Any]:
        """Calculate additional relevance metrics for result."""
        try:
            content = result.get('content', '')
            title = result.get('title', '')
            
            # Query term coverage
            query_terms = await self.tokenizer.tokenize_text(query, remove_stopwords=True, stem_words=False)
            content_terms = await self.tokenizer.tokenize_text(content, remove_stopwords=True, stem_words=False)
            
            if query_terms:
                term_coverage = len(set(query_terms) & set(content_terms)) / len(set(query_terms))
            else:
                term_coverage = 0.0
            
            # Title match bonus
            title_match = 0.0
            if title and query.lower() in title.lower():
                title_match = 1.0
            elif title:
                title_terms = await self.tokenizer.tokenize_text(title, remove_stopwords=True, stem_words=False)
                if query_terms and title_terms:
                    title_match = len(set(query_terms) & set(title_terms)) / len(set(query_terms))
            
            # Entity match bonus
            entity_match = 0.0
            entities = result.get('entities', [])
            if entities:
                entity_texts = [entity.get('text', '').lower() for entity in entities]
                if any(query.lower() in entity_text for entity_text in entity_texts):
                    entity_match = 1.0
            
            # Keyword match bonus
            keyword_match = 0.0
            keywords = result.get('keywords', [])
            if keywords:
                keyword_texts = [kw.lower() for kw in keywords]
                if any(query.lower() in keyword for keyword in keyword_texts):
                    keyword_match = 1.0
            
            return {
                'term_coverage': term_coverage,
                'title_match': title_match,
                'entity_match': entity_match,
                'keyword_match': keyword_match,
                'content_length': len(content),
                'query_length': len(query)
            }
            
        except Exception as e:
            logger.error(f"Relevance metrics calculation failed: {e}")
            return {}
    
    async def _detect_language(self, text: str) -> str:
        """Detect language of text."""
        try:
            if not text:
                return 'english'
            
            # Simple language detection based on character sets
            cyrillic_count = len([c for c in text if '\u0400' <= c <= '\u04FF'])
            latin_count = len([c for c in text if c.isalpha() and c.isascii()])
            
            if cyrillic_count > latin_count:
                return 'russian'
            else:
                return 'english'
                
        except Exception:
            return 'english'
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document from the index."""
        try:
            # Delete by query
            response = await self.es_client.delete_by_query(
                index=self.index_name,
                body={
                    "query": {
                        "term": {"document_id": document_id}
                    }
                }
            )
            
            deleted_count = response.get('deleted', 0)
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Document deletion failed for {document_id}: {e}")
            return False
    
    async def update_synonyms(self, 
                            language: str, 
                            new_synonyms: List[str]) -> bool:
        """Update synonym dictionary for a language."""
        try:
            if language not in self.synonyms:
                logger.error(f"Unsupported language: {language}")
                return False
            
            # Update in-memory synonyms
            self.synonyms[language].extend(new_synonyms)
            
            # Close index for settings update
            await self.es_client.indices.close(index=self.index_name)
            
            # Update index settings
            filter_name = f"{language}_synonyms"
            await self.es_client.indices.put_settings(
                index=self.index_name,
                body={
                    "analysis": {
                        "filter": {
                            filter_name: {
                                "type": "synonym",
                                "synonyms": self.synonyms[language]
                            }
                        }
                    }
                }
            )
            
            # Reopen index
            await self.es_client.indices.open(index=self.index_name)
            
            logger.info(f"Updated {language} synonyms")
            return True
            
        except Exception as e:
            logger.error(f"Synonym update failed: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            # Get index stats
            stats_response = await self.es_client.indices.stats(index=self.index_name)
            index_stats = stats_response.get('indices', {}).get(self.index_name, {})
            
            # Get document count
            count_response = await self.es_client.count(index=self.index_name)
            doc_count = count_response.get('count', 0)
            
            # Get mapping info
            mapping_response = await self.es_client.indices.get_mapping(index=self.index_name)
            mapping_info = mapping_response.get(self.index_name, {})
            
            return {
                'index_name': self.index_name,
                'document_count': doc_count,
                'index_size': index_stats.get('total', {}).get('store', {}).get('size_in_bytes', 0),
                'shard_count': index_stats.get('total', {}).get('segments', {}).get('count', 0),
                'mapping_fields': len(mapping_info.get('mappings', {}).get('properties', {})),
                'synonyms': {lang: len(syns) for lang, syns in self.synonyms.items()}
            }
            
        except Exception as e:
            logger.error(f"Index stats retrieval failed: {e}")
            return {'error': str(e)}
    
    async def analyze_query(self, 
                          query: str, 
                          analyzer: str = "multilingual_analyzer") -> Dict[str, Any]:
        """Analyze query using specified analyzer."""
        try:
            response = await self.es_client.indices.analyze(
                index=self.index_name,
                body={
                    "analyzer": analyzer,
                    "text": query
                }
            )
            
            tokens = [token['token'] for token in response.get('tokens', [])]
            
            return {
                'original_query': query,
                'analyzer': analyzer,
                'tokens': tokens,
                'token_count': len(tokens)
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {'error': str(e)}
    
    async def suggest_completions(self, 
                                prefix: str, 
                                field: str = "content",
                                size: int = 5) -> List[str]:
        """Get query completion suggestions."""
        try:
            response = await self.es_client.search(
                index=self.index_name,
                body={
                    "suggest": {
                        "completion_suggest": {
                            "prefix": prefix,
                            "completion": {
                                "field": field,
                                "size": size
                            }
                        }
                    }
                }
            )
            
            suggestions = []
            suggest_results = response.get('suggest', {}).get('completion_suggest', [])
            
            for suggest_result in suggest_results:
                for option in suggest_result.get('options', []):
                    suggestions.append(option.get('text', ''))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Completion suggestion failed: {e}")
            return []
    
    async def get_retriever_stats(self) -> Dict[str, Any]:
        """Get sparse retriever statistics."""
        try:
            index_stats = await self.get_index_stats()
            
            return {
                'index_name': self.index_name,
                'max_results': self.max_results,
                'min_score': self.min_score,
                'tokenizer_initialized': self.tokenizer._initialized if hasattr(self.tokenizer, '_initialized') else False,
                'term_weighting_initialized': self.term_weighting._initialized if hasattr(self.term_weighting, '_initialized') else False,
                'elasticsearch_available': self.es_client is not None,
                'index_stats': index_stats
            }
            
        except Exception as e:
            logger.error(f"Retriever stats retrieval failed: {e}")
            return {'error': str(e)}


# Global instance
sparse_retriever = SparseRetriever()