"""
Elasticsearch client with trio support for full-text search.
"""
import logging
from typing import Dict, List, Optional, Any
import threading

import trio
from elasticsearch import AsyncElasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from app.core.config import settings

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Elasticsearch client with trio compatibility and Russian/English analyzers.
    """
    _instance: Optional['ElasticsearchClient'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.client: Optional[AsyncElasticsearch] = None
            self.initialized = False
            self._connection_lock = trio.Lock()
    
    async def connect(self):
        """
        Connect to Elasticsearch with trio support.
        """
        async with self._connection_lock:
            if self.client is None:
                try:
                    # Create Elasticsearch client
                    self.client = AsyncElasticsearch(
                        hosts=[{
                            'host': settings.ELASTICSEARCH_HOST,
                            'port': settings.ELASTICSEARCH_PORT,
                            'scheme': 'http'
                        }],
                        timeout=30,
                        max_retries=3,
                        retry_on_timeout=True
                    )
                    
                    # Test connection
                    await self.client.ping()
                    
                    logger.info(f"Connected to Elasticsearch at {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT}")
                    self.initialized = True
                    
                    # Create index if it doesn't exist
                    await self._ensure_index_exists()
                    
                except Exception as e:
                    logger.error(f"Failed to connect to Elasticsearch: {e}")
                    raise
    
    async def disconnect(self):
        """
        Disconnect from Elasticsearch.
        """
        async with self._connection_lock:
            if self.client:
                try:
                    await self.client.close()
                    self.client = None
                    self.initialized = False
                    logger.info("Disconnected from Elasticsearch")
                except Exception as e:
                    logger.error(f"Error disconnecting from Elasticsearch: {e}")
    
    async def _ensure_index_exists(self):
        """
        Ensure the main index exists with proper mappings and analyzers.
        """
        index_name = settings.ELASTICSEARCH_INDEX
        
        try:
            # Check if index exists
            exists = await self.client.indices.exists(index=index_name)
            
            if not exists:
                # Create index with custom analyzers
                index_config = {
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "russian_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase",
                                        "russian_stop",
                                        "russian_keywords",
                                        "russian_stemmer"
                                    ]
                                },
                                "english_analyzer": {
                                    "type": "custom", 
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase",
                                        "english_stop",
                                        "english_keywords",
                                        "english_stemmer"
                                    ]
                                },
                                "multilingual_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase",
                                        "asciifolding",
                                        "multilingual_stop",
                                        "multilingual_stemmer"
                                    ]
                                }
                            },
                            "filter": {
                                "russian_stop": {
                                    "type": "stop",
                                    "stopwords": "_russian_"
                                },
                                "russian_keywords": {
                                    "type": "keyword_marker",
                                    "keywords": []
                                },
                                "russian_stemmer": {
                                    "type": "stemmer",
                                    "language": "russian"
                                },
                                "english_stop": {
                                    "type": "stop",
                                    "stopwords": "_english_"
                                },
                                "english_keywords": {
                                    "type": "keyword_marker",
                                    "keywords": []
                                },
                                "english_stemmer": {
                                    "type": "stemmer",
                                    "language": "english"
                                },
                                "multilingual_stop": {
                                    "type": "stop",
                                    "stopwords": ["the", "a", "an", "и", "в", "на", "с", "по", "для", "от", "до", "из", "к", "о"]
                                },
                                "multilingual_stemmer": {
                                    "type": "stemmer",
                                    "language": "english"
                                }
                            }
                        },
                        "number_of_shards": 1,
                        "number_of_replicas": 0
                    },
                    "mappings": {
                        "properties": {
                            "document_id": {
                                "type": "keyword"
                            },
                            "chunk_id": {
                                "type": "keyword"
                            },
                            "content": {
                                "type": "text",
                                "analyzer": "multilingual_analyzer",
                                "fields": {
                                    "russian": {
                                        "type": "text",
                                        "analyzer": "russian_analyzer"
                                    },
                                    "english": {
                                        "type": "text", 
                                        "analyzer": "english_analyzer"
                                    },
                                    "exact": {
                                        "type": "keyword"
                                    }
                                }
                            },
                            "keywords": {
                                "type": "text",
                                "analyzer": "multilingual_analyzer"
                            },
                            "questions": {
                                "type": "text",
                                "analyzer": "multilingual_analyzer"
                            },
                            "tags": {
                                "type": "keyword"
                            },
                            "chunk_type": {
                                "type": "keyword"
                            },
                            "chunk_strategy": {
                                "type": "keyword"
                            },
                            "page_number": {
                                "type": "integer"
                            },
                            "token_count": {
                                "type": "integer"
                            },
                            "document_name": {
                                "type": "text",
                                "analyzer": "multilingual_analyzer",
                                "fields": {
                                    "exact": {
                                        "type": "keyword"
                                    }
                                }
                            },
                            "created_at": {
                                "type": "date"
                            }
                        }
                    }
                }
                
                await self.client.indices.create(index=index_name, body=index_config)
                logger.info(f"Created Elasticsearch index: {index_name}")
            else:
                logger.info(f"Elasticsearch index already exists: {index_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise
    
    async def index_chunk(self, chunk_data: Dict[str, Any]) -> bool:
        """
        Index a document chunk.
        
        Args:
            chunk_data: Chunk data to index
            
        Returns:
            True if successful
        """
        if not self.initialized:
            await self.connect()
        
        try:
            index_name = settings.ELASTICSEARCH_INDEX
            
            response = await self.client.index(
                index=index_name,
                id=chunk_data.get('chunk_id'),
                body=chunk_data
            )
            
            return response.get('result') in ['created', 'updated']
            
        except Exception as e:
            logger.error(f"Failed to index chunk: {e}")
            return False
    
    async def bulk_index_chunks(self, chunks_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bulk index multiple chunks.
        
        Args:
            chunks_data: List of chunk data to index
            
        Returns:
            Bulk operation results
        """
        if not self.initialized:
            await self.connect()
        
        if not chunks_data:
            return {"indexed": 0, "errors": []}
        
        try:
            index_name = settings.ELASTICSEARCH_INDEX
            
            # Prepare bulk operations
            bulk_body = []
            for chunk_data in chunks_data:
                bulk_body.append({
                    "index": {
                        "_index": index_name,
                        "_id": chunk_data.get('chunk_id')
                    }
                })
                bulk_body.append(chunk_data)
            
            # Execute bulk operation
            response = await self.client.bulk(body=bulk_body)
            
            # Process results
            indexed = 0
            errors = []
            
            for item in response.get('items', []):
                if 'index' in item:
                    if item['index'].get('status') in [200, 201]:
                        indexed += 1
                    else:
                        errors.append(item['index'].get('error', 'Unknown error'))
            
            logger.info(f"Bulk indexed {indexed} chunks, {len(errors)} errors")
            
            return {
                "indexed": indexed,
                "errors": errors,
                "took": response.get('took', 0)
            }
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return {"indexed": 0, "errors": [str(e)]}
    
    async def search(self, query: str, document_ids: List[str] = None, 
                    filters: Dict[str, Any] = None, size: int = 20) -> Dict[str, Any]:
        """
        Search for chunks using full-text search.
        
        Args:
            query: Search query
            document_ids: Filter by document IDs
            filters: Additional filters
            size: Number of results to return
            
        Returns:
            Search results
        """
        if not self.initialized:
            await self.connect()
        
        try:
            index_name = settings.ELASTICSEARCH_INDEX
            
            # Build search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "content^2",
                                        "content.russian^1.5",
                                        "content.english^1.5",
                                        "keywords^1.2",
                                        "questions^1.1",
                                        "document_name"
                                    ],
                                    "type": "best_fields",
                                    "fuzziness": "AUTO"
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "highlight": {
                    "fields": {
                        "content": {
                            "fragment_size": 150,
                            "number_of_fragments": 3
                        }
                    }
                },
                "size": size,
                "sort": [
                    {"_score": {"order": "desc"}}
                ]
            }
            
            # Add document ID filter
            if document_ids:
                search_body["query"]["bool"]["filter"].append({
                    "terms": {"document_id": document_ids}
                })
            
            # Add additional filters
            if filters:
                for field, value in filters.items():
                    if isinstance(value, list):
                        search_body["query"]["bool"]["filter"].append({
                            "terms": {field: value}
                        })
                    else:
                        search_body["query"]["bool"]["filter"].append({
                            "term": {field: value}
                        })
            
            # Execute search
            response = await self.client.search(
                index=index_name,
                body=search_body
            )
            
            # Process results
            hits = response.get('hits', {})
            results = []
            
            for hit in hits.get('hits', []):
                source = hit['_source']
                result = {
                    "chunk_id": source.get('chunk_id'),
                    "document_id": source.get('document_id'),
                    "content": source.get('content'),
                    "score": hit['_score'],
                    "highlights": hit.get('highlight', {}),
                    "chunk_type": source.get('chunk_type'),
                    "page_number": source.get('page_number'),
                    "document_name": source.get('document_name')
                }
                results.append(result)
            
            return {
                "results": results,
                "total": hits.get('total', {}).get('value', 0),
                "max_score": hits.get('max_score', 0),
                "took": response.get('took', 0)
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"results": [], "total": 0, "max_score": 0, "took": 0}
    
    async def delete_document_chunks(self, document_id: str) -> bool:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID to delete chunks for
            
        Returns:
            True if successful
        """
        if not self.initialized:
            await self.connect()
        
        try:
            index_name = settings.ELASTICSEARCH_INDEX
            
            # Delete by query
            response = await self.client.delete_by_query(
                index=index_name,
                body={
                    "query": {
                        "term": {"document_id": document_id}
                    }
                }
            )
            
            deleted = response.get('deleted', 0)
            logger.info(f"Deleted {deleted} chunks for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Elasticsearch health.
        """
        try:
            if not self.initialized:
                await self.connect()
            
            # Get cluster health
            health = await self.client.cluster.health()
            
            # Get index stats
            index_name = settings.ELASTICSEARCH_INDEX
            stats = await self.client.indices.stats(index=index_name)
            
            return {
                "status": "healthy",
                "cluster_status": health.get('status'),
                "number_of_nodes": health.get('number_of_nodes'),
                "active_shards": health.get('active_shards'),
                "index_name": index_name,
                "document_count": stats.get('_all', {}).get('total', {}).get('docs', {}).get('count', 0),
                "index_size": stats.get('_all', {}).get('total', {}).get('store', {}).get('size_in_bytes', 0)
            }
            
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global instance
elasticsearch_client = ElasticsearchClient()