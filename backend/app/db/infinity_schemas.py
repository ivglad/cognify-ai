"""
Infinity DB collection schemas for vector storage.
"""
from typing import Dict, Any

from app.core.config import settings

# Yandex embeddings dimension
YANDEX_EMBEDDING_DIMENSION = 256

# Main chunks collection with embeddings
CHUNKS_COLLECTION_SCHEMA = {
    "chunk_id": {"type": "varchar", "primary": True},
    "document_id": {"type": "varchar", "not_null": True},
    "content": {"type": "varchar", "not_null": True},
    "embedding": {"type": f"vector,{YANDEX_EMBEDDING_DIMENSION},float"},
    
    # Metadata for filtering
    "chunk_type": {"type": "varchar"},
    "chunk_strategy": {"type": "varchar"},
    "page_number": {"type": "integer"},
    "token_count": {"type": "integer"},
    
    # Enrichment data for search
    "keywords": {"type": "varchar"},  # JSON array as string
    "questions": {"type": "varchar"},  # JSON array as string
    "tags": {"type": "varchar"},  # JSON array as string
    
    # Timestamps
    "created_at": {"type": "timestamp", "default": "now()"}
}

# Entities collection for graph search
ENTITIES_COLLECTION_SCHEMA = {
    "entity_id": {"type": "varchar", "primary": True},
    "document_id": {"type": "varchar", "not_null": True},
    "entity_name": {"type": "varchar", "not_null": True},
    "entity_type": {"type": "varchar"},
    "description": {"type": "varchar"},
    "embedding": {"type": f"vector,{YANDEX_EMBEDDING_DIMENSION},float"},  # Entity name embedding
    "confidence": {"type": "float"},
    "created_at": {"type": "timestamp", "default": "now()"}
}

# Relations collection for relationship search
RELATIONS_COLLECTION_SCHEMA = {
    "relation_id": {"type": "varchar", "primary": True},
    "document_id": {"type": "varchar", "not_null": True},
    "source_entity": {"type": "varchar", "not_null": True},
    "target_entity": {"type": "varchar", "not_null": True},
    "relation_type": {"type": "varchar"},
    "description": {"type": "varchar"},
    "confidence": {"type": "float"},
    "created_at": {"type": "timestamp", "default": "now()"}
}

# RAPTOR summaries collection
SUMMARIES_COLLECTION_SCHEMA = {
    "summary_id": {"type": "varchar", "primary": True},
    "document_id": {"type": "varchar", "not_null": True},
    "level": {"type": "integer"},  # RAPTOR hierarchy level
    "content": {"type": "varchar", "not_null": True},
    "embedding": {"type": f"vector,{YANDEX_EMBEDDING_DIMENSION},float"},
    "child_chunk_ids": {"type": "varchar"},  # JSON array of child chunk IDs
    "cluster_info": {"type": "varchar"},  # JSON cluster metadata
    "created_at": {"type": "timestamp", "default": "now()"}
}

# Collection names
COLLECTION_NAMES = {
    "chunks": "chunks_collection",
    "entities": "entities_collection", 
    "relations": "relations_collection",
    "summaries": "summaries_collection"
}

# All schemas mapping
COLLECTION_SCHEMAS = {
    COLLECTION_NAMES["chunks"]: CHUNKS_COLLECTION_SCHEMA,
    COLLECTION_NAMES["entities"]: ENTITIES_COLLECTION_SCHEMA,
    COLLECTION_NAMES["relations"]: RELATIONS_COLLECTION_SCHEMA,
    COLLECTION_NAMES["summaries"]: SUMMARIES_COLLECTION_SCHEMA
}


async def create_infinity_collections(infinity_client):
    """
    Create all required Infinity DB collections.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        db = await infinity_client.get_database(settings.INFINITY_DATABASE)
        
        for collection_name, schema in COLLECTION_SCHEMAS.items():
            try:
                # Try to create collection
                await infinity_client.create_table(
                    settings.INFINITY_DATABASE,
                    collection_name,
                    schema
                )
                logger.info(f"Created Infinity collection: {collection_name}")
            except Exception as e:
                # Collection might already exist
                logger.debug(f"Collection {collection_name} might already exist: {e}")
        
        logger.info("Infinity DB collections initialization completed")
        
    except Exception as e:
        logger.error(f"Failed to create Infinity collections: {e}")
        raise