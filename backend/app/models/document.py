"""
SQLAlchemy models for document management.
"""
import enum
from datetime import datetime
from typing import List
from uuid import uuid4

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, 
    ForeignKey, Enum, JSON, func, Index
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship

from app.db.session import Base


class DocumentStatus(str, enum.Enum):
    """Document processing status."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ChunkType(str, enum.Enum):
    """Chunk type classification."""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    SUMMARY = "summary"


class ProcessingStrategy(str, enum.Enum):
    """Document processing strategies."""
    NAIVE = "naive"
    HIERARCHICAL = "hierarchical"
    QA = "qa"
    TABLE = "table"


class Document(Base):
    """
    Core document table with processing metadata.
    """
    __tablename__ = "documents"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # File information
    file_name = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    status = Column(Enum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING)
    
    # Processing configuration
    processing_strategy = Column(String(50), default="hierarchical")
    deepdoc_enabled = Column(Boolean, default=True)
    raptor_enabled = Column(Boolean, default=False)
    kg_extraction_enabled = Column(Boolean, default=True)
    
    # Processing results
    content_length = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0)
    entities_count = Column(Integer, default=0)
    relations_count = Column(Integer, default=0)
    processing_time_seconds = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    kg_entities = relationship("KGEntity", back_populates="document", cascade="all, delete-orphan")
    kg_relations = relationship("KGRelation", back_populates="document", cascade="all, delete-orphan")
    kg_communities = relationship("KGCommunity", back_populates="document", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_status', 'status'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_file_name', 'file_name'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_name='{self.file_name}', status='{self.status}')>"


class Chunk(Base):
    """
    Document chunks with metadata and hierarchy support.
    """
    __tablename__ = "chunks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=True)  # SHA-256 for deduplication
    token_count = Column(Integer, nullable=True)
    
    # Chunk metadata
    chunk_type = Column(Enum(ChunkType), default=ChunkType.TEXT)
    chunk_strategy = Column(String(50), nullable=True)
    chunk_index = Column(Integer, nullable=True)  # Order within document
    
    # Position information
    page_number = Column(Integer, nullable=True)
    position_info = Column(JSON, nullable=True)  # Coordinates, bounding boxes, etc.
    
    # RAPTOR hierarchy
    parent_chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id"), nullable=True)
    raptor_level = Column(Integer, default=0)
    cluster_id = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    parent_chunk = relationship("Chunk", remote_side=[id])
    enrichment = relationship("ChunkEnrichment", back_populates="chunk", uselist=False, cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_chunks_document_id', 'document_id'),
        Index('idx_chunks_type', 'chunk_type'),
        Index('idx_chunks_strategy', 'chunk_strategy'),
        Index('idx_chunks_parent', 'parent_chunk_id'),
        Index('idx_chunks_content_hash', 'content_hash'),
    )
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, type='{self.chunk_type}')>"


class ChunkEnrichment(Base):
    """
    Chunk enrichment data from LLM processing.
    """
    __tablename__ = "chunk_enrichments"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False)
    
    # Enrichment data
    keywords = Column(ARRAY(String), default=list)
    generated_questions = Column(ARRAY(String), default=list)
    tags = Column(ARRAY(String), default=list)
    
    # Confidence scores
    keyword_confidence = Column(Float, nullable=True)
    question_confidence = Column(Float, nullable=True)
    tag_confidence = Column(Float, nullable=True)
    
    # Processing metadata
    enrichment_model = Column(String(100), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    chunk = relationship("Chunk", back_populates="enrichment")
    
    # Indexes
    __table_args__ = (
        Index('idx_enrichments_chunk_id', 'chunk_id'),
    )
    
    def __repr__(self):
        return f"<ChunkEnrichment(id={self.id}, chunk_id={self.chunk_id})>"


class KGEntity(Base):
    """
    Knowledge graph entities.
    """
    __tablename__ = "kg_entities"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Entity data
    entity_name = Column(String(255), nullable=False)
    entity_type = Column(String(50), nullable=True)  # PERSON, ORGANIZATION, LOCATION, etc.
    description = Column(Text, nullable=True)
    
    # Confidence and source
    confidence = Column(Float, nullable=True)
    source_chunk_ids = Column(ARRAY(UUID), default=list)
    
    # Normalization
    normalized_name = Column(String(255), nullable=True)  # For entity resolution
    canonical_id = Column(UUID(as_uuid=True), nullable=True)  # Points to canonical entity
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="kg_entities")
    source_relations = relationship("KGRelation", foreign_keys="KGRelation.source_entity_id", back_populates="source_entity")
    target_relations = relationship("KGRelation", foreign_keys="KGRelation.target_entity_id", back_populates="target_entity")
    
    # Indexes
    __table_args__ = (
        Index('idx_entities_document_id', 'document_id'),
        Index('idx_entities_type', 'entity_type'),
        Index('idx_entities_name', 'entity_name'),
        Index('idx_entities_normalized_name', 'normalized_name'),
    )
    
    def __repr__(self):
        return f"<KGEntity(id={self.id}, name='{self.entity_name}', type='{self.entity_type}')>"


class KGRelation(Base):
    """
    Knowledge graph relations.
    """
    __tablename__ = "kg_relations"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Relation data
    source_entity_id = Column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(UUID(as_uuid=True), ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    
    # Confidence and source
    confidence = Column(Float, nullable=True)
    source_chunk_ids = Column(ARRAY(UUID), default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="kg_relations")
    source_entity = relationship("KGEntity", foreign_keys=[source_entity_id], back_populates="source_relations")
    target_entity = relationship("KGEntity", foreign_keys=[target_entity_id], back_populates="target_relations")
    
    # Indexes
    __table_args__ = (
        Index('idx_relations_document_id', 'document_id'),
        Index('idx_relations_source', 'source_entity_id'),
        Index('idx_relations_target', 'target_entity_id'),
        Index('idx_relations_type', 'relation_type'),
    )
    
    def __repr__(self):
        return f"<KGRelation(id={self.id}, type='{self.relation_type}')>"


class KGCommunity(Base):
    """
    Knowledge graph communities from Leiden algorithm.
    """
    __tablename__ = "kg_communities"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Community data
    community_id = Column(String(50), nullable=False)
    entity_ids = Column(ARRAY(UUID), default=list)
    summary = Column(Text, nullable=True)
    
    # Community metrics
    entity_count = Column(Integer, nullable=True)
    relation_count = Column(Integer, nullable=True)
    coherence_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="kg_communities")
    
    # Indexes
    __table_args__ = (
        Index('idx_communities_document_id', 'document_id'),
        Index('idx_communities_community_id', 'community_id'),
    )
    
    def __repr__(self):
        return f"<KGCommunity(id={self.id}, community_id='{self.community_id}')>"


class ProcessingJob(Base):
    """
    Processing jobs for async operations with trio.
    """
    __tablename__ = "processing_jobs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Job details
    job_type = Column(String(50), nullable=False)  # 'ingestion', 'reprocessing', 'kg_extraction'
    status = Column(Enum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING)
    priority = Column(Integer, default=0)
    
    # Configuration
    job_config = Column(JSON, nullable=True)
    
    # Progress tracking
    progress_percentage = Column(Integer, default=0)
    current_step = Column(String(100), nullable=True)
    steps_completed = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=True)
    
    # Results and errors
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="processing_jobs")
    
    # Indexes
    __table_args__ = (
        Index('idx_jobs_status', 'status'),
        Index('idx_jobs_type', 'job_type'),
        Index('idx_jobs_document_id', 'document_id'),
        Index('idx_jobs_priority', 'priority'),
    )
    
    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, type='{self.job_type}', status='{self.status}')>"