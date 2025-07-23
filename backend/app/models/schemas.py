"""
Pydantic models for RAGFlow implementation.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

# Import enums from SQLAlchemy models
from app.models.document import DocumentStatus, ChunkType, ProcessingStrategy


# Request/Response Models
class DocumentUploadRequest(BaseModel):
    """Document upload request."""
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Document upload response."""
    document_ids: List[UUID]
    message: str


class DocumentResponse(BaseModel):
    """Document information response."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    file_name: str
    content_type: Optional[str]
    status: DocumentStatus
    file_size_bytes: Optional[int]
    content_length: Optional[int]
    chunk_count: Optional[int]
    entities_count: Optional[int]
    relations_count: Optional[int]
    processing_time_seconds: Optional[float]
    error_message: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    processed_at: Optional[datetime]


class ChunkResponse(BaseModel):
    """Chunk information response."""
    id: UUID
    content: str
    chunk_type: ChunkType
    chunk_strategy: Optional[str]
    page_number: Optional[int]
    token_count: Optional[int]
    keywords: List[str] = Field(default_factory=list)
    generated_questions: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: List[UUID] = Field(default_factory=list)
    search_options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    history: List['ChatMessage'] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """Chat message model."""
    role: Literal["user", "assistant"]
    content: str


class SearchResult(BaseModel):
    """Search result model."""
    content: str
    document_id: UUID
    document_name: str
    chunk_id: UUID
    confidence_score: float
    chunk_type: ChunkType
    page_number: Optional[int]
    citations: List['Citation'] = Field(default_factory=list)


class Citation(BaseModel):
    """Citation model."""
    text: str
    source_chunk_id: UUID
    confidence: float


class SearchResponse(BaseModel):
    """Search response model."""
    answer: str
    sources: List[SearchResult] = Field(default_factory=list)
    reasoning_steps: Optional[List[str]] = Field(default_factory=list)
    processing_time_ms: int
    total_chunks_searched: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    components: Dict[str, Dict[str, Any]]
    timestamp: datetime
    version: str


# Internal Models
class DocumentStructure(BaseModel):
    """Document structure from DeepDoc processing."""
    pages: List['PageStructure']
    tables: List['Table']
    images: List['Image']
    metadata: Dict[str, Any]


class PageStructure(BaseModel):
    """Page structure information."""
    page_number: int
    layout_blocks: List['LayoutBlock']
    text_blocks: List['TextBlock']


class LayoutBlock(BaseModel):
    """Layout block information."""
    type: str  # title, text, table, figure, header, footer
    bbox: List[float]  # [x0, y0, x1, y1]
    confidence: float


class TextBlock(BaseModel):
    """Text block information."""
    text: str
    bbox: List[float]
    confidence: float
    font_info: Optional[Dict[str, Any]]


class Table(BaseModel):
    """Table information."""
    html_content: str
    markdown_content: str
    bbox: List[float]
    page_number: int
    confidence: float


class Image(BaseModel):
    """Image information."""
    image_data: bytes
    description: Optional[str]
    bbox: List[float]
    page_number: int


class Chunk(BaseModel):
    """Chunk model."""
    id: UUID
    content: str
    chunk_type: ChunkType
    chunk_strategy: str
    document_id: UUID
    page_number: Optional[int]
    position_info: Optional[Dict[str, Any]]
    token_count: int
    parent_chunk_id: Optional[UUID]
    raptor_level: int = 0


class ChunkEnrichment(BaseModel):
    """Chunk enrichment data."""
    chunk_id: UUID
    keywords: List[str]
    generated_questions: List[str]
    tags: List[str]
    keyword_confidence: float
    question_confidence: float
    tag_confidence: float


class Entity(BaseModel):
    """Knowledge graph entity."""
    id: UUID
    name: str
    type: str
    description: Optional[str]
    confidence: float
    source_chunk_ids: List[UUID]


class Relation(BaseModel):
    """Knowledge graph relation."""
    id: UUID
    source_entity_id: UUID
    target_entity_id: UUID
    relation_type: str
    description: Optional[str]
    confidence: float
    source_chunk_ids: List[UUID]


class Community(BaseModel):
    """Knowledge graph community."""
    id: str
    entity_ids: List[UUID]
    summary: str
    entity_count: int
    relation_count: int
    coherence_score: float


# Update forward references
SearchRequest.model_rebuild()
SearchResult.model_rebuild()