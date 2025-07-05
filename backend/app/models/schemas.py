from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Literal
from .document import DocumentStatus

class DocumentBase(BaseModel):
    file_name: str
    content_type: str | None = None

class DocumentResponse(DocumentBase):
    id: UUID
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime | None = None
    
    # Метаданные документа
    file_size_bytes: int | None = None
    content_length: int | None = None
    chunk_count: int | None = None
    processing_time_seconds: float | None = None

    class Config:
        from_attributes = True

# Модель для результата поиска с информацией об источнике
class SearchResult(BaseModel):
    content: str = ""  # Контент чанка (опционально для уменьшения размера ответа)
    document_id: UUID
    document_name: str
    confidence_score: float | None = None

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    query: str
    document_ids: list[UUID] = []
    history: list[ChatMessage] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchResult] = []
    history: list[ChatMessage] = Field(default_factory=list) 