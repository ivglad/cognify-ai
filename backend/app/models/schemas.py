from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import List, Literal
from .document import DocumentStatus

class DocumentBase(BaseModel):
    file_name: str
    content_type: str | None = None

class DocumentResponse(DocumentBase):
    id: UUID
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    query: str
    document_ids: list[UUID] = []
    history: List[ChatMessage] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    sources: list[DocumentResponse] = []
    history: List[ChatMessage] = Field(default_factory=list) 