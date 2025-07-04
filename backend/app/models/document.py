import uuid
from sqlalchemy import Column, String, DateTime, func, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base
import enum

class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    status = Column(SAEnum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Document(id={self.id}, file_name='{self.file_name}')>" 