"""
Document management API endpoints with trio support.
"""
import logging
from typing import List, Optional
from uuid import UUID

import trio
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    processing_options: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload multiple documents for processing.
    
    Processing Options (JSON string):
    - deepdoc_enabled: bool = True
    - chunking_strategy: str = "hierarchical" 
    - raptor_enabled: bool = False
    - kg_extraction_enabled: bool = True
    - chunk_size: int = 1000
    - chunk_overlap: int = 200
    """
    import json
    from uuid import uuid4
    from app.db.session import SessionLocal
    from app.models.document import Document, DocumentStatus, ProcessingStrategy
    from app.core.storage import storage_manager
    
    start_time = trio.current_time()
    
    # Parse processing options
    options = {}
    if processing_options:
        try:
            options = json.loads(processing_options)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid processing_options JSON format"
            )
    
    # Validate file types
    supported_extensions = {f".{ext}" for ext in settings.SUPPORTED_FORMATS}
    
    document_ids = []
    uploaded_files = []
    
    # Process each file
    for file in files:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File must have a filename"
            )
        
        # Check file extension
        file_ext = "." + file.filename.split(".")[-1].lower()
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {list(supported_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Check file size
        if file_size > settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File {file.filename} is too large. Maximum size: {settings.MAX_DOCUMENT_SIZE_MB}MB"
            )
        
        # Generate document ID
        document_id = uuid4()
        
        # Save file to storage
        try:
            file_path = await storage_manager.save_uploaded_file(
                file_content, file.filename, str(document_id)
            )
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save file {file.filename}"
            )
        
        # Create document record
        db = SessionLocal()
        try:
            document = Document(
                id=document_id,
                file_name=file.filename,
                content_type=file.content_type,
                file_size_bytes=file_size,
                status=DocumentStatus.PENDING,
                processing_strategy=options.get("chunking_strategy", "hierarchical"),
                deepdoc_enabled=options.get("deepdoc_enabled", True),
                raptor_enabled=options.get("raptor_enabled", False),
                kg_extraction_enabled=options.get("kg_extraction_enabled", True)
            )
            
            db.add(document)
            db.commit()
            db.refresh(document)
            
            document_ids.append(document_id)
            uploaded_files.append({
                "document_id": str(document_id),
                "filename": file.filename,
                "size_bytes": file_size,
                "file_path": file_path
            })
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create document record for {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create document record for {file.filename}"
            )
        finally:
            db.close()
    
    # TODO: Add documents to processing queue with trio
    # For now, just log that they're ready for processing
    logger.info(f"Uploaded {len(document_ids)} documents, ready for processing")
    
    processing_time = (trio.current_time() - start_time) * 1000
    
    return {
        "document_ids": [str(doc_id) for doc_id in document_ids],
        "uploaded_files": uploaded_files,
        "message": f"Successfully uploaded {len(files)} documents",
        "processing_time_ms": processing_time,
        "status": "uploaded_pending_processing"
    }


@router.get("/")
async def list_documents(
    status: Optional[str] = None,
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc")
):
    """List documents with filtering and pagination."""
    from sqlalchemy import desc, asc
    from app.db.session import SessionLocal
    from app.models.document import Document, DocumentStatus
    
    db = SessionLocal()
    try:
        # Build query
        query = db.query(Document)
        
        # Apply status filter
        if status:
            try:
                status_enum = DocumentStatus(status.upper())
                query = query.filter(Document.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: {[s.value for s in DocumentStatus]}"
                )
        
        # Apply sorting
        sort_column = getattr(Document, sort_by, None)
        if not sort_column:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by field: {sort_by}"
            )
        
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        documents = query.offset(offset).limit(limit).all()
        
        # Convert to response format
        document_list = []
        for doc in documents:
            document_list.append({
                "id": str(doc.id),
                "file_name": doc.file_name,
                "content_type": doc.content_type,
                "status": doc.status.value,
                "file_size_bytes": doc.file_size_bytes,
                "content_length": doc.content_length,
                "chunk_count": doc.chunk_count,
                "entities_count": doc.entities_count,
                "relations_count": doc.relations_count,
                "processing_time_seconds": doc.processing_time_seconds,
                "error_message": doc.error_message,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None
            })
        
        return {
            "documents": document_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents"
        )
    finally:
        db.close()


@router.get("/{document_id}")
async def get_document(document_id: UUID):
    """Get detailed document information."""
    from app.db.session import SessionLocal
    from app.models.document import Document
    
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document with id {document_id} not found"
            )
        
        return {
            "id": str(document.id),
            "file_name": document.file_name,
            "content_type": document.content_type,
            "status": document.status.value,
            "file_size_bytes": document.file_size_bytes,
            "content_length": document.content_length,
            "chunk_count": document.chunk_count,
            "entities_count": document.entities_count,
            "relations_count": document.relations_count,
            "processing_time_seconds": document.processing_time_seconds,
            "processing_strategy": document.processing_strategy,
            "deepdoc_enabled": document.deepdoc_enabled,
            "raptor_enabled": document.raptor_enabled,
            "kg_extraction_enabled": document.kg_extraction_enabled,
            "error_message": document.error_message,
            "error_details": document.error_details,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "processed_at": document.processed_at.isoformat() if document.processed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document"
        )
    finally:
        db.close()


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: UUID,
    chunk_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0)
):
    """Get document chunks with filtering."""
    from app.db.session import SessionLocal
    from app.models.document import Document, Chunk, ChunkType, ChunkEnrichment
    from sqlalchemy.orm import joinedload
    
    db = SessionLocal()
    try:
        # Check if document exists
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document with id {document_id} not found"
            )
        
        # Build query for chunks
        query = db.query(Chunk).filter(Chunk.document_id == document_id)
        
        # Apply chunk type filter
        if chunk_type:
            try:
                chunk_type_enum = ChunkType(chunk_type.lower())
                query = query.filter(Chunk.chunk_type == chunk_type_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid chunk_type: {chunk_type}. Valid values: {[ct.value for ct in ChunkType]}"
                )
        
        # Order by chunk index
        query = query.order_by(Chunk.chunk_index.asc().nullslast(), Chunk.created_at.asc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination and load enrichment data
        chunks = query.options(joinedload(Chunk.enrichment)).offset(offset).limit(limit).all()
        
        # Convert to response format
        chunk_list = []
        for chunk in chunks:
            chunk_data = {
                "id": str(chunk.id),
                "content": chunk.content,
                "chunk_type": chunk.chunk_type.value,
                "chunk_strategy": chunk.chunk_strategy,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count,
                "position_info": chunk.position_info,
                "raptor_level": chunk.raptor_level,
                "cluster_id": chunk.cluster_id,
                "parent_chunk_id": str(chunk.parent_chunk_id) if chunk.parent_chunk_id else None,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                "keywords": [],
                "generated_questions": [],
                "tags": []
            }
            
            # Add enrichment data if available
            if chunk.enrichment:
                chunk_data.update({
                    "keywords": chunk.enrichment.keywords or [],
                    "generated_questions": chunk.enrichment.generated_questions or [],
                    "tags": chunk.enrichment.tags or [],
                    "keyword_confidence": chunk.enrichment.keyword_confidence,
                    "question_confidence": chunk.enrichment.question_confidence,
                    "tag_confidence": chunk.enrichment.tag_confidence
                })
            
            chunk_list.append(chunk_data)
        
        return {
            "document_id": str(document_id),
            "chunks": chunk_list,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document chunks"
        )
    finally:
        db.close()


@router.delete("/")
async def delete_documents(
    document_ids: Optional[List[UUID]] = Query(None),
    delete_all: bool = Query(False),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Delete specific documents or all documents."""
    from app.db.session import SessionLocal
    from app.models.document import Document
    from app.core.storage import storage_manager
    from app.db.infinity_client import infinity_client
    from app.db.infinity_schemas import COLLECTION_NAMES
    
    db = SessionLocal()
    try:
        deleted_count = 0
        deleted_document_ids = []
        
        if delete_all:
            # Delete all documents
            documents = db.query(Document).all()
            
            for document in documents:
                # Delete from file storage
                try:
                    await storage_manager.delete_document_files(str(document.id))
                except Exception as e:
                    logger.warning(f"Failed to delete files for document {document.id}: {e}")
                
                # Delete from Infinity DB collections
                try:
                    chunks_table = await infinity_client.get_table("default", COLLECTION_NAMES["chunks"])
                    chunks_table.delete(f"document_id = '{str(document.id)}'")
                    
                    entities_table = await infinity_client.get_table("default", COLLECTION_NAMES["entities"])
                    entities_table.delete(f"document_id = '{str(document.id)}'")
                    
                    relations_table = await infinity_client.get_table("default", COLLECTION_NAMES["relations"])
                    relations_table.delete(f"document_id = '{str(document.id)}'")
                    
                    summaries_table = await infinity_client.get_table("default", COLLECTION_NAMES["summaries"])
                    summaries_table.delete(f"document_id = '{str(document.id)}'")
                except Exception as e:
                    logger.warning(f"Failed to delete vector data for document {document.id}: {e}")
                
                deleted_document_ids.append(str(document.id))
            
            # Delete from PostgreSQL (cascade will handle related records)
            deleted_count = db.query(Document).delete()
            db.commit()
            
            logger.info(f"Deleted all {deleted_count} documents")
            
        elif document_ids:
            # Delete specific documents
            for doc_id in document_ids:
                document = db.query(Document).filter(Document.id == doc_id).first()
                
                if document:
                    # Delete from file storage
                    try:
                        await storage_manager.delete_document_files(str(document.id))
                    except Exception as e:
                        logger.warning(f"Failed to delete files for document {document.id}: {e}")
                    
                    # Delete from Infinity DB collections
                    try:
                        chunks_table = await infinity_client.get_table("default", COLLECTION_NAMES["chunks"])
                        chunks_table.delete(f"document_id = '{str(document.id)}'")
                        
                        entities_table = await infinity_client.get_table("default", COLLECTION_NAMES["entities"])
                        entities_table.delete(f"document_id = '{str(document.id)}'")
                        
                        relations_table = await infinity_client.get_table("default", COLLECTION_NAMES["relations"])
                        relations_table.delete(f"document_id = '{str(document.id)}'")
                        
                        summaries_table = await infinity_client.get_table("default", COLLECTION_NAMES["summaries"])
                        summaries_table.delete(f"document_id = '{str(document.id)}'")
                    except Exception as e:
                        logger.warning(f"Failed to delete vector data for document {document.id}: {e}")
                    
                    # Delete from PostgreSQL
                    db.delete(document)
                    deleted_document_ids.append(str(doc_id))
                    deleted_count += 1
                else:
                    logger.warning(f"Document {doc_id} not found for deletion")
            
            db.commit()
            logger.info(f"Deleted {deleted_count} documents")
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Either provide document_ids or set delete_all=true"
            )
        
        return {
            "deleted_document_ids": deleted_document_ids,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} documents"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete documents"
        )
    finally:
        db.close()


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: UUID,
    processing_options: dict,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Reprocess document with new options."""
    
    # TODO: Implement document reprocessing
    return {
        "document_id": str(document_id),
        "message": "Document reprocessing not implemented yet",
        "processing_options": processing_options
    }