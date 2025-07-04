from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import aiofiles
from pathlib import Path
import uuid
from typing import List

from app.models.schemas import DocumentResponse
from app.models.document import Document
from app.db.session import SessionLocal
from app.services.ingestion_service import ingestion_service

router = APIRouter()

# Define a temporary directory
TEMP_DIR = Path("/app/tmp")
TEMP_DIR.mkdir(exist_ok=True) # Ensure the directory exists

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[DocumentResponse])
async def read_documents():
    """
    Retrieves a list of all uploaded documents.
    """
    try:
        documents = ingestion_service.get_documents()
        return documents
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/")
async def delete_data(file_name: str | None = Query(default=None, description="File name to delete. If not provided, all data will be deleted.")):
    """
    Deletes data for a specific file or for all files if no name is provided.
    """
    try:
        result = ingestion_service.delete_data(file_name=file_name)
        return result
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document, save it temporarily, and schedule for processing.
    """
    try:
        # Save file temporarily
        temp_file_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
        async with aiofiles.open(temp_file_path, 'wb') as f:
            await f.write(await file.read())

        # Create a new document record in the database
        new_document = Document(
            file_name=file.filename,
            content_type=file.content_type,
        )
        db.add(new_document)
        db.commit()
        db.refresh(new_document)

        # Add the processing task to the background with the file path
        background_tasks.add_task(
            ingestion_service.process_file, 
            temp_file_path=str(temp_file_path),
            original_filename=file.filename,
            content_type=file.content_type,
            document_id=str(new_document.id)
        )

        return new_document
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create document record: {e}")

def process_document(file_name: str):
    # This function is now part of the IngestionService
    pass 