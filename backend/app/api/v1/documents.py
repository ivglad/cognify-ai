from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import aiofiles
from pathlib import Path
import uuid
import asyncio

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

@router.get("/", response_model=list[DocumentResponse])
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

@router.post("/upload", response_model=list[DocumentResponse], status_code=201)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """
    Универсальный эндпоинт для загрузки от 1 до 5 документов с параллельной обработкой.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 5 files per upload.")
    
    uploaded_documents = []
    temp_file_paths = []
    
    try:
        # Сохраняем все файлы параллельно
        async def save_file(file: UploadFile):
            temp_file_path = TEMP_DIR / f"{uuid.uuid4()}_{file.filename}"
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(await file.read())
            return temp_file_path, file.filename, file.content_type
        
        # Параллельное сохранение файлов
        file_data = await asyncio.gather(*[save_file(file) for file in files])
        temp_file_paths = [path for path, _, _ in file_data]
        
        # Создаем записи в базе данных для всех документов
        for temp_file_path, filename, content_type in file_data:
            new_document = Document(
                file_name=filename,
                content_type=content_type,
            )
            db.add(new_document)
            uploaded_documents.append((new_document, temp_file_path, filename, content_type))
        
        db.commit()
        
        # Обновляем все документы с их ID
        for new_document, _, _, _ in uploaded_documents:
            db.refresh(new_document)
        
        # Определяем стратегию обработки в зависимости от количества файлов
        if len(files) == 1:
            # Одиночный файл - запускаем обработку напрямую
            doc, temp_path, filename, content_type = uploaded_documents[0]
            background_tasks.add_task(
                ingestion_service.process_file,
                temp_file_path=str(temp_path),
                original_filename=filename,
                content_type=content_type,
                document_id=str(doc.id)
            )
        else:
            # Несколько файлов - запускаем параллельную обработку
            background_tasks.add_task(
                _process_multiple_documents,
                [(str(doc.id), str(temp_path), filename, content_type) 
                 for doc, temp_path, filename, content_type in uploaded_documents]
            )
        
        return [doc for doc, _, _, _ in uploaded_documents]
        
    except Exception as e:
        db.rollback()
        # Очистка временных файлов в случае ошибки
        for temp_path in temp_file_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки документов: {e}")

async def _process_multiple_documents(document_data: list[tuple]):
    """
    Обрабатывает несколько документов параллельно с учетом лимитов API.
    """
    max_concurrent = 3  # Максимум 3 документа одновременно для избежания перегрузки API
    
    async def process_single_document(doc_id: str, temp_path: str, filename: str, content_type: str):
        try:
            await ingestion_service.process_file(
                temp_file_path=temp_path,
                original_filename=filename,
                content_type=content_type,
                document_id=doc_id
            )
        except Exception as e:
            print(f"Ошибка обработки документа {filename} (ID: {doc_id}): {e}")
    
    # Группируем документы в батчи для параллельной обработки
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(doc_data):
        async with semaphore:
            await process_single_document(*doc_data)
    
    # Запускаем обработку всех документов с ограничением concurrency
    await asyncio.gather(*[process_with_semaphore(doc_data) for doc_data in document_data])

def process_document(file_name: str):
    # This function is now part of the IngestionService
    pass 