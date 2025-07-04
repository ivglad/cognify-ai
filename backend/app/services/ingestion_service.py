from fastapi import UploadFile, HTTPException, Depends
import io
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from infinity.common import ConflictType
from infinity.index import IndexInfo, IndexType
from app.core.config import settings
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.db.infinity_client import infinity_client
import asyncio
from app.models.document import Document, DocumentStatus
import logging
from unstructured.partition.auto import partition
import time
from app.llm.factory import embedding_model, YANDEX_EMBEDDING_DIMENSION
from app.services.kg_service import kg_service

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        # Here we can initialize models, clients, etc.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        # The embedding model is now loaded from a central place
        self.embedding_model = embedding_model
        
        # Download NLTK data if not already present (useful for local dev)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading missing NLTK data...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')

        try:
            self.russian_stopwords = set(stopwords.words("russian"))
        except LookupError:
            logger.warning("Russian stopwords not available, using empty set")
            self.russian_stopwords = set()

        # Используем централизованный Infinity клиент
        logger.info("IngestionService using centralized Infinity client")

    async def process_file(self, temp_file_path: str, original_filename: str, content_type: str, document_id: str):
        """
        Main method to process a temporarily saved file.
        """
        start_time = time.monotonic()
        db = SessionLocal()
        try:
            logger.info(f"Starting ingestion for {original_filename} (ID: {document_id})...")
            self._update_document_status(db, document_id, DocumentStatus.PROCESSING)

            with open(temp_file_path, 'rb') as f:
                file_bytes = f.read()

            content = await asyncio.to_thread(self._parse_content, file_bytes, content_type, original_filename)
            chunks = await asyncio.to_thread(self._chunk_content, content)
            await self._embed_and_store(chunks, document_id=document_id)

            # After storing chunks, extract entities and build knowledge graph
            await kg_service.add_document_and_extract_entities(
                document_id=document_id,
                document_name=original_filename,
                text_chunks=chunks
            )

            self._update_document_status(db, document_id, DocumentStatus.COMPLETED)
            end_time = time.monotonic()
            logger.info(f"Finished ingestion for {original_filename} (ID: {document_id}) in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logger.error(f"Failed ingestion for {original_filename} (ID: {document_id}): {e}", exc_info=True)
            self._update_document_status(db, document_id, DocumentStatus.FAILED)
        finally:
            db.close()
            os.remove(temp_file_path) # Clean up the temporary file
            logger.info(f"Removed temporary file: {temp_file_path}")

    def get_documents(self):
        """
        Retrieves all document records from the database.
        """
        db = SessionLocal()
        try:
            documents = db.query(Document).order_by(Document.created_at.desc()).all()
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve documents.")
        finally:
            db.close()

    def delete_data(self, file_name: str | None = None):
        """
        Deletes data from the single collection for a specific file or all files.
        """
        db = SessionLocal()
        infinity_db = infinity_client.get_database("default")
        table_name = "chunks_collection"
        
        try:
            if file_name:
                logger.info(f"Attempting to delete data for file: {file_name}")
                doc_to_delete = db.query(Document).filter(Document.file_name == file_name).first()
                if not doc_to_delete:
                    raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found.")
                
                doc_id_str = str(doc_to_delete.id)
                
                # Delete from Infinity collection
                try:
                    table = infinity_db.get_table(table_name)
                    table.delete(f"document_id = '{doc_id_str}'")
                    logger.info(f"Deleted chunks for document ID '{doc_id_str}' from '{table_name}'.")
                except Exception as e:
                    logger.warning(f"Could not delete chunks for doc '{doc_id_str}' from Infinity. It might be empty. Error: {e}")

                # Delete from PostgreSQL
                db.delete(doc_to_delete)
                db.commit()
                logger.info(f"Deleted document record for '{file_name}' (ID: {doc_id_str}).")
                return {"message": f"Successfully deleted data for file '{file_name}'."}
            else:
                logger.info("Attempting to delete all data from the collection.")
                # Drop the entire collection from Infinity
                infinity_db.drop_table(table_name, ConflictType.Ignore)
                logger.info(f"Dropped collection '{table_name}' from Infinity DB.")

                # Delete all document records from PostgreSQL
                num_deleted = db.query(Document).delete()
                db.commit()
                logger.info(f"Deleted {num_deleted} document records from PostgreSQL.")
                return {"message": f"Successfully deleted all data."}

        except Exception as e:
            db.rollback()
            logger.error(f"Error during data deletion: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred during data deletion: {e}")
        finally:
            db.close()

    def _update_document_status(self, db: Session, document_id: str, status: DocumentStatus):
        db.query(Document).filter(Document.id == document_id).update({"status": status})
        db.commit()

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes a text chunk by tokenizing, removing stop words and punctuation, and converting to lower case.
        """
        try:
            # Tokenize and convert to lower case
            tokens = word_tokenize(text.lower(), language='russian')
            # Filter out punctuation and stopwords
            filtered_tokens = [
                token for token in tokens
                if token.isalnum() and token not in self.russian_stopwords
            ]
            return " ".join(filtered_tokens)
        except LookupError as e:
            logger.error(f"NLTK tokenization failed: {e}. Falling back to simple split.")
            # Fallback к простому разделению по словам
            simple_tokens = text.lower().split()
            filtered_tokens = [
                token for token in simple_tokens
                if token.isalnum() and token not in self.russian_stopwords
            ]
            return " ".join(filtered_tokens)

    def _parse_content(self, file_bytes: bytes, content_type: str, filename: str) -> str:
        """
        Parses content from file bytes using the 'unstructured' library.
        """
        logger.info(f"Parsing content from {filename} using 'unstructured'...")
        
        try:
            # 'unstructured' can work with file-like objects
            elements = partition(file=io.BytesIO(file_bytes), content_type=content_type)
            return "\\n\\n".join([str(el) for el in elements])
        except Exception as e:
            logger.error(f"Error parsing file {filename} with unstructured: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to parse file with unstructured: {e}")

    def _chunk_content(self, content: str) -> list[str]:
        """
        Splits the content into chunks using LangChain's text splitter.
        """
        logger.info("Splitting content into chunks...")
        if not content:
            return []
        chunks = self.text_splitter.split_text(content)
        logger.info(f"Content split into {len(chunks)} chunks.")
        return chunks

    async def _embed_and_store(self, chunks: list[str], document_id: str):
        """
        Embeds chunks and stores them in a single collection in Infinity DB.
        """
        if not chunks:
            logger.warning("No chunks to embed.")
            return

        logger.info(f"Embedding and storing {len(chunks)} chunks for document {document_id} into collection 'chunks'...")
        
        try:
            db = infinity_client.get_database("default")
            table_name = "chunks_collection"
            
            # Логируем текущее состояние таблиц перед созданием
            existing_tables = infinity_client.list_tables("default")
            logger.info(f"Tables in database before creation: {existing_tables}")
            
            # Create the single collection if it doesn't exist
            table_created = False
            try:
                table = db.get_table(table_name)
                logger.info(f"Table '{table_name}' already exists")
                
                # Проверяем и создаем полнотекстовый индекс если его нет
                try:
                    indexes_response = table.list_indexes()
                    index_names = getattr(indexes_response, 'index_names', [])
                    fulltext_index_exists = any("fulltext_idx" in str(idx) for idx in index_names)
                    if not fulltext_index_exists:
                        logger.info("Creating missing fulltext index...")
                        table.create_index(
                            "fulltext_idx",
                            IndexInfo(
                                "chunk_text_normalized",
                                IndexType.FullText
                            )
                        )
                        logger.info("Created fulltext index on 'chunk_text_normalized' column")
                except Exception as e:
                    logger.warning(f"Failed to check/create fulltext index: {e}")
            except Exception:
                logger.info(f"Table '{table_name}' not found, creating it...")
                table = db.create_table(
                    table_name,
                    {
                        "document_id": {"type": "varchar"},
                        "chunk_text": {"type": "varchar"},
                        "chunk_text_normalized": {"type": "varchar"},
                        "embedding": {"type": f"vector,{YANDEX_EMBEDDING_DIMENSION},double"}
                    }
                )
                table_created = True
                logger.info(f"Table '{table_name}' created successfully")
                
                # Создаем полнотекстовый индекс для поиска
                try:
                    table.create_index(
                        "fulltext_idx",
                        IndexInfo(
                            "chunk_text_normalized",
                            IndexType.FullText
                        )
                    )
                    logger.info(f"Created fulltext index on 'chunk_text_normalized' column")
                except Exception as e:
                    logger.warning(f"Failed to create fulltext index: {e}")
                    # Не прерываем выполнение, так как это не критично
            
            # Проверяем что таблица действительно создана
            if table_created:
                tables_after_creation = infinity_client.list_tables("default")
                logger.info(f"Tables in database after creation: {tables_after_creation}")
                
                # Проверяем существование конкретной таблицы
                table_exists = infinity_client.table_exists(table_name, "default")
                logger.info(f"Table '{table_name}' exists after creation: {table_exists}")
            
            # Используем асинхронный метод для создания эмбеддингов
            if hasattr(self.embedding_model, 'aembed_documents'):
                embeddings = await self.embedding_model.aembed_documents(chunks)
            else:
                # Fallback для не-Yandex провайдеров
                embeddings = await asyncio.to_thread(self.embedding_model.embed_documents, chunks)
            
            records = []
            for i, chunk in enumerate(chunks):
                # Преобразуем float в double для совместимости с Infinity
                embedding_double = [float(x) for x in embeddings[i]]
                records.append({
                    "document_id": document_id,
                    "chunk_text": chunk,
                    "chunk_text_normalized": self._normalize_text(chunk),
                    "embedding": embedding_double
                })
            
            table.insert(records)
            logger.info(f"Successfully stored chunks for document {document_id} in collection '{table_name}'")
            
            # Проверяем что данные действительно вставлены
            table_info = infinity_client.get_table_info(table_name, "default")
            logger.info(f"Table '{table_name}' info after insertion: {table_info}")
            
            # Дополнительная проверка - получаем записи для этого документа
            try:
                verification_result = table.output(['document_id', 'chunk_text']).filter(f"document_id = '{document_id}'").to_df()
                if verification_result and len(verification_result) > 0:
                    doc_chunk_count = len(verification_result[0])
                    logger.info(f"Verification: Found {doc_chunk_count} chunks for document {document_id}")
                else:
                    logger.warning(f"Verification: No chunks found for document {document_id} after insertion!")
            except Exception as e:
                logger.error(f"Failed to verify inserted data for document {document_id}: {e}")
            
            # Финальная проверка состояния базы данных
            final_tables = infinity_client.list_tables("default")
            logger.info(f"Final tables in database: {final_tables}")

        except Exception as e:
            logger.error(f"Error with Infinity DB for document {document_id}: {e}", exc_info=True)

ingestion_service = IngestionService() 