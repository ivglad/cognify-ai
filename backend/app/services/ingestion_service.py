from fastapi import UploadFile, HTTPException, Depends
import io
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
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
from app.services.tokenizer_service import tokenizer_service

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import spacy

logger = logging.getLogger(__name__)

class SpacyTextSplitter(TextSplitter):
    """
    Text splitter that uses Spacy to split text into sentences.
    It then groups sentences into chunks of a specified size.
    """
    def __init__(self, spacy_model_name: str, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            logger.info(f"Загрузка Spacy модели '{spacy_model_name}' для SpacyTextSplitter...")
            self.nlp = spacy.load(spacy_model_name, disable=["parser", "tagger", "ner"])
            self.nlp.add_pipe("sentencizer")
            logger.info("Spacy модель загружена для разбиения предложений.")
        except OSError:
            logger.error(f"Spacy модель '{spacy_model_name}' не найдена. Выполните 'python -m spacy download {spacy_model_name}'")
            raise

    def split_text(self, text: str) -> list[str]:
        doc = self.nlp(text)
        sents = [sent.text for sent in doc.sents]
        
        chunks = []
        current_chunk = ""
        for sent in sents:
            if len(current_chunk) + len(sent) > self.chunk_size:
                chunks.append(current_chunk.strip())
                # Handle overlap by starting the new chunk with the last part of the old one
                # This is a simple approximation of overlap
                overlap_text = current_chunk[-self.chunk_overlap:]
                current_chunk = overlap_text + sent + " "
            else:
                current_chunk += sent + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return [chunk for chunk in chunks if chunk]


class IngestionService:
    def __init__(self):
        # Here we can initialize models, clients, etc.
        # Уменьшен размер чанков для соответствия лимиту Yandex API (2048 токенов)
        self.text_splitter = SpacyTextSplitter(
            spacy_model_name=settings.SPACY_MODEL_NAME,
            chunk_size=500,
            chunk_overlap=100
        )
        # The embedding model is now loaded from a central place
        self.embedding_model = embedding_model
        
        # Download NLTK data if not already present (useful for local dev)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Загрузка отсутствующих данных NLTK...")
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')

        try:
            self.russian_stopwords = set(stopwords.words("russian"))
        except LookupError:
            logger.warning("Русские стоп-слова недоступны, используется пустой набор")
            self.russian_stopwords = set()

        # Используем централизованный Infinity клиент
        logger.info("IngestionService использует центр. клиент Infinity")

    async def process_file(self, temp_file_path: str, original_filename: str, content_type: str, document_id: str):
        """
        Main method to process a temporarily saved file with improved error handling.
        """
        start_time = time.monotonic()
        db = SessionLocal()
        processing_errors = []
        
        # Переменные для метаданных
        file_size_bytes = 0
        content_length = 0
        chunk_count = 0
        
        try:
            logger.info(f"Начало обработки {original_filename} (ID: {document_id})...")
            self._update_document_status(db, document_id, DocumentStatus.PROCESSING)

            # Этап 1: Чтение файла
            try:
                with open(temp_file_path, 'rb') as f:
                    file_bytes = f.read()
                file_size_bytes = len(file_bytes)
                logger.info(f"Прочитано {file_size_bytes} байт из {original_filename}")
            except Exception as e:
                error_msg = f"Ошибка чтения файла {original_filename}: {e}"
                logger.error(error_msg, exc_info=True)
                processing_errors.append(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            # Этап 2: Парсинг контента
            try:
                content = await asyncio.to_thread(self._parse_content, file_bytes, content_type, original_filename)
                content_length = len(content)
                logger.info(f"Контент обработан из {original_filename}, длина: {content_length} символов")
            except Exception as e:
                error_msg = f"Ошибка парсинга контента из {original_filename}: {e}"
                logger.error(error_msg, exc_info=True)
                processing_errors.append(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            # Этап 3: Разбиение на чанки
            try:
                chunks = await asyncio.to_thread(self._chunk_content, content)
                chunk_count = len(chunks)
                logger.info(f"Создано {chunk_count} чанков из {original_filename}")
            except Exception as e:
                error_msg = f"Ошибка разбиения контента для {original_filename}: {e}"
                logger.error(error_msg, exc_info=True)
                processing_errors.append(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            # Этап 4: Создание эмбеддингов и сохранение (критично)
            try:
                await self._embed_and_store(chunks, document_id=document_id)
                logger.info(f"Эмбеддинги созданы и сохранены для {original_filename}")
            except Exception as e:
                error_msg = f"Ошибка создания эмбеддингов для {original_filename}: {e}"
                logger.error(error_msg, exc_info=True)
                processing_errors.append(error_msg)
                # Критичная ошибка - прерываем обработку
                raise HTTPException(status_code=500, detail=error_msg)

            # Этап 5: Knowledge Graph (не критично - может продолжить без этого)
            try:
                # Извлекаем только текст для передачи в KG
                text_chunks_for_kg = [chunk_text for chunk_text, _ in chunks]
                
                await kg_service.add_document_and_extract_entities(
                    document_id=document_id,
                    document_name=original_filename,
                    text_chunks=text_chunks_for_kg
                )
                logger.info(f"Knowledge Graph обработан для {original_filename}")
            except Exception as e:
                error_msg = f"Ошибка Knowledge Graph для {original_filename}: {e}"
                logger.warning(error_msg, exc_info=True)
                processing_errors.append(error_msg)
                # Не критичная ошибка - продолжаем обработку
                logger.info(f"Документ {original_filename} доступен для поиска несмотря на ошибки KG")

            # Определяем финальный статус
            if processing_errors:
                # Есть ошибки, но документ обработан частично
                final_status = DocumentStatus.COMPLETED  # Или можно добавить PARTIAL_SUCCESS статус
                logger.warning(f"Документ {original_filename} обработан с {len(processing_errors)} предупреждениями: {processing_errors}")
            else:
                final_status = DocumentStatus.COMPLETED
                logger.info(f"Документ {original_filename} обработан успешно без ошибок")

            # Вычисляем время обработки
            end_time = time.monotonic()
            processing_time_seconds = end_time - start_time

            # Обновляем документ со всеми метаданными
            self._update_document_with_metadata(
                db, document_id, final_status,
                file_size_bytes, content_length, chunk_count, processing_time_seconds
            )
            
            logger.info(f"Завершена обработка {original_filename} (ID: {document_id}) за {processing_time_seconds:.2f} сек.")

        except HTTPException:
            # HTTPException уже содержат нужную информацию
            end_time = time.monotonic()
            processing_time_seconds = end_time - start_time
            self._update_document_with_metadata(
                db, document_id, DocumentStatus.FAILED,
                file_size_bytes, content_length, chunk_count, processing_time_seconds
            )
            raise
        except Exception as e:
            # Неожиданные ошибки
            error_msg = f"Неожиданная ошибка при обработке {original_filename} (ID: {document_id}): {e}"
            logger.error(error_msg, exc_info=True)
            end_time = time.monotonic()
            processing_time_seconds = end_time - start_time
            self._update_document_with_metadata(
                db, document_id, DocumentStatus.FAILED,
                file_size_bytes, content_length, chunk_count, processing_time_seconds
            )
            raise HTTPException(status_code=500, detail=error_msg)
        finally:
            db.close()
            # Очистка временного файла в любом случае
            try:
                os.remove(temp_file_path)
                logger.info(f"Удален временный файл: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Ошибка удаления временного файла {temp_file_path}: {cleanup_error}")

    def get_documents(self):
        """
        Retrieves all document records from the database.
        """
        db = SessionLocal()
        try:
            documents = db.query(Document).order_by(Document.created_at.desc()).all()
            return documents
        except Exception as e:
            logger.error(f"Ошибка получения документов: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Ошибка получения документов.")
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
                logger.info(f"Попытка удаления данных для файла: {file_name}")
                doc_to_delete = db.query(Document).filter(Document.file_name == file_name).first()
                if not doc_to_delete:
                    raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found.")
                
                doc_id_str = str(doc_to_delete.id)
                
                # Delete from Infinity collection
                try:
                    table = infinity_db.get_table(table_name)
                    table.delete(f"document_id = '{doc_id_str}'")
                    logger.info(f"Удалены чанки для документа ID '{doc_id_str}' из '{table_name}'.")
                except Exception as e:
                    logger.warning(f"Не удалось удалить чанки для документа '{doc_id_str}' из Infinity. Возможно пуст. Ошибка: {e}")

                # Delete from PostgreSQL
                db.delete(doc_to_delete)
                db.commit()
                logger.info(f"Удалена запись документа для '{file_name}' (ID: {doc_id_str}).")
                return {"message": f"Данные для файла '{file_name}' успешно удалены."}
            else:
                logger.info("Попытка удаления всех данных из коллекции.")
                # Drop the entire collection from Infinity
                infinity_db.drop_table(table_name, ConflictType.Ignore)
                logger.info(f"Удалена коллекция '{table_name}' из Infinity DB.")

                # Delete all document records from PostgreSQL
                num_deleted = db.query(Document).delete()
                db.commit()
                logger.info(f"Удалено {num_deleted} записей документов из PostgreSQL.")
                return {"message": f"Все данные успешно удалены."}

        except Exception as e:
            db.rollback()
            logger.error(f"Ошибка при удалении данных: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Произошла ошибка при удалении данных: {e}")
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
            logger.error(f"Ошибка токенизации NLTK: {e}. Переход к простому разделению.")
            # Fallback к простому разделению по словам
            simple_tokens = text.lower().split()
            filtered_tokens = [
                token for token in simple_tokens
                if token.isalnum() and token not in self.russian_stopwords
            ]
            return " ".join(filtered_tokens)

    async def _count_tokens(self, text: str) -> int:
        """
        Точный подсчет токенов с использованием Yandex API.
        Поддерживает русский и английский языки.
        """
        return await tokenizer_service.count_tokens(text)

    async def _split_large_chunk(self, text: str, max_tokens: int = 1800) -> list[str]:
        """
        Разбивает слишком большой чанк на части с точным подсчетом токенов.
        max_tokens установлен в 1800 для безопасности (оставляем запас от лимита 2048).
        """
        logger.debug(f"Начинаем разбиение чанка длиной {len(text)} символов")
        try:
            result = await tokenizer_service.split_text_by_tokens(text, max_tokens)
            logger.debug(f"Успешно разбили чанк на {len(result)} частей")
            return result
        except Exception as e:
            logger.error(f"Ошибка при разбиении чанка: {e}")
            raise

    def _parse_content(self, file_bytes: bytes, content_type: str, filename: str) -> list[tuple[str, str | None]]:
        """
        Интеллектуальный парсинг контента из файла.
        - Для Excel использует pandas для чтения листов и конвертации в Markdown.
        - Для других файлов использует 'unstructured'.
        Возвращает список кортежей (текст, имя_листа).
        """
        logger.info(f"Парсинг контента из {filename} ({content_type}) подходящим парсером...")
        
        # Список для хранения пар (текст_части, имя_листа)
        parsed_parts = []
        
        excel_types = [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel.sheet.macroEnabled.12"
        ]
        
        if content_type in excel_types or filename.endswith(('.xlsx', '.xls')):
            try:
                import pandas as pd
                # Используем BytesIO для чтения из байтов в памяти
                xls = pd.ExcelFile(io.BytesIO(file_bytes))
                
                if not xls.sheet_names:
                    logger.warning(f"Excel файл {filename} не содержит листов.")
                    return []
                    
                # Обрабатываем каждый лист
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df.empty:
                        # Конвертируем DataFrame в Markdown-таблицу
                        markdown_table = df.to_markdown(index=False)
                        # Добавляем заголовок с именем листа
                        sheet_content = f"# {sheet_name}\n\n{markdown_table}"
                        parsed_parts.append((sheet_content, sheet_name))
                
                logger.info(f"Успешно обработано {len(parsed_parts)} листов из Excel файла {filename}")
                return parsed_parts
                
            except Exception as e:
                logger.error(f"Ошибка парсинга Excel файла {filename} с pandas: {e}", exc_info=True)
                # Fallback to unstructured if pandas fails
                pass

        # Fallback или стандартный парсинг для других типов файлов
        try:
            from unstructured.partition.auto import partition
            # 'unstructured' может работать с file-like объектами
            elements = partition(file=io.BytesIO(file_bytes), content_type=content_type, strategy="hi_res")
            
            # Конвертируем HTML таблицы в Markdown для лучшего понимания LLM
            from markdownify import markdownify as md
            
            processed_elements = []
            for el in elements:
                if "text/html" in str(type(el)): # Проверяем, является ли элемент таблицей
                    try:
                        # Удаляем лишние атрибуты для чистоты
                        table_html = re.sub(r'<table\s*[^>]*>', '<table>', str(el))
                        processed_elements.append(md(table_html))
                    except Exception as md_exc:
                        logger.warning(f"Не удалось конвертировать HTML таблицу в Markdown: {md_exc}")
                        processed_elements.append(str(el)) # Fallback к обычному тексту
                else:
                    processed_elements.append(str(el))

            full_content = "\n\n".join(processed_elements)
            parsed_parts.append((full_content, None)) # Имя листа None для не-Excel файлов
            logger.info(f"Успешно обработан не-Excel файл {filename} с unstructured.")
            return parsed_parts
            
        except Exception as e:
            logger.error(f"Ошибка парсинга файла {filename} с unstructured: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to parse file with unstructured: {e}")

    def _chunk_content(self, parsed_parts: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
        """
        Разбивает контент на чанки, сохраняя привязку к имени листа.
        Принимает: список кортежей (текст_части, имя_листа)
        Возвращает: список кортежей (чанк, имя_листа)
        """
        logger.info("Разбиение контента на чанки с сохранением имен листов...")
        all_chunks = []
        if not parsed_parts:
            return []
            
        for content, sheet_name in parsed_parts:
            if not content:
                continue
            
            # Разбиваем контент части на чанки
            chunks = self.text_splitter.split_text(content)
            
            # Привязываем каждый чанк к имени листа
            for chunk in chunks:
                all_chunks.append((chunk, sheet_name))
        
        logger.info(f"Контент разбит на {len(all_chunks)} чанков.")
        return all_chunks

    async def _embed_and_store(self, chunks: list[tuple[str, str | None]], document_id: str):
        """
        Embeds chunks and stores them in a single collection in Infinity DB.
        Optimized with batch processing and rate limiting for Yandex API.
        Теперь chunks это список кортежей (chunk_text, sheet_name).
        """
        if not chunks:
            logger.warning("Нет чанков для создания эмбеддингов.")
            return

        logger.info(f"Создание эмбеддингов и сохранение {len(chunks)} чанков для документа {document_id} в коллекцию 'chunks'...")
        
        # Получаем название документа из PostgreSQL
        db_session = SessionLocal()
        try:
            document = db_session.query(Document).filter(Document.id == document_id).first()
            if document:
                document_name = document.file_name
                document_name_normalized = self._normalize_text(document_name)
                logger.info(f"Получено имя документа: '{document_name}' для документа {document_id}")
            else:
                logger.warning(f"Документ с ID {document_id} не найден в PostgreSQL, используется fallback имя")
                document_name = "Unknown Document"
                document_name_normalized = self._normalize_text(document_name)
        except Exception as e:
            logger.error(f"Ошибка получения имени документа из PostgreSQL: {e}")
            document_name = "Unknown Document"
            document_name_normalized = self._normalize_text(document_name)
        finally:
            db_session.close()
        
        try:
            db = infinity_client.get_database("default")
            table_name = "chunks_collection"
            
            # Логируем текущее состояние таблиц перед созданием
            existing_tables = infinity_client.list_tables("default")
            logger.info(f"Таблицы в БД перед созданием: {existing_tables}")
            
            # Create the single collection if it doesn't exist
            table_created = False
            try:
                table = db.get_table(table_name)
                logger.info(f"Таблица '{table_name}' уже существует")
                
                # Проверяем и создаем полнотекстовые индексы если их нет
                try:
                    indexes_response = table.list_indexes()
                    index_names = getattr(indexes_response, 'index_names', [])
                    
                    # Проверяем и создаем недостающие индексы
                    if not any("fulltext_idx" in str(idx) for idx in index_names):
                        logger.info("Создание отсутствующего полнотекстового индекса на 'chunk_text_normalized'...")
                        table.create_index("fulltext_idx", IndexInfo("chunk_text_normalized", IndexType.FullText))
                        logger.info("Создан полнотекстовый индекс на колонке 'chunk_text_normalized'")

                    if not any("document_name_fulltext_idx" in str(idx) for idx in index_names):
                        logger.info("Создание отсутствующего полнотекстового индекса имени документа...")
                        table.create_index("document_name_fulltext_idx", IndexInfo("document_name_normalized", IndexType.FullText))
                        logger.info("Создан полнотекстовый индекс на колонке 'document_name_normalized'")
                    
                    # НОВЫЙ ИНДЕКС: Проверяем и создаем индекс для названий листов
                    if not any("sheet_name_fulltext_idx" in str(idx) for idx in index_names):
                        logger.info("Создание отсутствующего полнотекстового индекса имени листа...")
                        try:
                            table.create_index(
                                "sheet_name_fulltext_idx",
                                IndexInfo(
                                    "sheet_name_normalized",
                                    IndexType.FullText
                                )
                            )
                            logger.info("Создан полнотекстовый индекс на колонке 'sheet_name_normalized'")
                        except Exception as idx_error:
                            logger.warning(f"Ошибка создания индекса имени листа (возможно таблица еще не имеет колонку): {idx_error}")

                except Exception as e:
                    logger.warning(f"Ошибка проверки/создания индексов: {e}")
            except Exception:
                logger.info(f"Таблица '{table_name}' не найдена, создаем ее...")
                table = db.create_table(
                    table_name,
                    {
                        "document_id": {"type": "varchar"},
                        "chunk_text": {"type": "varchar"},
                        "chunk_text_normalized": {"type": "varchar"},
                        "document_name": {"type": "varchar"},
                        "document_name_normalized": {"type": "varchar"},
                        "sheet_name": {"type": "varchar"},                 # НОВОЕ ПОЛЕ: Название листа
                        "sheet_name_normalized": {"type": "varchar"},      # НОВОЕ ПОЛЕ: Нормализованное название листа
                        "embedding": {"type": f"vector,{YANDEX_EMBEDDING_DIMENSION},double"}
                    }
                )
                table_created = True
                logger.info(f"Таблица '{table_name}' создана успешно")
                
                # Создаем все необходимые индексы для новой таблицы
                try:
                    # Индекс для содержимого
                    table.create_index("fulltext_idx", IndexInfo("chunk_text_normalized", IndexType.FullText))
                    logger.info(f"Создан полнотекстовый индекс на колонке 'chunk_text_normalized'")
                    
                    # Индекс для названий документов
                    table.create_index("document_name_fulltext_idx", IndexInfo("document_name_normalized", IndexType.FullText))
                    logger.info(f"Создан полнотекстовый индекс на колонке 'document_name_normalized'")

                    # НОВЫЙ ИНДЕКС: Индекс для названий листов
                    table.create_index("sheet_name_fulltext_idx", IndexInfo("sheet_name_normalized", IndexType.FullText))
                    logger.info(f"Создан полнотекстовый индекс на колонке 'sheet_name_normalized'")

                except Exception as e:
                    logger.warning(f"Ошибка создания одного или нескольких индексов для новой таблицы: {e}")
            
            # Проверяем что таблица действительно создана
            if table_created:
                tables_after_creation = infinity_client.list_tables("default")
                logger.info(f"Таблицы в БД после создания: {tables_after_creation}")
                
                # Проверяем существование конкретной таблицы
                table_exists = infinity_client.table_exists(table_name, "default")
                logger.info(f"Таблица '{table_name}' существует после создания: {table_exists}")
            
            # Батчевая обработка embeddings с учетом rate limiting
            all_embeddings = await self._create_embeddings_in_batches(chunks, document_id)
            
            # Подготовка записей для вставки
            records = []
            for i, (chunk_text, sheet_name) in enumerate(chunks):
                # Преобразуем float в double для совместимости с Infinity
                embedding_double = [float(x) for x in all_embeddings[i]]
                
                # Получаем связанные с чанком данные (текст, название листа)
                normalized_sheet_name = self._normalize_text(sheet_name) if sheet_name else ""

                records.append({
                    "document_id": document_id,
                    "chunk_text": chunk_text,
                    "chunk_text_normalized": self._normalize_text(chunk_text),
                    "document_name": document_name,
                    "document_name_normalized": document_name_normalized,
                    "sheet_name": sheet_name or "", # Сохраняем название листа или пустую строку
                    "sheet_name_normalized": normalized_sheet_name,
                    "embedding": embedding_double
                })
            
            # Вставляем все записи одной операцией
            table.insert(records)
            logger.info(f"Успешно сохранено {len(records)} чанков для документа {document_id} в коллекцию '{table_name}'")
            
            # Проверяем что данные действительно вставлены
            table_info = infinity_client.get_table_info(table_name, "default")
            logger.info(f"Информация о таблице '{table_name}' после вставки: {table_info}")
            
            # Дополнительная проверка - получаем записи для этого документа
            try:
                verification_result = table.output(['document_id', 'chunk_text']).filter(f"document_id = '{document_id}'").to_df()
                if verification_result and len(verification_result) > 0:
                    doc_chunk_count = len(verification_result[0])
                    logger.info(f"Проверка: найдено {doc_chunk_count} чанков для документа {document_id}")
                else:
                    logger.warning(f"Проверка: не найдено чанков для документа {document_id} после вставки!")
            except Exception as e:
                logger.error(f"Ошибка проверки вставленных данных для документа {document_id}: {e}")
            
            # Финальная проверка состояния базы данных
            final_tables = infinity_client.list_tables("default")
            logger.info(f"Финальные таблицы в БД: {final_tables}")

        except Exception as e:
            logger.error(f"Ошибка с Infinity DB для документа {document_id}: {e}", exc_info=True)

    async def _create_embeddings_in_batches(self, chunks: list[tuple[str, str | None]], document_id: str) -> list[list[float]]:
        """
        Создает embeddings в батчах с учетом rate limiting Yandex API (10 req/sec).
        Включает проверку размера токенов и автоматическое разбиение больших чанков.
        Теперь принимает список кортежей (chunk_text, sheet_name).
        """
        # Определяем размер батча на основе лимитов API
        batch_size = 25
        all_embeddings = []
        
        # Извлекаем только текст для передачи в модель эмбеддингов
        chunk_texts = [chunk_text for chunk_text, _ in chunks]
        
        # Проверка токенов и разбиение больших чанков с точным подсчетом
        validated_chunks = []
        split_count = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            logger.debug(f"Проверяем чанк {i+1}/{len(chunk_texts)} (длина: {len(chunk_text)} символов)")
            
            try:
                # Добавляем timeout для проверки токенов
                token_count = await asyncio.wait_for(
                    self._count_tokens(chunk_text), 
                    timeout=15.0  # 15 секунд на подсчет токенов
                )
                
                if token_count > 1800:  # Превышает безопасный лимит
                    logger.warning(f"Чанк {i+1} превышает лимит токенов ({token_count} токенов), разбиваем...")
                    
                    # Добавляем timeout для разбиения
                    split_chunks = await asyncio.wait_for(
                        self._split_large_chunk(chunk_text, max_tokens=1800),
                        timeout=60.0  # 60 секунд на разбиение большого чанка
                    )
                    
                    validated_chunks.extend(split_chunks)
                    split_count += len(split_chunks) - 1
                    logger.info(f"Успешно разбили чанк {i+1} на {len(split_chunks)} частей")
                else:
                    validated_chunks.append(chunk_text)
                    logger.debug(f"Чанк {i+1} прошел валидацию ({token_count} токенов)")
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout при обработке чанка {i+1}, используем приблизительное разбиение")
                # Fallback: используем приблизительное разбиение
                if len(chunk_text) > 3600:  # Примерно 1800 токенов
                    # Простое разбиение пополам
                    mid = len(chunk_text) // 2
                    part1 = chunk_text[:mid]
                    part2 = chunk_text[mid:]
                    validated_chunks.extend([part1, part2])
                    logger.info(f"Применено экстренное разбиение к чанку {i+1}")
                else:
                    validated_chunks.append(chunk_text)
                    logger.info(f"Чанк {i+1} принят без валидации из-за timeout")
            except Exception as e:
                logger.error(f"Ошибка при обработке чанка {i+1}: {e}")
                # В случае ошибки просто добавляем чанк как есть
                validated_chunks.append(chunk_text)
                logger.info(f"Чанк {i+1} принят из-за ошибки обработки")
        
        if split_count > 0:
            logger.info(f"Разбито {split_count} слишком больших чанков. Всего чанков: {len(validated_chunks)}")
        
        logger.info(f"Создание эмбеддингов для {len(validated_chunks)} чанков батчами по {batch_size} с точным подсчетом токенов")
        
        start_time = time.time()
        
        for i in range(0, len(validated_chunks), batch_size):
            batch = validated_chunks[i:i + batch_size]
            batch_start_time = time.time()
            
            # Дополнительная проверка токенов в батче перед отправкой
            valid_batch = []
            for chunk_text in batch:
                token_count = await self._count_tokens(chunk_text)
                if token_count <= 1800:
                    valid_batch.append(chunk_text)
                else:
                    logger.warning(f"Пропускаем чанк с {token_count} токенами (все еще слишком большой после разбиения)")
                    # Создаем dummy embedding для пропущенного чанка
                    dummy_embedding = [0.0] * YANDEX_EMBEDDING_DIMENSION
                    all_embeddings.append(dummy_embedding)
            
            if not valid_batch:
                logger.warning(f"Нет валидных чанков в батче {i//batch_size + 1}, пропускаем")
                continue
            
            try:
                # Используем асинхронный метод для создания эмбеддингов батча
                if hasattr(self.embedding_model, 'aembed_documents'):
                    batch_embeddings = await self.embedding_model.aembed_documents(valid_batch)
                else:
                    # Fallback для не-Yandex провайдеров
                    batch_embeddings = await asyncio.to_thread(self.embedding_model.embed_documents, valid_batch)
                
                all_embeddings.extend(batch_embeddings)
                
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                logger.info(f"Обработан батч {i//batch_size + 1}/{(len(validated_chunks) + batch_size - 1)//batch_size} "
                           f"({len(valid_batch)} чанков) за {batch_duration:.2f}с для документа {document_id}")
                
                # Rate limiting: обеспечиваем минимальный интервал между батчами
                min_batch_interval = len(valid_batch) * 0.1  # 0.1 сек на чанк
                if batch_duration < min_batch_interval and i + batch_size < len(validated_chunks):
                    sleep_time = min_batch_interval - batch_duration
                    logger.debug(f"Rate limiting: пауза {sleep_time:.2f}с перед следующим батчем")
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Ошибка создания эмбеддингов для батча {i//batch_size + 1}: {e}", exc_info=True)
                # В случае ошибки пытаемся обработать чанки по одному
                logger.info(f"Переход к индивидуальной обработке чанков для батча {i//batch_size + 1}")
                for chunk_text in valid_batch:
                    try:
                        # Финальная проверка токенов перед индивидуальной обработкой
                        token_count = await self._count_tokens(chunk_text)
                        if token_count > 1800:
                            logger.warning(f"Пропускаем индивидуальный чанк с {token_count} токенами")
                            dummy_embedding = [0.0] * YANDEX_EMBEDDING_DIMENSION
                            all_embeddings.append(dummy_embedding)
                            continue
                        
                        if hasattr(self.embedding_model, 'aembed_query'):
                            individual_embedding = await self.embedding_model.aembed_query(chunk_text)
                        else:
                            individual_embedding = await asyncio.to_thread(self.embedding_model.embed_query, chunk_text)
                        all_embeddings.append(individual_embedding)
                        # Небольшая пауза между индивидуальными запросами
                        await asyncio.sleep(0.1)
                    except Exception as individual_error:
                        logger.error(f"Ошибка создания эмбеддинга для индивидуального чанка: {individual_error}")
                        # Создаем dummy embedding чтобы не нарушить порядок
                        dummy_embedding = [0.0] * YANDEX_EMBEDDING_DIMENSION
                        all_embeddings.append(dummy_embedding)
                        
        # Добавляем dummy embeddings если количество не совпадает с исходными чанками
        expected_count = len(chunks)  # Исходное количество чанков
        actual_count = len(all_embeddings)
        
        if actual_count < expected_count:
            missing_count = expected_count - actual_count
            logger.warning(f"Создание {missing_count} dummy эмбеддингов для соответствия количеству чанков")
            for _ in range(missing_count):
                dummy_embedding = [0.0] * YANDEX_EMBEDDING_DIMENSION
                all_embeddings.append(dummy_embedding)
        
        total_time = time.time() - start_time
        avg_time_per_chunk = total_time / len(validated_chunks) if validated_chunks else 0
        
        logger.info(f"Завершено создание эмбеддингов для документа {document_id}: "
                   f"{len(validated_chunks)} валидированных чанков за {total_time:.2f}с "
                   f"(среднее {avg_time_per_chunk:.2f}с на чанк)")
        
        return all_embeddings

    def _update_document_with_metadata(self, db: Session, document_id: str, status: DocumentStatus,
                                      file_size_bytes: int, content_length: int, chunk_count: int,
                                      processing_time_seconds: float):
        """Обновляет документ со статусом и метаданными"""
        db.query(Document).filter(Document.id == document_id).update({
            "status": status,
            "file_size_bytes": file_size_bytes,
            "content_length": content_length,
            "chunk_count": chunk_count,
            "processing_time_seconds": processing_time_seconds
        })
        db.commit()

ingestion_service = IngestionService() 