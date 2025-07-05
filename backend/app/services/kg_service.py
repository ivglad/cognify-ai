import logging
import spacy
import kuzu
import os
from pathlib import Path
from app.core.config import settings
import ast
import asyncio
import re
import threading
import time

from app.llm.factory import llm

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    def __init__(self):
        self._db = None
        self._conn = None
        self._connection_failed = False  # Флаг для отключения KG при проблемах
        self._retry_count = 0
        self._init_lock = threading.Lock()  # Только для инициализации
        self._operation_timeout = 30.0  # Timeout для операций в секундах
        
        try:
            logger.info(f"Загрузка Spacy модели: '{settings.SPACY_MODEL_NAME}'...")
            self.nlp = spacy.load(settings.SPACY_MODEL_NAME)
            logger.info("Spacy модель загружена для KnowledgeGraphService.")
        except OSError:
            logger.error(f"Spacy модель '{settings.SPACY_MODEL_NAME}' не найдена. Выполните 'python -m spacy download {settings.SPACY_MODEL_NAME}'")
            # Fallback to a blank model to avoid crashing the app on startup
            self.nlp = spacy.blank("ru")

        # Инициализируем базу данных при создании сервиса
        self._init_database()

    def _init_database(self):
        """Инициализирует базу данных Kuzu с retry логикой"""
        with self._init_lock:  # Только для инициализации
            try:
                # Создаем директорию для базы данных
                db_path = Path("/app/data/knowledge_graph")
                db_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Инициализация БД Kuzu в {db_path}")
                
                # Создаем базу данных и подключение
                self._db = kuzu.Database(str(db_path))
                self._conn = kuzu.Connection(self._db)
                
                # Создаем схему графа знаний
                self._create_schema()
                
                self._retry_count = 0
                self._connection_failed = False
                logger.info("БД Kuzu инициализирована успешно")
                
            except Exception as e:
                self._retry_count += 1
                logger.error(f"Ошибка инициализации БД Kuzu (попытка {self._retry_count}): {e}")
                if self._retry_count >= 3:
                    self._connection_failed = True
                    logger.warning("Knowledge Graph отключен после 3 неудачных попыток инициализации")
                else:
                    # Пробуем еще раз через некоторое время
                    self._db = None
                    self._conn = None

    def _create_schema(self):
        """Создает схему графа знаний в Kuzu"""
        try:
            # Создаем таблицы узлов
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Document(
                    id STRING, 
                    name STRING, 
                    PRIMARY KEY(id)
                )
            """)
            
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Chunk(
                    id STRING, 
                    text STRING, 
                    PRIMARY KEY(id)
                )
            """)
            
            # Исправленная схема Entity с поддержкой дубликатов между документами
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    id STRING,
                    name STRING, 
                    type STRING,
                    document_id STRING,
                    PRIMARY KEY(id)
                )
            """)
            
            # Создаем таблицы связей
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS CONTAINS_CHUNK(
                    FROM Document TO Chunk
                )
            """)
            
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS MENTIONS(
                    FROM Chunk TO Entity
                )
            """)
            
            self._conn.execute("""
                CREATE REL TABLE IF NOT EXISTS RELATION(
                    FROM Entity TO Entity,
                    type STRING
                )
            """)
            
            logger.info("Схема графа знаний создана успешно")
            
        except Exception as e:
            logger.error(f"Ошибка создания схемы: {e}")
            raise

    def _ensure_connection(self):
        """Проверяет и восстанавливает подключение к базе данных"""
        if self._connection_failed:
            raise Exception("Kuzu connection disabled due to repeated failures")
            
        if self._db is None or self._conn is None:
            self._init_database()
            
        if self._db is None or self._conn is None:
            raise Exception("Failed to establish Kuzu connection")

    def _execute_with_timeout(self, query: str, parameters: dict = None):
        """
        Выполняет запрос с timeout для избежания зависаний
        """
        def execute_query():
            if parameters:
                return self._conn.execute(query, parameters=parameters)
            else:
                return self._conn.execute(query)
        
        # Используем asyncio.wait_for эквивалент для sync кода
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(execute_query)
            try:
                return future.result(timeout=self._operation_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Timeout запроса Kuzu после {self._operation_timeout}с: {query[:100]}...")
                raise TimeoutError(f"Query timeout after {self._operation_timeout}s")

    def _clean_llm_response(self, response_text: str) -> str:
        """
        Очищает ответ LLM от непечатаемых символов и форматирования.
        """
        if not response_text:
            return ""
            
        # Удаляем zero-width space и другие невидимые символы
        response_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', response_text)
        
        # Удаляем markdown форматирование
        if response_text.startswith('```') and response_text.endswith('```'):
            lines = response_text.split('\n')
            if len(lines) > 2:
                response_text = '\n'.join(lines[1:-1])
            else:
                response_text = response_text[3:-3]
        
        if response_text.startswith('`') and response_text.endswith('`'):
            response_text = response_text[1:-1]
        
        # Удаляем лишние пробелы и переносы строк
        response_text = response_text.strip()
        
        return response_text

    async def close(self):
        """Закрывает подключение к базе данных"""
        try:
            if self._conn is not None:
                # Kuzu connections закрываются автоматически при удалении объекта
                self._conn = None
                
            if self._db is not None:
                # Kuzu database закрывается автоматически при удалении объекта
                self._db = None
                
            logger.info("Подключение Kuzu закрыто.")
        except Exception as e:
            logger.error(f"Ошибка закрытия подключения Kuzu: {e}")

    async def _extract_relations_with_llm(self, sentence: str) -> list[tuple[str, str, str]]:
        """
        Uses an LLM to extract relationships (subject, relation, object) from a sentence.
        Optimized prompt with enhanced validation and clearer instructions.
        """
        prompt = f"""Ты эксперт по извлечению семантических отношений. Извлеки ТОЛЬКО фактические отношения из предложения.

СТРОГИЕ ПРАВИЛА:
1. Возвращай ИСКЛЮЧИТЕЛЬНО Python список кортежей: [('субъект', 'отношение', 'объект')]
2. Никаких объяснений, комментариев или дополнительного текста
3. Субъект и объект - ТОЛЬКО существительные или именные группы
4. Отношение - глагол или предлог (максимум 3 слова)
5. Извлекай только ЗНАЧИМЫЕ фактические связи
6. Если нет отношений, верни: []

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ:
[('entity1', 'relation', 'entity2'), ('entity3', 'relation', 'entity4')]

ВАЛИДНЫЕ ПРИМЕРЫ:
"Компания Google приобрела стартап DeepMind в 2014 году"
→ [('Google', 'приобрела', 'DeepMind')]

"Джон работает директором в Microsoft и живет в Сиэтле"  
→ [('Джон', 'работает в', 'Microsoft'), ('Джон', 'живет в', 'Сиэтле')]

"Температура воды составляет 20 градусов"
→ [('температура воды', 'составляет', '20 градусов')]

НЕВАЛИДНЫЕ (НЕ извлекай):
- Абстрактные понятия без конкретных субъектов
- Описательные прилагательные как отношения
- Временные наречия как объекты (кроме дат/периодов)

ПРЕДЛОЖЕНИЕ: "{sentence}"

ОТВЕТ (только Python список):"""
        
        try:
            response = await llm.ainvoke(prompt)
            response_text = self._clean_llm_response(response.content)
            
            # Если ответ пустой, возвращаем пустой список
            if not response_text or response_text.isspace():
                return []
            
            # Пытаемся распарсить как Python список
            try:
                relations = ast.literal_eval(response_text)
            except (ValueError, SyntaxError) as parse_error:
                # Пробуем альтернативный парсинг
                logger.debug(f"Первичный парсинг не удался, пробуем альтернативный: {parse_error}")
                
                # Ищем список в скобках
                import re
                match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
                if match:
                    try:
                        relations = ast.literal_eval('[' + match.group(1) + ']')
                    except (ValueError, SyntaxError):
                        logger.warning(f"Альтернативный парсинг также не удался для: '{response_text[:100]}...'")
                        return []
                else:
                    logger.warning(f"Не найден паттерн списка в ответе: '{response_text[:100]}...'")
                    return []
            
            if not isinstance(relations, list):
                logger.warning(f"LLM вернул не список: {type(relations)}")
                return []
                
            # Проверяем и фильтруем валидные отношения
            valid_relations = []
            for rel in relations:
                if isinstance(rel, (tuple, list)) and len(rel) == 3:
                    # Проверяем, что все элементы - строки
                    if all(isinstance(item, str) and item.strip() for item in rel):
                        valid_relations.append(tuple(str(item).strip() for item in rel))
                    else:
                        logger.debug(f"Невалидные элементы отношения: {rel}")
                else:
                    logger.debug(f"Невалидный формат отношения: {rel}")
                    
            if valid_relations:
                logger.info(f"Извлечено {len(valid_relations)} отношений из '{sentence[:50]}...'")
            return valid_relations
            
        except Exception as e:
            logger.error(f"Ошибка извлечения отношений LLM для предложения '{sentence[:50]}...': {e}", exc_info=True)
            return []

    async def add_document_and_extract_entities(self, document_id: str, document_name: str, text_chunks: list[str]):
        """
        Processes text chunks to extract entities, relationships and build the knowledge graph.
        Неблокирующая версия без глобальной блокировки.
        """
        if self._connection_failed:
            logger.warning("Обработка Knowledge Graph пропущена - Kuzu отключен")
            return
        
        start_time = time.time()
        timeout_exceeded = False
        
        try:
            # Проверяем подключение без блокировки
            self._ensure_connection()
            
            logger.info(f"Начало обработки Knowledge Graph для документа {document_id} ({document_name})")
            
            # Создаем узел документа с timeout
            try:
                self._execute_with_timeout(
                    "MERGE (d:Document {id: $id, name: $name})",
                    {"id": document_id, "name": document_name}
                )
                logger.debug(f"Создан узел документа для {document_id}")
            except TimeoutError:
                logger.warning(f"Timeout создания документа для {document_id}")
                timeout_exceeded = True
            except Exception as doc_error:
                logger.debug(f"Документ '{document_id}' уже существует или ошибка создания: {doc_error}")

            if timeout_exceeded:
                logger.warning(f"Обработка Knowledge Graph прервана для {document_id} из-за timeout")
                return

            # Обрабатываем чанки с ограничением времени
            processed_chunks = 0
            max_chunks_per_timeout = 10  # Ограничиваем количество чанков для избежания зависаний
            
            for i, chunk in enumerate(text_chunks[:max_chunks_per_timeout]):  # Ограничиваем количество чанков
                if time.time() - start_time > self._operation_timeout:
                    logger.warning(f"Превышен глобальный timeout для документа {document_id}, обработано {processed_chunks} чанков")
                    timeout_exceeded = True
                    break
                    
                chunk_id = f"{document_id}_chunk_{i}"
                
                try:
                    # Создаем узел чанка и связываем с документом с timeout
                    self._execute_with_timeout(
                        "MERGE (c:Chunk {id: $chunk_id, text: $text})",
                        {"chunk_id": chunk_id, "text": chunk}
                    )
                    
                    self._execute_with_timeout(
                        """
                        MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
                        MERGE (d)-[:CONTAINS_CHUNK]->(c)
                        """,
                        {"doc_id": document_id, "chunk_id": chunk_id}
                    )
                    
                    # Извлекаем сущности из чанка
                    doc = self.nlp(chunk)
                    for ent in doc.ents:
                        try:
                            self._execute_with_timeout(
                                "MERGE (e:Entity {id: $id, name: $name, type: $type, document_id: $doc_id})",
                                {"id": f"{document_id}_{ent.text}", "name": ent.text, "type": ent.label_, "doc_id": document_id}
                            )
                            
                            self._execute_with_timeout(
                                """
                                MATCH (c:Chunk {id: $chunk_id}), (e:Entity {id: $id})
                                MERGE (c)-[:MENTIONS]->(e)
                                """,
                                {"chunk_id": chunk_id, "id": f"{document_id}_{ent.text}"}
                            )
                        except TimeoutError:
                            logger.debug(f"Timeout обработки сущности для {ent.text} в документе {document_id}")
                            continue
                        except Exception as ent_error:
                            logger.debug(f"Ошибка обработки сущности '{ent.text}': {ent_error}")
                            continue
                    
                    # Извлекаем отношения с ограничением по времени
                    for sent in doc.sents:
                        if time.time() - start_time > self._operation_timeout * 0.8:  # 80% от timeout
                            break
                            
                        sent_text = sent.text.strip()
                        
                        # Фильтрация неподходящих предложений
                        if len(sent_text) < 10 or len(sent_text) > 500:
                            continue
                            
                        if not any(c.isalpha() for c in sent_text):
                            continue
                            
                        if sum(1 for c in sent_text if c.isupper()) > len(sent_text) * 0.5:
                            continue
                        
                        if len(sent.ents) > 1:
                            try:
                                # Добавляем timeout для LLM вызова
                                relations = await asyncio.wait_for(
                                    self._extract_relations_with_llm(sent_text),
                                    timeout=10.0  # 10 секунд на LLM запрос
                                )
                                
                                for subj, rel, obj in relations:
                                    try:
                                        self._execute_with_timeout(
                                            "MERGE (e1:Entity {id: $id, name: $subj, type: 'EXTRACTED', document_id: $doc_id})",
                                            {"id": f"{document_id}_{subj}", "subj": subj, "doc_id": document_id}
                                        )
                                        
                                        self._execute_with_timeout(
                                            "MERGE (e2:Entity {id: $id, name: $obj, type: 'EXTRACTED', document_id: $doc_id})",
                                            {"id": f"{document_id}_{obj}", "obj": obj, "doc_id": document_id}
                                        )
                                        
                                        self._execute_with_timeout(
                                            """
                                            MATCH (e1:Entity {id: $id1}), (e2:Entity {id: $id2})
                                            MERGE (e1)-[:RELATION {type: $rel}]->(e2)
                                            """,
                                            {"id1": f"{document_id}_{subj}", "id2": f"{document_id}_{obj}", "rel": rel}
                                        )
                                    except TimeoutError:
                                        logger.debug(f"Timeout создания отношения для {subj}-{rel}-{obj}")
                                        continue
                                    except Exception as rel_error:
                                        logger.debug(f"Ошибка создания отношения: {rel_error}")
                                        continue
                                        
                            except asyncio.TimeoutError:
                                logger.debug(f"Timeout извлечения отношений LLM для предложения в документе {document_id}")
                                continue
                            except Exception as llm_error:
                                logger.debug(f"Ошибка извлечения отношений LLM: {llm_error}")
                                continue
                    
                    processed_chunks += 1
                    
                except TimeoutError:
                    logger.warning(f"Timeout обработки чанка для чанка {i} в документе {document_id}")
                    continue
                except Exception as chunk_error:
                    logger.debug(f"Ошибка обработки чанка для чанка {i}: {chunk_error}")
                    continue

            end_time = time.time()
            processing_time = end_time - start_time
            
            if timeout_exceeded:
                logger.warning(f"Обработка Knowledge Graph для документа {document_id} завершена с timeouts. "
                              f"Обработано {processed_chunks}/{len(text_chunks)} чанков за {processing_time:.2f}с")
            else:
                logger.info(f"Успешно обработан Knowledge Graph для {document_name} "
                           f"({processed_chunks}/{len(text_chunks)} чанков за {processing_time:.2f}с)")
                
        except Exception as e:
            logger.error(f"Ошибка обработки графа знаний для документа {document_id}: {e}", exc_info=True)
            # Не прерываем выполнение, просто логируем ошибку

    async def find_related_entities(self, query: str) -> list[str]:
        """
        Finds entities related to a given query string by looking for co-occurring entities in the same documents.
        Неблокирующая версия с timeout.
        """
        if self._connection_failed:
            logger.debug("Поиск Knowledge Graph пропущен - Kuzu отключен")
            return []
            
        doc = self.nlp(query)
        query_entities = [ent.text for ent in doc.ents]

        if not query_entities:
            logger.debug("В запросе не найдено сущностей для поиска в графе знаний.")
            return []

        logger.debug(f"Найдены сущности в запросе: {query_entities}. Поиск связанных сущностей в Kuzu.")
        
        try:
            self._ensure_connection()
            related_entities = set()

            for entity_name in query_entities:
                try:
                    # Находим связанные сущности через документы (с новой схемой) - с timeout
                    cypher_query = """
                    MATCH (start_entity:Entity)<-[:MENTIONS]-(:Chunk)<-[:CONTAINS_CHUNK]-(doc:Document)
                    WHERE start_entity.name = $entity_name
                    MATCH (doc)-[:CONTAINS_CHUNK]->(:Chunk)-[:MENTIONS]->(related_entity:Entity)
                    WHERE start_entity.id <> related_entity.id
                    RETURN DISTINCT related_entity.name AS name, related_entity.document_id AS doc_id
                    LIMIT 10
                    """
                    
                    result = self._execute_with_timeout(cypher_query, {"entity_name": entity_name})
                    
                    # Получаем результаты
                    while result.has_next():
                        record = result.get_next()
                        entity_name_result = record[0]
                        doc_id = record[1]
                        # Добавляем информацию о документе для контекста
                        related_entities.add(f"{entity_name_result} (из документа {doc_id[:8]}...)")
                    
                    # Также ищем прямые отношения между сущностями
                    relation_query = """
                    MATCH (start_entity:Entity)-[:RELATION]->(related_entity:Entity)
                    WHERE start_entity.name = $entity_name
                    RETURN DISTINCT related_entity.name AS name, related_entity.document_id AS doc_id
                    LIMIT 5
                    """
                    
                    result = self._execute_with_timeout(relation_query, {"entity_name": entity_name})
                    
                    while result.has_next():
                        record = result.get_next()
                        entity_name_result = record[0]
                        doc_id = record[1]
                        related_entities.add(f"{entity_name_result} (связана с {entity_name})")
                        
                except TimeoutError:
                    logger.warning(f"Timeout поиска связанных сущностей для '{entity_name}'")
                    continue
                except Exception as search_error:
                    logger.debug(f"Ошибка поиска связанных сущностей для '{entity_name}': {search_error}")
                    continue
            
            # Убираем исходные сущности запроса из результата
            final_related_entities = list(related_entities)
            
            if final_related_entities:
                logger.info(f"Найдены связанные сущности в KG: {final_related_entities}")
                
            return final_related_entities
            
        except Exception as e:
            logger.warning(f"Ошибка поиска связанных сущностей в Kuzu: {e}")
            return []  # Возвращаем пустой список при ошибке

kg_service = KnowledgeGraphService() 