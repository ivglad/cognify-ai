import logging
import spacy
import kuzu
import os
from pathlib import Path
from app.core.config import settings
from typing import List, Tuple
import ast
import asyncio
import re
import threading

from app.llm.factory import llm

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    def __init__(self):
        self._db = None
        self._conn = None
        self._connection_failed = False  # Флаг для отключения KG при проблемах
        self._retry_count = 0
        self._lock = threading.Lock()  # Для потокобезопасности
        
        try:
            self.nlp = spacy.load("ru_core_news_sm")
            logger.info("Spacy model 'ru_core_news_sm' loaded successfully.")
        except OSError:
            logger.error("Spacy model 'ru_core_news_sm' not found. Please run 'python -m spacy download ru_core_news_sm'")
            # Fallback to a blank model to avoid crashing the app on startup
            self.nlp = spacy.blank("ru")

        # Инициализируем базу данных при создании сервиса
        self._init_database()

    def _init_database(self):
        """Инициализирует базу данных Kuzu"""
        try:
            # Создаем директорию для базы данных
            db_path = Path("/app/data/knowledge_graph")
            db_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing Kuzu database at {db_path}")
            
            # Создаем базу данных и подключение
            self._db = kuzu.Database(str(db_path))
            self._conn = kuzu.Connection(self._db)
            
            # Создаем схему графа знаний
            self._create_schema()
            
            self._retry_count = 0
            logger.info("Kuzu database initialized successfully")
            
        except Exception as e:
            self._retry_count += 1
            logger.error(f"Failed to initialize Kuzu database (attempt {self._retry_count}): {e}")
            if self._retry_count >= 3:
                self._connection_failed = True
                logger.warning("Knowledge Graph disabled after 3 failed initialization attempts")
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
            
            self._conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Entity(
                    name STRING, 
                    type STRING, 
                    PRIMARY KEY(name)
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
            
            logger.info("Knowledge graph schema created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def _ensure_connection(self):
        """Проверяет и восстанавливает подключение к базе данных"""
        if self._connection_failed:
            raise Exception("Kuzu connection disabled due to repeated failures")
            
        if self._db is None or self._conn is None:
            self._init_database()
            
        if self._db is None or self._conn is None:
            raise Exception("Failed to establish Kuzu connection")

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
                
            logger.info("Kuzu connection closed.")
        except Exception as e:
            logger.error(f"Error closing Kuzu connection: {e}")

    async def _extract_relations_with_llm(self, sentence: str) -> List[Tuple[str, str, str]]:
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
                logger.debug(f"Primary parsing failed, trying alternative: {parse_error}")
                
                # Ищем список в скобках
                import re
                match = re.search(r'\[(.*?)\]', response_text, re.DOTALL)
                if match:
                    try:
                        relations = ast.literal_eval('[' + match.group(1) + ']')
                    except (ValueError, SyntaxError):
                        logger.warning(f"Alternative parsing also failed for: '{response_text[:100]}...'")
                        return []
                else:
                    logger.warning(f"No list pattern found in response: '{response_text[:100]}...'")
                    return []
            
            if not isinstance(relations, list):
                logger.warning(f"LLM returned non-list response: {type(relations)}")
                return []
                
            # Проверяем и фильтруем валидные отношения
            valid_relations = []
            for rel in relations:
                if isinstance(rel, (tuple, list)) and len(rel) == 3:
                    # Проверяем, что все элементы - строки
                    if all(isinstance(item, str) and item.strip() for item in rel):
                        valid_relations.append(tuple(str(item).strip() for item in rel))
                    else:
                        logger.debug(f"Invalid relation elements: {rel}")
                else:
                    logger.debug(f"Invalid relation format: {rel}")
                    
            if valid_relations:
                logger.info(f"Extracted {len(valid_relations)} relations from '{sentence[:50]}...'")
            return valid_relations
            
        except Exception as e:
            logger.error(f"LLM relation extraction failed for sentence '{sentence[:50]}...': {e}", exc_info=True)
            return []

    async def add_document_and_extract_entities(self, document_id: str, document_name: str, text_chunks: List[str]):
        """
        Processes text chunks to extract entities, relationships and build the knowledge graph.
        """
        if self._connection_failed:
            logger.warning("Knowledge Graph processing skipped - Kuzu is disabled")
            return
            
        try:
            with self._lock:  # Обеспечиваем потокобезопасность
                self._ensure_connection()
                
                # Создаем узел документа
                try:
                    self._conn.execute(
                        "MERGE (d:Document {id: $id, name: $name})",
                        parameters={"id": document_id, "name": document_name}
                    )
                except Exception as doc_error:
                    logger.debug(f"Document '{document_id}' already exists or error creating: {doc_error}")

                for i, chunk in enumerate(text_chunks):
                    chunk_id = f"{document_id}_chunk_{i}"
                    
                    # Создаем узел чанка и связываем с документом
                    try:
                        self._conn.execute(
                            "MERGE (c:Chunk {id: $chunk_id, text: $text})",
                            parameters={"chunk_id": chunk_id, "text": chunk}
                        )
                    except Exception as chunk_error:
                        logger.debug(f"Chunk '{chunk_id}' already exists or error creating: {chunk_error}")
                    
                    try:
                        self._conn.execute(
                            """
                            MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id})
                            MERGE (d)-[:CONTAINS_CHUNK]->(c)
                            """,
                            parameters={"doc_id": document_id, "chunk_id": chunk_id}
                        )
                    except Exception as contains_error:
                        logger.debug(f"Error creating CONTAINS_CHUNK relation: {contains_error}")

                    # Извлекаем сущности из чанка
                    doc = self.nlp(chunk)
                    for ent in doc.ents:
                        # Создаем узел сущности и связываем с чанком - безопасно
                        try:
                            self._conn.execute(
                                "MERGE (e:Entity {name: $name, type: $type})",
                                parameters={"name": ent.text, "type": ent.label_}
                            )
                        except Exception as ent_error:
                            logger.debug(f"Entity '{ent.text}' already exists or error creating: {ent_error}")
                        
                        try:
                            self._conn.execute(
                                """
                                MATCH (c:Chunk {id: $chunk_id}), (e:Entity {name: $name})
                                MERGE (c)-[:MENTIONS]->(e)
                                """,
                                parameters={"chunk_id": chunk_id, "name": ent.text}
                            )
                        except Exception as rel_error:
                            logger.debug(f"Error creating MENTIONS relation for '{ent.text}': {rel_error}")
                    
                    # Извлекаем и сохраняем отношения между сущностями в предложениях
                    for sent in doc.sents:
                        # Фильтрация неподходящих предложений
                        sent_text = sent.text.strip()
                        
                        # Пропускаем слишком короткие или длинные предложения
                        if len(sent_text) < 10 or len(sent_text) > 500:
                            continue
                            
                        # Пропускаем предложения без букв (только числа/символы)
                        if not any(c.isalpha() for c in sent_text):
                            continue
                            
                        # Пропускаем предложения с слишком много заглавных букв (вероятно, таблицы)
                        if sum(1 for c in sent_text if c.isupper()) > len(sent_text) * 0.5:
                            continue
                        
                        if len(sent.ents) > 1:  # Обрабатываем только предложения с несколькими сущностями
                            relations = await self._extract_relations_with_llm(sent_text)
                            for subj, rel, obj in relations:
                                # Создаем сущности если они не существуют - используем безопасный подход
                                try:
                                    self._conn.execute(
                                        "MERGE (e1:Entity {name: $subj, type: 'EXTRACTED'})",
                                        parameters={"subj": subj}
                                    )
                                except Exception as e1:
                                    logger.debug(f"Entity '{subj}' already exists or error creating: {e1}")
                                
                                try:
                                    self._conn.execute(
                                        "MERGE (e2:Entity {name: $obj, type: 'EXTRACTED'})",
                                        parameters={"obj": obj}
                                    )
                                except Exception as e2:
                                    logger.debug(f"Entity '{obj}' already exists or error creating: {e2}")
                                
                                # Создаем отношение
                                try:
                                    self._conn.execute(
                                        """
                                        MATCH (e1:Entity {name: $subj}), (e2:Entity {name: $obj})
                                        MERGE (e1)-[:RELATION {type: $rel}]->(e2)
                                        """,
                                        parameters={"subj": subj, "rel": rel, "obj": obj}
                                    )
                                except Exception as e3:
                                    logger.debug(f"Error creating relation between '{subj}' and '{obj}': {e3}")

                logger.info(f"Processed {len(text_chunks)} chunks for document {document_id} and updated knowledge graph.")
                
        except Exception as e:
            logger.error(f"Failed to process knowledge graph for document {document_id}: {e}", exc_info=True)
            # Не прерываем выполнение, просто логируем ошибку

    async def find_related_entities(self, query: str) -> List[str]:
        """
        Finds entities related to a given query string by looking for co-occurring entities in the same documents.
        """
        if self._connection_failed:
            logger.debug("Knowledge Graph search skipped - Kuzu is disabled")
            return []
            
        doc = self.nlp(query)
        query_entities = [ent.text for ent in doc.ents]

        if not query_entities:
            logger.debug("No entities found in the query to search for in the knowledge graph.")
            return []

        logger.debug(f"Found entities in query: {query_entities}. Searching for related entities in Kuzu.")
        
        try:
            with self._lock:  # Обеспечиваем потокобезопасность
                self._ensure_connection()
                related_entities = set()

                for entity_name in query_entities:
                    # Находим связанные сущности через документы
                    cypher_query = """
                    MATCH (start_entity:Entity {name: $entity_name})<-[:MENTIONS]-(:Chunk)<-[:CONTAINS_CHUNK]-(doc:Document)
                    MATCH (doc)-[:CONTAINS_CHUNK]->(:Chunk)-[:MENTIONS]->(related_entity:Entity)
                    WHERE start_entity.name <> related_entity.name
                    RETURN DISTINCT related_entity.name AS name
                    LIMIT 10
                    """
                    
                    result = self._conn.execute(cypher_query, parameters={"entity_name": entity_name})
                    
                    # Получаем результаты
                    while result.has_next():
                        record = result.get_next()
                        related_entities.add(record[0])  # Получаем значение по индексу
            
            # Убираем исходные сущности запроса из результата
            final_related_entities = list(related_entities - set(query_entities))
            
            if final_related_entities:
                logger.info(f"Found related entities in KG: {final_related_entities}")
                
            return final_related_entities
            
        except Exception as e:
            logger.warning(f"Failed to find related entities in Kuzu: {e}")
            return []  # Возвращаем пустой список при ошибке

kg_service = KnowledgeGraphService() 