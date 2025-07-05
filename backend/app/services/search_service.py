from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage, SearchResult
from app.llm.factory import llm, embedding_model
from app.db.infinity_client import infinity_client
from app.db.session import SessionLocal
from app.models.document import Document
from infinity.index import IndexInfo, IndexType
import asyncio
import logging
import time
import uuid

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app.services.kg_service import kg_service
import spacy

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.embedding_model = embedding_model
        # Используем централизованный Infinity клиент
        logger.info("SearchService использует центр. клиент Infinity")
        
        # Загружаем Spacy модель для извлечения сущностей из запросов
        try:
            logger.info(f"Загрузка Spacy модели '{settings.SPACY_MODEL_NAME}' для SearchService (NER)...")
            self.nlp = spacy.load(settings.SPACY_MODEL_NAME)
            logger.info("Spacy модель загружена для SearchService.")
        except OSError:
            logger.error(f"Spacy модель '{settings.SPACY_MODEL_NAME}' не найдена для SearchService. NER в запросах отключен.")
            self.nlp = None

        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Загрузка отсутствующих данных NLTK...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        try:
            self.russian_stopwords = set(stopwords.words("russian"))
        except LookupError:
            logger.warning("Русские стоп-слова недоступны, используется пустой набор")
            self.russian_stopwords = set()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalizes text for BM25 search.
        """
        try:
            tokens = word_tokenize(text.lower(), language='russian')
            return " ".join([token for token in tokens if token.isalnum() and token not in self.russian_stopwords])
        except LookupError as e:
            logger.error(f"Ошибка токенизации NLTK: {e}. Переход к простому разделению.")
            # Fallback к простому разделению по словам
            simple_tokens = text.lower().split()
            return " ".join([token for token in simple_tokens if token.isalnum() and token not in self.russian_stopwords])
    
    async def _condense_query_with_history(self, query: str, history: list[ChatMessage]) -> str:
        """
        Condenses chat history and a new query into a standalone query.
        """
        if not history:
            return query

        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        logger.info(f"Сжатие запроса с историей: {history_str}")

        prompt = f"""Учитывая историю чата и новый вопрос, переформулируй новый вопрос так, чтобы он был самодостаточным.
        
        История чата:
        {history_str}
        
        Новый вопрос: {query}
        
        Самодостаточный вопрос:"""

        try:
            condensed_query = await llm.ainvoke(prompt)
            logger.info(f"Сжатый запрос: '{condensed_query.content.strip()}'")
            return condensed_query.content.strip()
        except Exception as e:
            logger.error(f"Ошибка сжатия запроса: {e}", exc_info=True)
            return query # Fallback to original query

    async def _rewrite_query(self, query: str) -> str:
        """
        Rewrites the user's query using an LLM for better retrieval.
        """
        logger.info(f"Переписывание запроса: '{query}'")
        
        prompt = f"""Переформулируй следующий вопрос для максимально эффективного поиска в векторной базе данных. 
        Сохрани ключевые слова и основной смысл. Ответь только переформулированным вопросом.
        
        Оригинальный вопрос: {query}
        
        Переформулированный вопрос:"""
        
        try:
            rewritten_query = await llm.ainvoke(prompt)
            # Clean up the response, as the model might add extra text
            rewritten_query = rewritten_query.content.strip()
            logger.info(f"Переписанный запрос: '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"Ошибка переписывания запроса: {e}", exc_info=True)
            return query # Fallback to original query

    async def _rerank_chunks(self, query: str, chunk_data: list[tuple[str, uuid.UUID, float]]) -> list[tuple[str, uuid.UUID, float]]:
        """
        Re-ranks a list of chunk data based on their relevance to the query using an LLM.
        Takes and returns list of tuples: (chunk_text, document_id, score)
        """
        if not chunk_data:
            return []

        logger.info(f"Переранжирование {len(chunk_data)} чанков для запроса: '{query}'")

        # Extract just the text for ranking
        chunks = [chunk_text for chunk_text, _, _ in chunk_data]
        
        # Create a numbered list of chunks for the prompt
        numbered_chunks = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])

        prompt = f"""Ниже приведен запрос пользователя и список извлеченных из документов фрагментов текста.
Твоя задача — оценить, насколько каждый фрагмент релевантен для ответа на запрос. Рассуждай шаг за шагом. Анализируй таблицы и списки внимательно. 
Верни только номера наиболее релевантных фрагментов, отсортированные по убыванию релевантности, в виде списка через запятую. Например: 3, 1, 5.
Не включай в ответ нерелевантные фрагменты. Если релевантных фрагментов нет, верни пустую строку.

Запрос: {query}

Фрагменты:
{numbered_chunks}

Отсортированный список номеров релевантных фрагментов:"""

        try:
            response = await llm.ainvoke(prompt)
            response_text = response.content.strip()
            logger.info(f"Ответ LLM переранжирования: '{response_text}'")

            if not response_text:
                return []

            # Parse the response
            relevant_indices = [int(i.strip()) - 1 for i in response_text.split(',') if i.strip().isdigit()]

            # Create the re-ranked list of chunk data
            reranked_chunk_data = [chunk_data[i] for i in relevant_indices if 0 <= i < len(chunk_data)]

            if not reranked_chunk_data:
                logger.warning("Переранжирование не вернуло чанков, возврат к исходному порядку.")
                return chunk_data

            logger.info(f"Переранжировано и оставлено {len(reranked_chunk_data)} чанков.")
            return reranked_chunk_data
        except Exception as e:
            logger.error(f"Ошибка переранжирования чанков: {e}", exc_info=True)
            return chunk_data # Fallback to original fused list

    async def search(self, request: ChatRequest) -> tuple[str, list[ChatMessage], list[SearchResult]]:
        """
        Performs a hybrid search and returns an answer, updated history, and sources.
        """
        start_time = time.monotonic()
        original_query = request.query
        history = request.history
        logger.info(f"Поиск: '{original_query}' в документах: {request.document_ids} с историей.")

        # 0. Condense query with history
        condensed_query = await self._condense_query_with_history(original_query, history)

        # 1. Rewrite the condensed query for better retrieval
        if settings.ENABLE_QUERY_REWRITING:
            rewritten_query = await self._rewrite_query(condensed_query)
        else:
            logger.info("Переписывание запросов отключено.")
            rewritten_query = condensed_query

        # 2. Embed the rewritten query
        if hasattr(self.embedding_model, 'aembed_query'):
            query_embedding = await self.embedding_model.aembed_query(rewritten_query)
        else:
            # Fallback для не-Yandex провайдеров
            query_embedding = await asyncio.to_thread(self.embedding_model.embed_query, rewritten_query)
        # Преобразуем в double для совместимости с Infinity
        query_embedding_double = [float(x) for x in query_embedding]

        # 3. Find relevant chunks using hybrid search - теперь возвращает данные о документах
        fused_chunk_data = await asyncio.to_thread(
            self._find_relevant_chunks,
            rewritten_query,
            query_embedding_double,
            request.document_ids
        )

        # 3.5 Re-rank the fused chunks using the original query for relevance
        reranked_chunk_data = await self._rerank_chunks(original_query, fused_chunk_data)

        # 4. Извлекаем только текст для LLM
        reranked_chunks = [chunk_text for chunk_text, _, _ in reranked_chunk_data]

        # 5. Generate a response using the LLM with full context
        related_entities = await kg_service.find_related_entities(condensed_query)
        kg_context = ""
        if related_entities:
            kg_context = f"Найдены связанные сущности в базе знаний: {', '.join(related_entities)}."

        answer = await self._generate_llm_response(original_query, reranked_chunks, history, kg_context)

        # 6. Проверяем успешность ответа и создаем sources только если ответ полезный
        search_results = []
        if self._is_answer_successful(answer) and reranked_chunk_data:
            search_results = await self._create_search_results(reranked_chunk_data)

        # 7. Update history
        updated_history = history + [
            ChatMessage(role="user", content=original_query),
            ChatMessage(role="assistant", content=answer)
        ]

        end_time = time.monotonic()
        logger.info(f"Запрос поиска '{original_query}' выполнен за {end_time - start_time:.2f} сек.")
        return answer, updated_history, search_results

    def _is_answer_successful(self, answer: str) -> bool:
        """Проверяет, дал ли LLM полезный ответ"""
        negative_indicators = [
            "я не нашел информации в документах",
            "я не нашел релевантной информации", 
            "нет информации в документах",
            "информация отсутствует",
            "не удалось найти"
        ]
        
        answer_lower = answer.lower()
        return not any(indicator in answer_lower for indicator in negative_indicators)

    def _find_relevant_chunks(self, query: str, query_embedding: list[float], document_ids: list[uuid.UUID]) -> list[tuple[str, uuid.UUID, float]]:
        """
        Гибридный поиск с поддержкой названий документов, листов Excel и NER из запроса.
        Возвращает список кортежей: (chunk_text, document_id, score)
        
        Использует динамическую стратегию поиска:
        - Поиск по нескольким документам: 5-компонентный поиск (вектор, контент, имя док, имя листа, NER) с weighted_sum.
        - Поиск по одному Excel-документу: 4-компонентный поиск (вектор, контент, имя листа, NER) с rrf.
        - Поиск по одному обычному документу: 3-компонентный поиск (вектор, контент, NER) с rrf.
        """
        logger.info("Выполняется динамический гибридный поиск в 'chunks_collection'...")
        
        # Извлекаем сущности из запроса с помощью Spacy
        query_entities = []
        if self.nlp:
            doc = self.nlp(query)
            query_entities = [ent.text for ent in doc.ents]
            if query_entities:
                logger.info(f"Извлечены сущности из запроса: {query_entities}")

        db = infinity_client.get_database("default")
        try:
            table = db.get_table("chunks_collection")
        except Exception as e:
            logger.error(f"Не удалось получить таблицу 'chunks_collection': {e}")
            return []
        
        normalized_query = self._normalize_text(query)
        
        query_builder = table.output(['chunk_text', 'document_id', 'document_name', 'sheet_name'])

        # Определяем стратегию поиска
        is_single_doc_search = len(document_ids) == 1
        
        # Если поиск по одному документу, нам нужно знать его тип
        is_excel = False
        if is_single_doc_search:
            pg_db = SessionLocal()
            try:
                doc = pg_db.query(Document).filter(Document.id == document_ids[0]).first()
                if doc and doc.content_type and 'spreadsheet' in doc.content_type:
                    is_excel = True
            finally:
                pg_db.close()

        try:
            # Этап 1: Формирование запроса на основе стратегии
            if is_single_doc_search:
                if is_excel:
                    # Сценарий B: Один Excel-документ (вектор + контент + имя листа)
                    logger.info("Стратегия поиска: Один Excel документ (вектор + контент + имя листа, rrf fusion)")
                    query_builder = query_builder.knn('embedding', query_embedding, 'double', 'cosine', 10) \
                                               .match_text('chunk_text_normalized', normalized_query, topn=10) \
                                               .match_text('sheet_name_normalized', normalized_query, topn=10)
                    fusion_method = 'rrf'
                else:
                    # Сценарий C: Один обычный документ (вектор + контент)
                    logger.info("Стратегия поиска: Один обычный документ (вектор + контент, rrf fusion)")
                    query_builder = query_builder.knn('embedding', query_embedding, 'double', 'cosine', 10) \
                                               .match_text('chunk_text_normalized', normalized_query, topn=10)
                    fusion_method = 'rrf'
            else:
                # Сценарий A: Несколько документов (вектор + контент + имя док + имя листа)
                logger.info("Стратегия поиска: Несколько документов (вектор + контент + имя док + имя листа, weighted_sum fusion)")
                query_builder = query_builder.knn('embedding', query_embedding, 'double', 'cosine', 10) \
                                           .match_text('chunk_text_normalized', normalized_query, topn=10) \
                                           .match_text('document_name_normalized', normalized_query, topn=5) \
                                           .match_text('sheet_name_normalized', normalized_query, topn=5)
                fusion_method = 'weighted_sum'

            # Добавляем поиск по сущностям во все сценарии
            if query_entities:
                entities_query = " ".join(query_entities)
                query_builder = query_builder.match_text('chunk_text_normalized', entities_query, topn=5)
            
            # Этап 2: Фильтрация по ID, если необходимо
            if document_ids:
                doc_id_list = ", ".join([f"'{str(doc_id)}'" for doc_id in document_ids])
                filter_condition = f"document_id IN ({doc_id_list})"
                query_builder = query_builder.filter(filter_condition)
                
            # Этап 3: Применение Fusion
            if fusion_method == 'weighted_sum':
                # Веса: вектор, контент, имя док, имя листа, NER
                # Если NER сущности есть, используем 5 весов, иначе 4
                weights = "0.35,0.25,0.1,0.1,0.2" if query_entities else "0.4,0.3,0.15,0.15"
                res_df_list = query_builder.fusion(fusion_method, 10, {"weights": weights}).to_df()
            else: # rrf
                res_df_list = query_builder.fusion(fusion_method, 10).to_df()
            
            # Этап 4: Обработка результатов
            if res_df_list and not res_df_list[0].empty:
                result_df = res_df_list[0]
                chunk_texts = result_df['chunk_text'].tolist()
                doc_ids = result_df['document_id'].tolist()
                
                # Создаем синтетические scores на основе позиции в результатах
                chunk_data = [
                    (text, uuid.UUID(doc_id), 1.0 - (i * 0.1)) 
                    for i, (text, doc_id) in enumerate(zip(chunk_texts, doc_ids))
                ]
                
                logger.info(f"Динамический гибридный поиск нашел {len(chunk_data)} чанков используя '{fusion_method}' fusion.")
                return chunk_data
            else:
                logger.warning(f"Динамический гибридный поиск с '{fusion_method}' fusion не вернул результатов.")

        except Exception as e:
            logger.error(f"Ошибка динамического гибридного поиска: {e}", exc_info=True)

        # Универсальный Fallback к векторному поиску, если все остальное не удалось
        try:
            logger.warning("Переход к векторному поиску.")
            query_builder = table.output(['chunk_text', 'document_id']) \
                                  .knn('embedding', query_embedding, 'double', 'cosine', 10)
            
            if document_ids:
                doc_id_list = ", ".join([f"'{str(doc_id)}'" for doc_id in document_ids])
                filter_condition = f"document_id IN ({doc_id_list})"
                query_builder = query_builder.filter(filter_condition)
                
            fallback_result = query_builder.to_df()
            
            if fallback_result and not fallback_result[0].empty:
                result_df = fallback_result[0]
                chunk_texts = result_df['chunk_text'].tolist()
                doc_ids = result_df['document_id'].tolist()
                
                chunk_data = [
                    (text, uuid.UUID(doc_id), 0.5)  # Фиксированный score для fallback
                    for text, doc_id in zip(chunk_texts, doc_ids)
                ]
                
                logger.info(f"Векторный fallback нашел {len(chunk_data)} чанков")
                return chunk_data
            
        except Exception as fallback_error:
            logger.error(f"Fallback векторный поиск также не удался: {fallback_error}")
            
        return []

    async def _create_search_results(self, chunk_data: list[tuple[str, uuid.UUID, float]]) -> list[SearchResult]:
        """
        Создает SearchResult объекты с информацией о документах из PostgreSQL.
        Content не включается для уменьшения размера ответа.
        """
        if not chunk_data:
            return []
        
        # Получаем уникальные document_ids
        document_ids = list(set(doc_id for _, doc_id, _ in chunk_data))
        
        # Получаем имена файлов из PostgreSQL
        db = SessionLocal()
        try:
            documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
            doc_id_to_name = {doc.id: doc.file_name for doc in documents}
        finally:
            db.close()
        
        # Группируем чанки по документам для уменьшения дублирования
        doc_to_chunks = {}
        for _, doc_id, score in chunk_data:
            if doc_id not in doc_to_chunks:
                doc_to_chunks[doc_id] = score  # Берем лучший score для документа
        
        # Создаем SearchResult объекты (один на документ)
        search_results = []
        for doc_id, score in doc_to_chunks.items():
            doc_name = doc_id_to_name.get(doc_id, "Неизвестный документ")
            search_results.append(SearchResult(
                content="",  # Контент не нужен в источниках
                document_id=doc_id,
                document_name=doc_name,
                confidence_score=score
            ))
        
        return search_results

    async def _generate_llm_response(self, query: str, context_chunks: list[str], history: list[ChatMessage], kg_context: str) -> str:
        """
        Generates a response using the YandexGPT LLM, including chat history.
        """
        
        if not context_chunks and not kg_context:
            return "К сожалению, я не нашел релевантной информации попробуйте уточнить вопрос."

        context = "\n---\n".join(context_chunks) if context_chunks else "Нет данных из документов."
        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        
        logger.info(f"LLM контекст: {context}")
        logger.info(f"LLM история: {history_str}")

        prompt_template = f"""Ты — умный ИИ-ассистент. Используй историю чата, контекст из документов и, если есть, данные из базы знаний, чтобы ответить на вопрос.
Контекст может содержать таблицы в формате Markdown. Анализируй их структуру (строки и столбцы) для точного ответа.
Если ты не знаешь ответа или информация отсутствует в предоставленном контексте, так и скажи: "Я не нашел информации в документах". Не придумывай ответ.
Формулировки должны быть краткими и понятными, на русском языке.

История чата:
{history_str}

Контекст из документов:
{context}

Данные из базы знаний:
{kg_context or "Нет данных."}

Вопрос: {query}
Ответ:"""
        
        try:
            response = await llm.ainvoke(prompt_template)
            logger.info(f"LLM ответ: {response}")
            return response.content
        except Exception as e:
            logger.error(f"Ошибка вызова YandexGPT: {e}", exc_info=True)
            return "Извините, произошла ошибка при обращении к языковой модели."

search_service = SearchService() 