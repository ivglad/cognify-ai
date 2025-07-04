from app.core.config import settings
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.llm.factory import llm, embedding_model
from app.db.infinity_client import infinity_client
from infinity.index import IndexInfo, IndexType
import asyncio
import logging
import time
from typing import List, Tuple
import uuid

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from app.services.kg_service import kg_service

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.embedding_model = embedding_model
        # Используем централизованный Infinity клиент
        logger.info("SearchService using centralized Infinity client")
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading missing NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        try:
            self.russian_stopwords = set(stopwords.words("russian"))
        except LookupError:
            logger.warning("Russian stopwords not available, using empty set")
            self.russian_stopwords = set()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalizes text for BM25 search.
        """
        try:
            tokens = word_tokenize(text.lower(), language='russian')
            return " ".join([token for token in tokens if token.isalnum() and token not in self.russian_stopwords])
        except LookupError as e:
            logger.error(f"NLTK tokenization failed: {e}. Falling back to simple split.")
            # Fallback к простому разделению по словам
            simple_tokens = text.lower().split()
            return " ".join([token for token in simple_tokens if token.isalnum() and token not in self.russian_stopwords])
    
    async def _condense_query_with_history(self, query: str, history: List[ChatMessage]) -> str:
        """
        Condenses chat history and a new query into a standalone query.
        """
        if not history:
            return query

        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        logger.info(f"Condensing query with history: {history_str}")

        prompt = f"""Учитывая историю чата и новый вопрос, переформулируй новый вопрос так, чтобы он был самодостаточным.
        
        История чата:
        {history_str}
        
        Новый вопрос: {query}
        
        Самодостаточный вопрос:"""

        try:
            condensed_query = await llm.ainvoke(prompt)
            logger.info(f"Condensed query: '{condensed_query.content.strip()}'")
            return condensed_query.content.strip()
        except Exception as e:
            logger.error(f"Failed to condense query: {e}", exc_info=True)
            return query # Fallback to original query

    async def _rewrite_query(self, query: str) -> str:
        """
        Rewrites the user's query using an LLM for better retrieval.
        """
        logger.info(f"Rewriting query: '{query}'")
        
        prompt = f"""Переформулируй следующий вопрос для максимально эффективного поиска в векторной базе данных. 
        Сохрани ключевые слова и основной смысл. Ответь только переформулированным вопросом.
        
        Оригинальный вопрос: {query}
        
        Переформулированный вопрос:"""
        
        try:
            rewritten_query = await llm.ainvoke(prompt)
            # Clean up the response, as the model might add extra text
            rewritten_query = rewritten_query.content.strip()
            logger.info(f"Rewritten query: '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}", exc_info=True)
            return query # Fallback to original query

    async def _rerank_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """
        Re-ranks a list of chunks based on their relevance to the query using an LLM.
        """
        if not chunks:
            return []

        logger.info(f"Re-ranking {len(chunks)} chunks for query: '{query}'")

        # Create a numbered list of chunks for the prompt
        numbered_chunks = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])

        prompt = f"""Ниже приведен запрос пользователя и список извлеченных из документов фрагментов текста.
Твоя задача — оценить, насколько каждый фрагмент релевантен для ответа на запрос.
Верни только номера наиболее релевантных фрагментов, отсортированные по убыванию релевантности, в виде списка через запятую. Например: 3, 1, 5.
Не включай в ответ нерелевантные фрагменты. Если релевантных фрагментов нет, верни пустую строку.

Запрос: {query}

Фрагменты:
{numbered_chunks}

Отсортированный список номеров релевантных фрагментов:"""

        try:
            response = await llm.ainvoke(prompt)
            response_text = response.content.strip()
            logger.info(f"LLM re-ranking response: '{response_text}'")

            if not response_text:
                return []

            # Parse the response
            relevant_indices = [int(i.strip()) - 1 for i in response_text.split(',') if i.strip().isdigit()]

            # Create the re-ranked list of chunks
            reranked_chunks = [chunks[i] for i in relevant_indices if 0 <= i < len(chunks)]

            if not reranked_chunks:
                logger.warning("Re-ranking returned no chunks, falling back to original order.")
                return chunks

            logger.info(f"Re-ranked and kept {len(reranked_chunks)} chunks.")
            return reranked_chunks
        except Exception as e:
            logger.error(f"Failed to re-rank chunks: {e}", exc_info=True)
            return chunks # Fallback to original fused list

    async def search(self, request: ChatRequest) -> Tuple[str, List[ChatMessage]]:
        """
        Performs a hybrid search and returns an answer and the updated history.
        """
        start_time = time.monotonic()
        original_query = request.query
        history = request.history
        logger.info(f"Searching for: '{original_query}' in documents: {request.document_ids} with history.")

        # 0. Condense query with history
        condensed_query = await self._condense_query_with_history(original_query, history)

        # 1. Rewrite the condensed query for better retrieval
        if settings.ENABLE_QUERY_REWRITING:
            rewritten_query = await self._rewrite_query(condensed_query)
        else:
            logger.info("Query rewriting is disabled.")
            rewritten_query = condensed_query

        # 2. Embed the rewritten query
        if hasattr(self.embedding_model, 'aembed_query'):
            query_embedding = await self.embedding_model.aembed_query(rewritten_query)
        else:
            # Fallback для не-Yandex провайдеров
            query_embedding = await asyncio.to_thread(self.embedding_model.embed_query, rewritten_query)
        # Преобразуем в double для совместимости с Infinity
        query_embedding_double = [float(x) for x in query_embedding]

        # 3. Find relevant chunks using hybrid search
        fused_chunks = await asyncio.to_thread(
            self._find_relevant_chunks,
            rewritten_query,
            query_embedding_double,
            request.document_ids
        )

        # 3.5 Re-rank the fused chunks using the original query for relevance
        reranked_chunks = await self._rerank_chunks(original_query, fused_chunks)

        # 4. Generate a response using the LLM with full context
        related_entities = await kg_service.find_related_entities(condensed_query)
        kg_context = ""
        if related_entities:
            kg_context = f"Найдены связанные сущности в базе знаний: {', '.join(related_entities)}."

        answer = await self._generate_llm_response(original_query, reranked_chunks, history, kg_context)

        # 5. Update history
        updated_history = history + [
            ChatMessage(role="user", content=original_query),
            ChatMessage(role="assistant", content=answer)
        ]

        end_time = time.monotonic()
        logger.info(f"Search request for '{original_query}' completed in {end_time - start_time:.2f} seconds.")
        return answer, updated_history

    def _find_relevant_chunks(self, query: str, query_embedding: List[float], document_ids: List[uuid.UUID]) -> list[str]:
        """
        Finds relevant text chunks from the single collection using a hybrid search.
        """
        logger.info("Performing hybrid search in 'chunks_collection'...")
        
        # Диагностика перед поиском
        available_tables = infinity_client.list_tables("default")
        logger.info(f"Available tables in database before search: {available_tables}")
        
        table_name = "chunks_collection"
        table_exists = infinity_client.table_exists(table_name, "default")
        logger.info(f"Table '{table_name}' exists before search: {table_exists}")
        
        if table_exists:
            table_info = infinity_client.get_table_info(table_name, "default")
            logger.info(f"Table '{table_name}' info before search: {table_info}")
        
        db = infinity_client.get_database("default")
        try:
            table = db.get_table("chunks_collection")
            logger.info(f"Successfully obtained table '{table_name}' for search")
            
            # Диагностика индексов для полнотекстового поиска
            try:
                indexes_response = table.list_indexes()
                logger.info(f"Available indexes on table '{table_name}': {indexes_response}")
                index_names = getattr(indexes_response, 'index_names', [])
                fulltext_index_exists = any("fulltext_idx" in str(idx) for idx in index_names)
                logger.info(f"Fulltext index exists: {fulltext_index_exists}")
                
                # Принудительно создаем индекс если его нет
                if not fulltext_index_exists:
                    logger.info("Creating missing fulltext index in search_service...")
                    try:
                        table.create_index(
                            "fulltext_idx",
                            IndexInfo(
                                "chunk_text_normalized",
                                IndexType.FullText
                            )
                        )
                        logger.info("Successfully created fulltext index in search_service")
                    except Exception as create_error:
                        logger.error(f"Failed to create fulltext index in search_service: {create_error}")
                        
            except Exception as idx_error:
                logger.warning(f"Could not check indexes: {idx_error}")
                
        except Exception as e:
            logger.error(f"Could not get 'chunks_collection' table. Error: {e}")
            logger.error(f"Available tables: {available_tables}")
            logger.error(f"Table exists check: {table_exists}")
            
            # Дополнительная диагностика - попробуем переподключиться
            try:
                logger.info("Attempting to reconnect to Infinity client...")
                infinity_client.reconnect()
                
                # Получаем новое подключение к базе данных
                fresh_db = infinity_client.get_database("default")
                fresh_tables = infinity_client.list_tables("default")
                logger.info(f"Tables after reconnection: {fresh_tables}")
                
                # Пробуем получить таблицу снова
                table = fresh_db.get_table("chunks_collection")
                logger.info("Successfully obtained table after reconnection")
            except Exception as e2:
                logger.error(f"Reconnection also failed: {e2}")
                return []
        
        normalized_query = self._normalize_text(query)
        logger.info(f"Original query: '{query}', Normalized query: '{normalized_query}'")

        # Гибридный поиск: векторный + полнотекстовый
        query_builder = table.output(['chunk_text', '_score']) \
                           .knn('embedding', query_embedding, 'double', 'cosine', 10) \
                           .match_text('chunk_text_normalized', normalized_query, 10)

        # Add filter if document_ids are specified
        if document_ids:
            # Ensure document_ids are properly formatted for a SQL 'IN' clause
            doc_id_list = ", ".join([f"'{str(doc_id)}'" for doc_id in document_ids])
            filter_condition = f"document_id IN ({doc_id_list})"
            query_builder = query_builder.filter(filter_condition)
            logger.info(f"Applying filter: {filter_condition}")

        # Add fusion and execute
        try:
            # Используем RRF (Reciprocal Rank Fusion)
            res_df_list = query_builder.fusion('rrf', 10).to_df()
            logger.info("Hybrid search executed successfully")
        except Exception as e:
            logger.error(f"Hybrid search failed on 'chunks_collection': {e}", exc_info=True)
            # Выполняем векторный поиск как fallback
            try:
                logger.info("Trying fallback to vector-only search...")
                # Согласно документации Infinity, _score доступен только с fusion/match_text/match_tensor
                fallback_result = table.output(['chunk_text']) \
                                      .knn('embedding', query_embedding, 'double', 'cosine', 10) \
                                      .to_df()
                if fallback_result and len(fallback_result) > 0:
                    logger.info("Fallback vector search succeeded")
                    final_chunks = fallback_result[0]['chunk_text'].tolist()
                    logger.info(f"Found {len(final_chunks)} relevant chunks using vector-only fallback.")
                    return final_chunks
                else:
                    logger.warning("Fallback vector search returned no results")
                    return []
            except Exception as fallback_error:
                logger.error(f"Fallback vector search also failed: {fallback_error}")
                return []

        if not res_df_list or res_df_list[0].empty:
            logger.warning("Hybrid search returned no chunks from 'chunks_collection'.")
            return []
        
        # The result is already sorted by the fusion score
        final_chunks = res_df_list[0]['chunk_text'].tolist()
        
        logger.info(f"Found {len(final_chunks)} relevant chunks from 'chunks_collection'.")
        return final_chunks

    async def _generate_llm_response(self, query: str, context_chunks: list[str], history: List[ChatMessage], kg_context: str) -> str:
        """
        Generates a response using the YandexGPT LLM, including chat history.
        """
        
        if not context_chunks and not kg_context:
            return "К сожалению, я не нашел релевантной информации попробуйте уточнить вопрос."

        context = "\n---\n".join(context_chunks) if context_chunks else "Нет данных из документов."
        history_str = "\n".join([f"{msg.role}: {msg.content}" for msg in history])
        
        logger.info(f"LLM context: {context}")
        logger.info(f"LLM history: {history_str}")

        prompt_template = f"""Ты — умный ИИ-ассистент. Используй историю чата, контекст из документов и, если есть, данные из базы знаний, чтобы ответить на вопрос.
        Если ты не знаешь ответа, так и скажи: "Я не нашел информации в документах". Не придумывай ответ. Формулировки должны быть краткими и понятными, на русском языке.

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
            logger.info(f"LLM response: {response}")
            return response.content
        except Exception as e:
            logger.error(f"Error calling YandexGPT: {e}", exc_info=True)
            return "Извините, произошла ошибка при обращении к языковой модели."

search_service = SearchService() 