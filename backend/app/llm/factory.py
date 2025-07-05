import logging
import asyncio
import time
import threading
import concurrent.futures
from langchain_core.language_models import BaseLLM, LLM
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.gigachat import GigaChat
from langchain_openai import ChatOpenAI
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from yandex_cloud_ml_sdk import YCloudML

from app.core.config import settings

logger = logging.getLogger(__name__)

# The embedding dimension for the YandexGPT model.
YANDEX_EMBEDDING_DIMENSION = 256

# Thread pool executor для gRPC операций (изолирует от uvloop)
grpc_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, 
    thread_name_prefix="grpc_worker"
)

# Логируем инициализацию thread pool
logger.info(f"Инициализирован gRPC пул {grpc_executor._max_workers} потоков для изоляции от uvloop")

class LLMResponse:
    """Простая обертка для ответа LLM с атрибутом content"""
    
    def __init__(self, content: str):
        self.content = content
    
    def __str__(self):
        return self.content

class ChatOpenRouter(ChatOpenAI):
    """Custom OpenRouter chat model that inherits from ChatOpenAI."""
    
    def __init__(self, model_name: str, openrouter_api_key: str, **kwargs):
        super().__init__(
            model=model_name,
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            **kwargs
        )

class YandexLLMWrapper:
    """Async wrapper для Yandex LLM с thread pool"""
    
    def __init__(self, folder_id: str, api_key: str):
        self.folder_id = folder_id
        self.api_key = api_key
        self._model = None
        
    def _get_model(self):
        """Инициализация модели в thread pool"""
        if self._model is None:
            sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
            self._model = sdk.models.completions("yandexgpt").langchain()
        return self._model
    
    async def ainvoke(self, prompt: str, **kwargs):
        """Асинхронный вызов LLM через thread pool"""
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(grpc_executor, self._get_model)
        response = await loop.run_in_executor(grpc_executor, model.invoke, prompt)
        
        # Обеспечиваем совместимость: если response уже имеет content, используем его
        # иначе создаем LLMResponse
        if hasattr(response, 'content'):
            return response
        else:
            return LLMResponse(str(response))

def get_llm() -> BaseLLM:
    """
    Factory function to get the language model instance based on the provider setting.
    """
    provider = settings.LLM_PROVIDER.lower()
    logger.info(f"Инициализация LLM для провайдера: {provider}")

    if provider == "yandex":
        if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
            raise ValueError("YANDEX_API_KEY and YANDEX_FOLDER_ID must be set for Yandex provider")
        
        # Используем wrapper для работы через thread pool
        return YandexLLMWrapper(
            folder_id=settings.YANDEX_FOLDER_ID,
            api_key=settings.YANDEX_API_KEY
        )
        
    elif provider == "giga":
        if not settings.GIGACHAT_CREDENTIALS:
            raise ValueError("GIGACHAT_CREDENTIALS must be set for GigaChat provider")
        return GigaChat(credentials=settings.GIGACHAT_CREDENTIALS, verify_ssl_certs=False)
    
    elif provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter provider")
        return ChatOpenRouter(
            model_name=settings.OPENROUTER_MODEL_NAME,
            openrouter_api_key=settings.OPENROUTER_API_KEY
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_embedding_model() -> Embeddings:
    """
    Factory function to get the embedding model instance based on the provider setting.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    logger.info(f"Инициализация модели эмбеддингов для провайдера: {provider}")

    if provider == "yandex":
        if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
            raise ValueError("YANDEX_API_KEY and YANDEX_FOLDER_ID must be set for Yandex embedding provider")
        
        # Создаем обертку для embeddings с thread pool
        class YandexEmbeddings(Embeddings):
            def __init__(self, folder_id: str, api_key: str):
                self.folder_id = folder_id
                self.api_key = api_key
                self._doc_model = None
                self._query_model = None
                
                # Оптимизированный rate limiting для батчей
                self._request_times = []
                self._lock = threading.Lock()
                self._max_requests_per_second = 10
                # Burst allowance - позволяет несколько быстрых запросов подряд
                self._burst_capacity = 3
                self._burst_tokens = 3
                self._last_refill = time.time()
                
            def _init_models(self):
                """Инициализация моделей в thread pool"""
                if self._doc_model is None or self._query_model is None:
                    sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
                    self._doc_model = sdk.models.text_embeddings("doc")
                    self._query_model = sdk.models.text_embeddings("query")
                return self._doc_model, self._query_model
                
            def _wait_for_rate_limit_optimized(self, batch_size: int = 1):
                """
                Оптимизированный rate limiting с поддержкой burst requests и батчей.
                """
                with self._lock:
                    current_time = time.time()
                    
                    # Refill burst tokens
                    time_passed = current_time - self._last_refill
                    tokens_to_add = int(time_passed * self._max_requests_per_second)
                    if tokens_to_add > 0:
                        self._burst_tokens = min(self._burst_capacity, self._burst_tokens + tokens_to_add)
                        self._last_refill = current_time
                    
                    # Удаляем запросы старше 1 секунды
                    self._request_times = [
                        req_time for req_time in self._request_times 
                        if current_time - req_time < 1.0
                    ]
                    
                    # Проверяем можем ли использовать burst tokens
                    if batch_size <= self._burst_tokens and len(self._request_times) + batch_size <= self._max_requests_per_second:
                        self._burst_tokens -= batch_size
                        for _ in range(batch_size):
                            self._request_times.append(current_time)
                        return  # Можем выполнить запрос сразу
                    
                    # Стандартный rate limiting с проверкой на пустой список
                    if len(self._request_times) + batch_size > self._max_requests_per_second:
                        # Проверяем что список не пустой перед обращением к [0]
                        if self._request_times:
                            sleep_time = 1.0 - (current_time - self._request_times[0])
                            if sleep_time > 0:
                                estimated_sleep = sleep_time + (batch_size * 0.1)  # Дополнительная пауза для батча
                                logger.debug(f"Достигнут лимит запросов, пауза {estimated_sleep:.3f}с для батча {batch_size}")
                                time.sleep(estimated_sleep)
                                # Обновляем время после сна
                                current_time = time.time()
                                # Очищаем старые запросы снова
                                self._request_times = [
                                    req_time for req_time in self._request_times 
                                    if current_time - req_time < 1.0
                                ]
                        else:
                            # Если список пустой, просто ждем минимальную паузу
                            logger.debug(f"Список запросов пуст, минимальная пауза для батча {batch_size}")
                            time.sleep(batch_size * 0.1)
                            current_time = time.time()
                    
                    # Добавляем текущие запросы
                    for _ in range(batch_size):
                        self._request_times.append(current_time)
            
            def _wait_for_rate_limit(self):
                """
                Backward compatibility - обеспечивает соблюдение лимита 10 запросов в секунду.
                """
                return self._wait_for_rate_limit_optimized(1)
            
            def _sync_embed_documents(self, texts: list[str]) -> list[list[float]]:
                """Оптимизированная синхронная версия для батчей"""
                doc_model, _ = self._init_models()
                embeddings = []
                
                # Для батчей проверяем rate limit один раз для всего батча
                self._wait_for_rate_limit_optimized(len(texts))
                
                batch_start_time = time.time()
                for i, text in enumerate(texts):
                    # Не используем _wait_for_rate_limit здесь, так как уже учли батч выше
                    embedding = doc_model.run(text)
                    embeddings.append(embedding)
                    
                    if (i + 1) % 10 == 0:
                        logger.debug(f"Обработано {i + 1}/{len(texts)} эмбеддингов в батче")
                
                batch_duration = time.time() - batch_start_time        
                logger.debug(f"Завершен батч эмбеддингов {len(texts)} за {batch_duration:.2f}с "
                           f"(среднее {batch_duration/len(texts):.3f}с на элемент)")
                        
                return embeddings
            
            def _sync_embed_query(self, text: str) -> list[float]:
                """Синхронная версия для thread pool"""
                _, query_model = self._init_models()
                self._wait_for_rate_limit()
                return query_model.run(text)
            
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                """Создание эмбеддингов для документов (синхронный интерфейс)."""
                # Это будет вызываться из asyncio.to_thread в ingestion_service
                return self._sync_embed_documents(texts)
            
            def embed_query(self, text: str) -> list[float]:
                """Создание эмбеддинга для запроса (синхронный интерфейс)."""
                # Это будет вызываться из asyncio.to_thread в search_service
                return self._sync_embed_query(text)
            
            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                """Асинхронное создание эмбеддингов для документов."""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(grpc_executor, self._sync_embed_documents, texts)
            
            async def aembed_query(self, text: str) -> list[float]:
                """Асинхронное создание эмбеддинга для запроса."""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(grpc_executor, self._sync_embed_query, text)
        
        return YandexEmbeddings(
            folder_id=settings.YANDEX_FOLDER_ID,
            api_key=settings.YANDEX_API_KEY,
        )
    # Add other embedding providers here in the future
    # elif provider == "sbert":
    #     return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

# Initialize models on startup
llm = get_llm()
embedding_model = get_embedding_model() 