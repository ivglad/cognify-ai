import logging
import asyncio
import time
import threading
import concurrent.futures
from typing import List, Iterator, Any
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
logger.info(f"Initialized gRPC thread pool with {grpc_executor._max_workers} workers to isolate from uvloop")

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
    logger.info(f"Initializing LLM for provider: {provider}")

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
    logger.info(f"Initializing embedding model for provider: {provider}")

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
                
                # Rate limiting: максимум 10 запросов в секунду
                self._request_times = []
                self._lock = threading.Lock()
                self._max_requests_per_second = 10
                
            def _init_models(self):
                """Инициализация моделей в thread pool"""
                if self._doc_model is None or self._query_model is None:
                    sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
                    self._doc_model = sdk.models.text_embeddings("doc")
                    self._query_model = sdk.models.text_embeddings("query")
                return self._doc_model, self._query_model
                
            def _wait_for_rate_limit(self):
                """
                Обеспечивает соблюдение лимита 10 запросов в секунду.
                """
                with self._lock:
                    current_time = time.time()
                    
                    # Удаляем запросы старше 1 секунды
                    self._request_times = [
                        req_time for req_time in self._request_times 
                        if current_time - req_time < 1.0
                    ]
                    
                    # Если достигнут лимит, ждем
                    if len(self._request_times) >= self._max_requests_per_second:
                        sleep_time = 1.0 - (current_time - self._request_times[0])
                        if sleep_time > 0:
                            logger.debug(f"Rate limit reached, sleeping for {sleep_time:.3f} seconds")
                            time.sleep(sleep_time)
                            # Обновляем время после сна
                            current_time = time.time()
                            # Очищаем старые запросы снова
                            self._request_times = [
                                req_time for req_time in self._request_times 
                                if current_time - req_time < 1.0
                            ]
                    
                    # Добавляем текущий запрос
                    self._request_times.append(current_time)
            
            def _sync_embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Синхронная версия для thread pool"""
                doc_model, _ = self._init_models()
                embeddings = []
                for i, text in enumerate(texts):
                    self._wait_for_rate_limit()
                    embedding = doc_model.run(text)
                    embeddings.append(embedding)
                    
                    if (i + 1) % 5 == 0:
                        logger.debug(f"Processed {i + 1}/{len(texts)} embeddings")
                        
                return embeddings
            
            def _sync_embed_query(self, text: str) -> List[float]:
                """Синхронная версия для thread pool"""
                _, query_model = self._init_models()
                self._wait_for_rate_limit()
                return query_model.run(text)
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Создание эмбеддингов для документов (синхронный интерфейс)."""
                # Это будет вызываться из asyncio.to_thread в ingestion_service
                return self._sync_embed_documents(texts)
            
            def embed_query(self, text: str) -> List[float]:
                """Создание эмбеддинга для запроса (синхронный интерфейс)."""
                # Это будет вызываться из asyncio.to_thread в search_service
                return self._sync_embed_query(text)
            
            async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
                """Асинхронное создание эмбеддингов для документов."""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(grpc_executor, self._sync_embed_documents, texts)
            
            async def aembed_query(self, text: str) -> List[float]:
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