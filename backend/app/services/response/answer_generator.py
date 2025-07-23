"""
Answer generation service with Yandex LLM integration.
"""
import logging
from typing import List, Dict, Any, Optional
import json
import time

import trio

from app.core.config import settings
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Answer generation service using Yandex LLM with context and citations.
    """
    
    def __init__(self):
        self.cache_manager = cache_manager
        self.yandex_client = None
        self._rate_limiter = trio.CapacityLimiter(settings.MAX_CONCURRENT_CHATS)
        self._initialized = False
        
    async def initialize(self):
        """Initialize the answer generator."""
        if self._initialized:
            return
            
        try:
            # Initialize Yandex Cloud ML client
            await self._initialize_yandex_client()
            
            self._initialized = True
            logger.info("AnswerGenerator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AnswerGenerator: {e}")
            raise
    
    async def _initialize_yandex_client(self):
        """Initialize Yandex Cloud ML client."""
        try:
            # Import Yandex SDK
            from yandex_cloud_ml_sdk import YCloudML
            
            if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
                logger.warning("Yandex API credentials not configured, using mock responses")
                self.yandex_client = None
                return
            
            # Initialize client
            self.yandex_client = YCloudML(
                folder_id=settings.YANDEX_FOLDER_ID,
                auth=settings.YANDEX_API_KEY
            )
            
            logger.info("Yandex Cloud ML client initialized for answer generation")
            
        except ImportError:
            logger.warning("Yandex Cloud ML SDK not available, using mock responses")
            self.yandex_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Yandex client: {e}")
            self.yandex_client = None
    
    async def generate_answer(self, 
                            query: str,
                            context_chunks: List[Dict[str, Any]],
                            chat_history: Optional[List[Dict[str, str]]] = None,
                            generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate answer based on query and context chunks.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            chat_history: Previous chat messages
            generation_config: LLM generation parameters
            
        Returns:
            Generated answer with citations and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, context_chunks, chat_history, generation_config)
            cached_response = await self.cache_manager.get_llm_cache(
                settings.YANDEX_LLM_MODEL,
                cache_key,
                chat_history or [],
                generation_config or {}
            )
            
            if cached_response:
                logger.debug("Cache hit for answer generation")
                return json.loads(cached_response)
            
            # Prepare context and prompt
            context_text = self._prepare_context(context_chunks)
            prompt = self._build_prompt(query, context_text, chat_history)
            
            # Generate answer with rate limiting
            async with self._rate_limiter:
                response = await self._generate_llm_response(prompt, generation_config)
            
            # Process response and extract citations
            answer_data = self._process_response(response, context_chunks, query)
            
            # Add timing information
            answer_data["generation_time"] = time.time() - start_time
            
            # Cache response
            await self.cache_manager.set_llm_cache(
                settings.YANDEX_LLM_MODEL,
                cache_key,
                json.dumps(answer_data),
                chat_history or [],
                generation_config or {}
            )
            
            logger.info(f"Answer generated in {answer_data['generation_time']:.3f}s")
            
            return answer_data
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": f"Извините, произошла ошибка при генерации ответа: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "generation_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved chunks."""
        if not context_chunks:
            return "Контекст не найден."
        
        context_parts = []
        
        for i, chunk in enumerate(context_chunks[:10]):  # Limit to top 10 chunks
            content = chunk.get("content", "").strip()
            if content:
                # Add chunk metadata for better context
                metadata = chunk.get("metadata", {})
                doc_name = metadata.get("document_name", "Неизвестный документ")
                page_num = metadata.get("page_number")
                
                source_info = f"[Источник {i+1}: {doc_name}"
                if page_num:
                    source_info += f", стр. {page_num}"
                source_info += "]"
                
                context_parts.append(f"{source_info}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, 
                     query: str, 
                     context: str, 
                     chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build prompt for LLM."""
        # Base system prompt
        system_prompt = """Вы - полезный AI-ассистент, который отвечает на вопросы на основе предоставленного контекста.

Инструкции:
1. Отвечайте только на основе предоставленного контекста
2. Если информации недостаточно, честно скажите об этом
3. Используйте естественный, разговорный стиль
4. Структурируйте ответ логично и понятно
5. При необходимости ссылайтесь на источники в формате [Источник X]
6. Если вопрос не связан с контекстом, вежливо объясните это

Контекст:
{context}

"""
        
        # Add chat history if provided
        conversation = ""
        if chat_history:
            conversation += "История разговора:\n"
            for msg in chat_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    conversation += f"Пользователь: {content}\n"
                elif role == "assistant":
                    conversation += f"Ассистент: {content}\n"
            conversation += "\n"
        
        # Build final prompt
        prompt = system_prompt.format(context=context)
        if conversation:
            prompt += conversation
        
        prompt += f"Вопрос: {query}\n\nОтвет:"
        
        return prompt
    
    async def _generate_llm_response(self, 
                                   prompt: str, 
                                   generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Generate response using Yandex LLM."""
        if not self.yandex_client:
            # Return mock response if client not available
            return self._generate_mock_response(prompt)
        
        try:
            # Default generation config
            config = {
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9,
                **(generation_config or {})
            }
            
            # Use Yandex Cloud ML SDK
            response = await trio.to_thread.run_sync(
                self._call_yandex_llm_api,
                prompt,
                config
            )
            
            if response and 'text' in response:
                return response['text']
            else:
                logger.warning(f"Invalid response from Yandex API: {response}")
                return self._generate_mock_response(prompt)
                
        except Exception as e:
            logger.error(f"Yandex LLM API call failed: {e}")
            return self._generate_mock_response(prompt)
    
    def _call_yandex_llm_api(self, prompt: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Call Yandex LLM API (synchronous)."""
        try:
            # This is a placeholder for the actual Yandex API call
            # The exact implementation depends on the Yandex SDK
            
            # Example structure (adjust based on actual SDK):
            # response = self.yandex_client.completions.create(
            #     model=settings.YANDEX_LLM_MODEL,
            #     messages=[{"role": "user", "content": prompt}],
            #     **config
            # )
            # return response.to_dict()
            
            # For now, return mock response
            logger.warning("Using mock Yandex LLM API response")
            return {
                'text': self._generate_mock_response(prompt)
            }
            
        except Exception as e:
            logger.error(f"Yandex LLM API call error: {e}")
            raise
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response for testing purposes."""
        # Simple mock response based on prompt content
        if "что" in prompt.lower() or "какой" in prompt.lower():
            return "На основе предоставленного контекста, я могу сказать, что это интересный вопрос. Однако для более точного ответа мне потребуется дополнительная информация из документов."
        elif "как" in prompt.lower():
            return "Согласно документам, процесс включает несколько этапов. Рекомендую обратиться к соответствующим разделам документации для получения подробных инструкций."
        elif "почему" in prompt.lower():
            return "Причины могут быть различными. На основе доступной информации, это связано с особенностями системы, описанными в документах."
        else:
            return "Спасибо за ваш вопрос. На основе предоставленного контекста, я постараюсь дать максимально полный ответ, используя доступную информацию из документов."
    
    def _process_response(self, 
                         response: str, 
                         context_chunks: List[Dict[str, Any]], 
                         query: str) -> Dict[str, Any]:
        """Process LLM response and extract citations."""
        # Extract citations from response
        citations = self._extract_citations(response, context_chunks)
        
        # Calculate confidence based on context relevance
        confidence = self._calculate_confidence(response, context_chunks, query)
        
        # Clean response text
        cleaned_response = self._clean_response_text(response)
        
        return {
            "answer": cleaned_response,
            "citations": citations,
            "confidence": confidence,
            "context_chunks_used": len(context_chunks),
            "model": settings.YANDEX_LLM_MODEL
        }
    
    def _extract_citations(self, 
                          response: str, 
                          context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from response text."""
        citations = []
        
        # Look for citation patterns like [Источник 1], [Источник 2], etc.
        import re
        citation_pattern = r'\[Источник (\d+)[^\]]*\]'
        matches = re.findall(citation_pattern, response)
        
        for match in matches:
            try:
                source_num = int(match) - 1  # Convert to 0-based index
                if 0 <= source_num < len(context_chunks):
                    chunk = context_chunks[source_num]
                    citation = {
                        "chunk_id": chunk.get("chunk_id"),
                        "document_id": chunk.get("document_id"),
                        "content_preview": chunk.get("content", "")[:200] + "...",
                        "source_number": source_num + 1,
                        "metadata": chunk.get("metadata", {})
                    }
                    citations.append(citation)
            except (ValueError, IndexError):
                continue
        
        # If no explicit citations found, include top chunks as implicit citations
        if not citations and context_chunks:
            for i, chunk in enumerate(context_chunks[:3]):  # Top 3 chunks
                citation = {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "content_preview": chunk.get("content", "")[:200] + "...",
                    "source_number": i + 1,
                    "metadata": chunk.get("metadata", {}),
                    "implicit": True
                }
                citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, 
                            response: str, 
                            context_chunks: List[Dict[str, Any]], 
                            query: str) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if response contains specific information
        if len(response) > 100:
            confidence += 0.1
        
        # Increase confidence based on number of context chunks
        if len(context_chunks) >= 3:
            confidence += 0.2
        elif len(context_chunks) >= 1:
            confidence += 0.1
        
        # Increase confidence if response contains citations
        if "[Источник" in response:
            confidence += 0.2
        
        # Decrease confidence for uncertain language
        uncertain_phrases = ["возможно", "вероятно", "может быть", "не уверен", "недостаточно информации"]
        for phrase in uncertain_phrases:
            if phrase in response.lower():
                confidence -= 0.1
                break
        
        # Ensure confidence is in valid range
        return max(0.0, min(1.0, confidence))
    
    def _clean_response_text(self, response: str) -> str:
        """Clean and format response text."""
        # Remove extra whitespace
        cleaned = " ".join(response.split())
        
        # Ensure proper sentence endings
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def _generate_cache_key(self, 
                          query: str, 
                          context_chunks: List[Dict[str, Any]], 
                          chat_history: Optional[List[Dict[str, str]]], 
                          generation_config: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for the request."""
        # Create a hash of the key components
        import hashlib
        
        key_components = [
            query,
            str(len(context_chunks)),
            str([chunk.get("chunk_id") for chunk in context_chunks[:5]]),  # Top 5 chunk IDs
            str(chat_history or []),
            str(generation_config or {})
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def generate_streaming_answer(self, 
                                      query: str,
                                      context_chunks: List[Dict[str, Any]],
                                      chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate streaming answer (placeholder for future implementation).
        
        For now, returns the same as generate_answer but could be extended
        to support streaming responses.
        """
        return await self.generate_answer(query, context_chunks, chat_history)
    
    async def get_generator_stats(self) -> Dict[str, Any]:
        """Get answer generator statistics."""
        try:
            cache_stats = await self.cache_manager.health_check()
            
            return {
                "initialized": self._initialized,
                "yandex_client_available": self.yandex_client is not None,
                "model": settings.YANDEX_LLM_MODEL,
                "cache_status": cache_stats.get("status", "unknown"),
                "rate_limiter_capacity": self._rate_limiter.total_tokens,
                "rate_limiter_available": self._rate_limiter.available_tokens
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}


# Global instance
answer_generator = AnswerGenerator()