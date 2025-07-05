import logging
import threading
import time
import asyncio
import concurrent.futures
from yandex_cloud_ml_sdk import YCloudML
from app.core.config import settings

logger = logging.getLogger(__name__)

# Thread pool executor для gRPC операций токенизации
tokenizer_grpc_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, 
    thread_name_prefix="tokenizer_grpc_worker"
)

class TokenizerService:
    """
    Сервис для точного подсчета токенов с использованием Yandex Cloud ML SDK.
    Поддерживает как русский, так и английский язык.
    """
    
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        
        # Rate limiting для токенизации (консервативный подход)
        self._request_times = []
        self._max_requests_per_second = 5  # Консервативный лимит для токенизации
        
        if settings.EMBEDDING_PROVIDER.lower() != "yandex":
            logger.warning("TokenizerService инициализирован, но EMBEDDING_PROVIDER не yandex. Будет использоваться fallback.")
            
    def _init_model(self):
        """Инициализация модели токенизации"""
        if self._model is None:
            if settings.EMBEDDING_PROVIDER.lower() == "yandex":
                if not settings.YANDEX_API_KEY or not settings.YANDEX_FOLDER_ID:
                    raise ValueError("YANDEX_API_KEY и YANDEX_FOLDER_ID должны быть установлены для Yandex tokenizer")
                
                sdk = YCloudML(folder_id=settings.YANDEX_FOLDER_ID, auth=settings.YANDEX_API_KEY)
                # Используем модель для токенизации YandexGPT
                self._model = sdk.models.completions("yandexgpt")
                logger.info("Yandex модель токенизации инициализирована успешно")
            else:
                logger.warning("Yandex tokenizer не может быть инициализирован без yandex provider")
        return self._model
    
    def _wait_for_rate_limit(self):
        """Rate limiting для запросов токенизации"""
        with self._lock:
            current_time = time.time()
            
            # Удаляем запросы старше 1 секунды
            self._request_times = [
                req_time for req_time in self._request_times 
                if current_time - req_time < 1.0
            ]
            
            # Проверяем лимит
            if len(self._request_times) >= self._max_requests_per_second:
                if self._request_times:
                    sleep_time = 1.0 - (current_time - self._request_times[0])
                    if sleep_time > 0:
                        logger.debug(f"Достигнут лимит токенизации, пауза {sleep_time:.3f} сек")
                        time.sleep(sleep_time)
                        current_time = time.time()
                        # Очищаем старые запросы снова
                        self._request_times = [
                            req_time for req_time in self._request_times 
                            if current_time - req_time < 1.0
                        ]
            
            # Добавляем текущий запрос
            self._request_times.append(current_time)
    
    def _sync_count_tokens(self, text: str) -> int:
        """Синхронный подсчет токенов через Yandex API с защитой от зависания"""
        # Защита от слишком больших текстов
        if len(text) > 20000:  # Более 20k символов
            logger.warning(f"Текст слишком большой для точного подсчета ({len(text)} символов), используем fallback")
            return self._estimate_token_count_fallback(text)
            
        try:
            model = self._init_model()
            if model is None:
                # Fallback к приблизительному подсчету
                return self._estimate_token_count_fallback(text)
            
            self._wait_for_rate_limit()
            
            # Логируем начало подсчета для отладки
            logger.debug(f"Начинаем точный подсчет токенов для текста длиной {len(text)} символов")
            
            # Используем метод tokenize для подсчета токенов
            # API ожидает строку или список сообщений
            result = model.tokenize(text)
            
            # result уже является итерируемым списком токенов
            # Подсчитываем количество токенов
            if hasattr(result, '__iter__'):
                token_list = list(result)
                token_count = len(token_list)
            else:
                # На случай если API изменится
                token_count = len(result)
            
            logger.debug(f"Точный подсчет токенов завершен: {token_count} для текста длиной {len(text)} символов")
            return token_count
            
        except Exception as e:
            logger.warning(f"Ошибка при точном подсчете токенов (текст: {len(text)} символов): {e}. Используем fallback.")
            return self._estimate_token_count_fallback(text)
    
    def _estimate_token_count_fallback(self, text: str) -> int:
        """
        Fallback-метод для приблизительного подсчета токенов.
        Улучшенная версия с учетом особенностей русского и английского языков.
        """
        if not text:
            return 0
        
        # Подсчет символов разных типов
        cyrillic_count = sum(1 for char in text if 'а' <= char.lower() <= 'я')
        latin_count = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        space_count = text.count(' ')
        other_count = len(text) - cyrillic_count - latin_count - space_count
        
        # Более точные коэффициенты для разных типов символов
        # Русские символы обычно требуют больше токенов
        estimated_tokens = (
            cyrillic_count * 0.5 +  # Русские символы: ~2 символа на токен
            latin_count * 0.4 +     # Английские символы: ~2.5 символа на токен
            other_count * 0.3 +     # Знаки препинания и числа
            space_count * 0.1       # Пробелы
        )
        
        # Дополнительная корректировка для безопасности
        # Увеличиваем оценку на 20% для компенсации неточности
        safety_margin = int(estimated_tokens * 1.2)
        
        return max(1, safety_margin)  # Минимум 1 токен
    
    async def count_tokens(self, text: str) -> int:
        """Асинхронный подсчет токенов с timeout"""
        if not text or not text.strip():
            return 0
        
        try:
            loop = asyncio.get_event_loop()
            # Добавляем timeout для предотвращения зависания
            return await asyncio.wait_for(
                loop.run_in_executor(
                    tokenizer_grpc_executor, 
                    self._sync_count_tokens, 
                    text
                ),
                timeout=10.0  # 10 секунд максимум на подсчет токенов
            )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout при подсчете токенов для текста длиной {len(text)} символов, используем fallback")
            return self._estimate_token_count_fallback(text)
        except Exception as e:
            logger.error(f"Ошибка при асинхронном подсчете токенов: {e}")
            return self._estimate_token_count_fallback(text)
    
    def count_tokens_sync(self, text: str) -> int:
        """Синхронный интерфейс для подсчета токенов"""
        if not text or not text.strip():
            return 0
        
        return self._sync_count_tokens(text)
    
    async def validate_token_limit(self, text: str, max_tokens: int = 2048) -> tuple[bool, int]:
        """
        Проверяет, не превышает ли текст лимит токенов.
        Возвращает (is_valid, actual_token_count)
        """
        token_count = await self.count_tokens(text)
        is_valid = token_count <= max_tokens
        return is_valid, token_count
    
    async def split_text_by_tokens(self, text: str, max_tokens: int = 1800) -> list[str]:
        """
        Разбивает текст на части, не превышающие лимит токенов.
        Использует точный подсчет токенов с защитой от зависания.
        """
        if not text or not text.strip():
            return []
        
        # Защита от слишком больших текстов - используем fallback
        if len(text) > 50000:  # Более 50k символов
            logger.warning(f"Текст слишком большой ({len(text)} символов), используем приблизительное разбиение")
            return await self._split_text_approximate(text, max_tokens)
        
        # Timeout для всего процесса разбиения
        try:
            return await asyncio.wait_for(
                self._split_text_with_exact_tokens(text, max_tokens),
                timeout=30.0  # 30 секунд максимум на разбиение
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout при точном разбиении текста ({len(text)} символов), используем fallback")
            return await self._split_text_approximate(text, max_tokens)
        except Exception as e:
            logger.error(f"Ошибка при точном разбиении текста: {e}, используем fallback")
            return await self._split_text_approximate(text, max_tokens)
    
    async def _split_text_with_exact_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Точное разбиение текста с подсчетом токенов"""
        # Сначала проверяем, нужно ли разбивать
        is_valid, token_count = await self.validate_token_limit(text, max_tokens)
        if is_valid:
            return [text]
        
        logger.info(f"Разбиваем текст с {token_count} токенами на части (макс {max_tokens} токенов)")
        
        # Ограничиваем количество итераций чтобы избежать зацикливания
        max_iterations = 1000
        iteration_count = 0
        
        # Разбиваем по предложениям
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            iteration_count += 1
            if iteration_count > max_iterations:
                logger.warning(f"Достигнут лимит итераций при разбиении текста, прерываем")
                break
                
            # Проверяем, помещается ли предложение в текущий чанк
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            potential_chunk = potential_chunk.strip()
            
            try:
                # Добавляем timeout для каждой проверки токенов
                is_valid, _ = await asyncio.wait_for(
                    self.validate_token_limit(potential_chunk, max_tokens),
                    timeout=5.0  # 5 секунд на проверку
                )
                
                if is_valid:
                    current_chunk = potential_chunk
                else:
                    # Сохраняем текущий чанк если он не пустой
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Проверяем предложение отдельно
                    is_sentence_valid, sentence_tokens = await asyncio.wait_for(
                        self.validate_token_limit(sentence, max_tokens),
                        timeout=5.0
                    )
                    
                    if is_sentence_valid:
                        current_chunk = sentence
                    else:
                        # Если предложение слишком большое, разбиваем его или обрезаем
                        if len(sentence) < 10000:  # Разбиваем только если не слишком большое
                            word_chunks = await self._split_sentence_by_words(sentence, max_tokens)
                            chunks.extend(word_chunks)
                        else:
                            # Обрезаем слишком большое предложение
                            logger.warning(f"Предложение слишком большое ({len(sentence)} символов), обрезаем")
                            truncated = sentence[:5000]  # Обрезаем до 5k символов
                            chunks.append(truncated)
                        current_chunk = ""
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout при проверке предложения {i}, пропускаем")
                continue
            except Exception as e:
                logger.warning(f"Ошибка при проверке предложения {i}: {e}, пропускаем")
                continue
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Фильтруем пустые чанки
        result_chunks = [chunk for chunk in chunks if chunk.strip()]
        
        logger.info(f"Текст разбит на {len(result_chunks)} частей с соблюдением лимита токенов")
        return result_chunks
    
    async def _split_text_approximate(self, text: str, max_tokens: int) -> list[str]:
        """Приблизительное разбиение текста без API-вызовов"""
        logger.info(f"Используем приблизительное разбиение для текста длиной {len(text)} символов")
        
        # Приблизительный размер чанка в символах (безопасная оценка)
        # Для русского текста: ~2.5 символа на токен, добавляем запас
        approx_chunk_size = max_tokens * 2  # Консервативная оценка
        
        chunks = []
        sentences = self._split_into_sentences(text)
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= approx_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Если предложение само по себе большое, разбиваем по словам
                if len(sentence) > approx_chunk_size:
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + " " + word) <= approx_chunk_size:
                            temp_chunk = temp_chunk + " " + word if temp_chunk else word
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        result_chunks = [chunk for chunk in chunks if chunk.strip()]
        logger.info(f"Приблизительное разбиение создало {len(result_chunks)} частей")
        return result_chunks
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Разбивает текст на предложения"""
        # Улучшенное разбиение по предложениям с учетом особенностей русского языка
        import re
        
        # Паттерн для разбиения по предложениям
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Восстанавливаем знаки препинания
        result = []
        for i, sentence in enumerate(sentences[:-1]):
            # Находим знак препинания после предложения
            end_pos = text.find(sentence) + len(sentence)
            while end_pos < len(text) and text[end_pos] in '.!? ':
                sentence += text[end_pos]
                end_pos += 1
            result.append(sentence.strip())
        
        # Добавляем последнее предложение
        if sentences[-1].strip():
            result.append(sentences[-1].strip())
        
        return [s for s in result if s.strip()]
    
    async def _split_sentence_by_words(self, sentence: str, max_tokens: int) -> list[str]:
        """Разбивает предложение по словам с соблюдением лимита токенов"""
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        # Ограничиваем количество слов для избежания зацикливания
        max_words = min(len(words), 500)  # Максимум 500 слов за раз
        
        for i, word in enumerate(words[:max_words]):
            if i > max_words:
                logger.warning(f"Достигнут лимит слов при разбиении предложения, прерываем")
                break
                
            potential_chunk = current_chunk + " " + word if current_chunk else word
            
            try:
                # Добавляем timeout для проверки слова
                is_valid, _ = await asyncio.wait_for(
                    self.validate_token_limit(potential_chunk, max_tokens),
                    timeout=3.0  # 3 секунды на слово
                )
                
                if is_valid:
                    current_chunk = potential_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Проверяем слово отдельно
                    try:
                        is_word_valid, _ = await asyncio.wait_for(
                            self.validate_token_limit(word, max_tokens),
                            timeout=3.0
                        )
                        if is_word_valid:
                            current_chunk = word
                        else:
                            # Если слово слишком длинное, обрезаем его
                            logger.warning(f"Слово '{word[:50]}...' превышает лимит токенов, обрезаем")
                            truncated_word = word[:max_tokens*2]  # Приблизительное обрезание
                            current_chunk = truncated_word
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout при проверке слова '{word[:50]}...', используем приблизительное обрезание")
                        current_chunk = word[:max_tokens*2]  # Обрезаем без проверки
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout при проверке фразы со словом {i}, пропускаем")
                continue
            except Exception as e:
                logger.warning(f"Ошибка при проверке слова {i}: {e}, пропускаем")
                continue
        
        # Если остались необработанные слова, добавляем их как отдельный чанк
        if len(words) > max_words:
            remaining_words = " ".join(words[max_words:])
            if len(remaining_words) > 0:
                logger.warning(f"Добавляем оставшиеся {len(words) - max_words} слов как отдельный чанк")
                # Обрезаем если слишком длинный
                if len(remaining_words) > max_tokens * 2:
                    remaining_words = remaining_words[:max_tokens * 2]
                chunks.append(remaining_words)
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

# Создаем глобальный экземпляр сервиса
tokenizer_service = TokenizerService() 