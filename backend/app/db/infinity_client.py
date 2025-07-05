import infinity
from app.core.config import settings
import logging
import threading

logger = logging.getLogger(__name__)

class InfinityClient:
    """
    Singleton класс для централизованного управления подключением к Infinity DB
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(InfinityClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info(f"Инициализация центр. клиента Infinity {settings.INFINITY_HOST}:{settings.INFINITY_PORT}")
        
        try:
            self._client = infinity.connect(
                infinity.NetworkAddress(settings.INFINITY_HOST, settings.INFINITY_PORT)
            )
            
            # Обеспечиваем существование базы данных по умолчанию
            try:
                self._client.get_database("default")
                logger.info("Подключение к существующей БД 'default' в Infinity")
            except Exception:
                self._client.create_database("default")
                logger.info("Создана БД 'default' в Infinity")
                
            self._initialized = True
            logger.info("Центр. клиент Infinity инициализирован успешно")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации клиента Infinity: {e}", exc_info=True)
            raise
    
    def reconnect(self):
        """Принудительное переподключение к Infinity DB"""
        logger.info("Принудительное переподключение к Infinity DB...")
        self._initialized = False
        self._client = None
        self.__init__()
    
    def get_client(self):
        """Возвращает клиент Infinity DB"""
        if not self._initialized:
            raise RuntimeError("Infinity client not initialized")
        return self._client
    
    def get_database(self, name: str = "default"):
        """Возвращает базу данных по имени"""
        return self.get_client().get_database(name)
    
    def list_tables(self, db_name: str = "default"):
        """Возвращает список таблиц в базе данных"""
        try:
            db = self.get_database(db_name)
            tables = db.list_tables()
            return tables
        except Exception as e:
            logger.error(f"Ошибка получения списка таблиц в БД '{db_name}': {e}")
            return []
    
    def table_exists(self, table_name: str, db_name: str = "default"):
        """Проверяет существование таблицы"""
        try:
            db = self.get_database(db_name)
            db.get_table(table_name)
            return True
        except Exception:
            return False
    
    def get_table_info(self, table_name: str, db_name: str = "default"):
        """Получает информацию о таблице"""
        try:
            db = self.get_database(db_name)
            table = db.get_table(table_name)
            
            # Пытаемся получить количество записей разными способами
            try:
                # Метод 1: выбираем все записи и считаем
                result = table.output(['document_id']).to_df()
                if result and len(result) > 0:
                    row_count = len(result[0])
                else:
                    row_count = 0
            except Exception as count_error:
                logger.warning(f"Не удалось подсчитать строки в таблице '{table_name}': {count_error}")
                row_count = "unknown"
            
            return {"exists": True, "row_count": row_count}
        except Exception as e:
            return {"exists": False, "error": str(e)}

# Создаем глобальный экземпляр
infinity_client = InfinityClient() 