"""
Infinity DB client with trio support.
"""
import logging
from typing import Optional, Dict, Any
import threading

import trio
import infinity
from infinity.common import ConflictType

from app.core.config import settings

logger = logging.getLogger(__name__)


class InfinityClient:
    """
    Singleton Infinity DB client with trio compatibility.
    """
    _instance: Optional['InfinityClient'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.client: Optional[infinity.connect] = None
            self.initialized = False
            self._connection_lock = trio.Lock()
    
    async def connect(self):
        """
        Connect to Infinity DB with trio support.
        """
        async with self._connection_lock:
            if self.client is None:
                try:
                    # Connect to Infinity DB
                    self.client = infinity.connect(
                        infinity.common.NetworkAddress(
                            settings.INFINITY_HOST, 
                            settings.INFINITY_PORT
                        )
                    )
                    
                    logger.info(f"Connected to Infinity DB at {settings.INFINITY_HOST}:{settings.INFINITY_PORT}")
                    self.initialized = True
                    
                except Exception as e:
                    logger.error(f"Failed to connect to Infinity DB: {e}")
                    raise
    
    async def disconnect(self):
        """
        Disconnect from Infinity DB.
        """
        async with self._connection_lock:
            if self.client:
                try:
                    self.client.disconnect()
                    self.client = None
                    self.initialized = False
                    logger.info("Disconnected from Infinity DB")
                except Exception as e:
                    logger.error(f"Error disconnecting from Infinity DB: {e}")
    
    async def get_database(self, name: str = "default"):
        """
        Get or create database.
        """
        if not self.initialized:
            await self.connect()
        
        try:
            # Try to get existing database
            db = self.client.get_database(name)
            return db
        except Exception:
            # Create database if it doesn't exist
            db = self.client.create_database(name, ConflictType.Ignore)
            logger.info(f"Created database: {name}")
            return db
    
    async def create_table(self, db_name: str, table_name: str, schema: Dict[str, Any]):
        """
        Create table with specified schema.
        """
        db = await self.get_database(db_name)
        
        try:
            # Create table with schema
            table = db.create_table(table_name, schema, ConflictType.Ignore)
            logger.info(f"Created table: {table_name} in database: {db_name}")
            return table
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise
    
    async def get_table(self, db_name: str, table_name: str):
        """
        Get existing table.
        """
        db = await self.get_database(db_name)
        
        try:
            table = db.get_table(table_name)
            return table
        except Exception as e:
            logger.error(f"Failed to get table {table_name}: {e}")
            raise
    
    async def list_tables(self, db_name: str = "default"):
        """
        List all tables in database.
        """
        db = await self.get_database(db_name)
        
        try:
            tables = db.list_tables()
            return tables
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Infinity DB health.
        """
        try:
            if not self.initialized:
                await self.connect()
            
            # Try to list databases as health check
            databases = self.client.list_databases()
            
            return {
                "status": "healthy",
                "host": settings.INFINITY_HOST,
                "port": settings.INFINITY_PORT,
                "databases": len(databases),
                "connected": True
            }
        except Exception as e:
            logger.error(f"Infinity DB health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }


# Global instance
infinity_client = InfinityClient()