"""
File storage infrastructure with local filesystem support.
For now using local storage, can be extended to MinIO later.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO
from uuid import uuid4

import trio
import aiofiles

from app.core.config import settings

logger = logging.getLogger(__name__)


class StorageManager:
    """
    File storage manager with trio support.
    Currently using local filesystem, can be extended to MinIO.
    """
    
    def __init__(self):
        self.storage_path = Path(settings.STORAGE_PATH)
        self.temp_path = Path(settings.TEMP_PATH)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """
        Ensure storage directories exist.
        """
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.storage_path / "documents").mkdir(exist_ok=True)
        (self.storage_path / "processed").mkdir(exist_ok=True)
        (self.storage_path / "images").mkdir(exist_ok=True)
        (self.temp_path / "uploads").mkdir(exist_ok=True)
    
    async def save_uploaded_file(self, file_data: bytes, filename: str, document_id: str) -> str:
        """
        Save uploaded file to storage.
        """
        try:
            # Create document directory
            doc_dir = self.storage_path / "documents" / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            file_path = doc_dir / filename
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_data)
            
            logger.info(f"Saved file {filename} for document {document_id}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            raise
    
    async def save_temp_file(self, file_data: bytes, filename: str) -> str:
        """
        Save temporary file.
        """
        try:
            # Generate unique filename
            temp_filename = f"{uuid4()}_{filename}"
            temp_path = self.temp_path / "uploads" / temp_filename
            
            async with aiofiles.open(temp_path, 'wb') as f:
                await f.write(file_data)
            
            logger.debug(f"Saved temporary file {temp_filename}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Failed to save temporary file {filename}: {e}")
            raise
    
    async def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Get file content.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            async with aiofiles.open(path, 'rb') as f:
                return await f.read()
                
        except Exception as e:
            logger.error(f"Failed to get file {file_path}: {e}")
            return None
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file.
        """
        try:
            path = Path(file_path)
            if path.exists():
                await trio.to_thread.run_sync(path.unlink)
                logger.debug(f"Deleted file {file_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
    
    async def delete_document_files(self, document_id: str) -> bool:
        """
        Delete all files for a document.
        """
        try:
            doc_dir = self.storage_path / "documents" / document_id
            if doc_dir.exists():
                await trio.to_thread.run_sync(shutil.rmtree, doc_dir)
                logger.info(f"Deleted all files for document {document_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete files for document {document_id}: {e}")
            return False
    
    async def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Cleanup old temporary files.
        """
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            temp_dir = self.temp_path / "uploads"
            deleted_count = 0
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        await trio.to_thread.run_sync(file_path.unlink)
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} temporary files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup temporary files: {e}")
            return 0
    
    async def get_storage_stats(self) -> dict:
        """
        Get storage statistics.
        """
        try:
            def get_dir_size(path: Path) -> int:
                total = 0
                for entry in path.rglob('*'):
                    if entry.is_file():
                        total += entry.stat().st_size
                return total
            
            storage_size = await trio.to_thread.run_sync(get_dir_size, self.storage_path)
            temp_size = await trio.to_thread.run_sync(get_dir_size, self.temp_path)
            
            return {
                "storage_size_bytes": storage_size,
                "temp_size_bytes": temp_size,
                "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                "temp_size_mb": round(temp_size / (1024 * 1024), 2),
                "storage_path": str(self.storage_path),
                "temp_path": str(self.temp_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists.
        """
        return Path(file_path).exists()
    
    def get_file_size(self, file_path: str) -> Optional[int]:
        """
        Get file size in bytes.
        """
        try:
            path = Path(file_path)
            if path.exists():
                return path.stat().st_size
            return None
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return None


# Global instance
storage_manager = StorageManager()