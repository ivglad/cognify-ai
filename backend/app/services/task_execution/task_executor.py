"""
Task execution system with trio async framework for document processing.
"""
import os
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import trio

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Task priority enumeration."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskInfo:
    """Task information structure."""
    task_id: str
    task_type: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    worker_id: Optional[str] = None
    heartbeat_at: Optional[datetime] = None
    timeout_seconds: int = 3600  # 1 hour default
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class WorkerInfo:
    """Worker information structure."""
    worker_id: str
    worker_type: str
    status: str
    current_task_id: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    started_at: datetime = None
    last_heartbeat: datetime = None
    resource_usage: Dict[str, Any] = None


class TaskExecutor:
    """
    Asynchronous task execution system with trio framework.
    Provides concurrency control, progress tracking, and health monitoring.
    """
    
    def __init__(self):
        # Concurrency limits from environment variables
        self.MAX_CONCURRENT_TASKS = int(os.environ.get('MAX_CONCURRENT_TASKS', '5'))
        self.MAX_CONCURRENT_CHUNK_BUILDERS = int(os.environ.get('MAX_CONCURRENT_CHUNK_BUILDERS', '1'))
        self.MAX_CONCURRENT_MINIO = int(os.environ.get('MAX_CONCURRENT_MINIO', '10'))
        
        # Trio concurrency controls
        self.task_limiter = trio.Semaphore(self.MAX_CONCURRENT_TASKS)
        self.chunk_limiter = trio.CapacityLimiter(self.MAX_CONCURRENT_CHUNK_BUILDERS)
        self.minio_limiter = trio.CapacityLimiter(self.MAX_CONCURRENT_MINIO)
        
        # Task tracking
        self.current_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: Dict[str, TaskInfo] = {}
        self.task_handlers: Dict[str, Callable] = {}
        
        # Worker tracking
        self.workers: Dict[str, WorkerInfo] = {}
        self.worker_heartbeat_timeout = int(os.environ.get('WORKER_HEARTBEAT_TIMEOUT', '120'))
        
        # Task queue
        self.task_queue: List[TaskInfo] = []
        self.queue_lock = trio.Lock()
        
        # Shutdown flag
        self.shutdown_requested = False
        
        # Statistics
        self.stats = {
            'total_tasks_processed': 0,
            'total_tasks_failed': 0,
            'total_tasks_cancelled': 0,
            'average_processing_time': 0.0,
            'uptime_start': datetime.now()
        }
    
    async def start(self):
        """Start the task executor with background workers."""
        try:
            logger.info("Starting Task Executor with trio async framework")
            
            # Start background tasks
            async with trio.open_nursery() as nursery:
                # Start task processor
                nursery.start_soon(self._task_processor)
                
                # Start health monitor
                nursery.start_soon(self._health_monitor)
                
                # Start cleanup worker
                nursery.start_soon(self._cleanup_worker)
                
                # Wait for shutdown signal
                await self._wait_for_shutdown()
                
        except Exception as e:
            logger.error(f"Task executor startup failed: {e}")
            raise
    
    async def submit_task(self, 
                         task_type: str, 
                         task_data: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         timeout_seconds: int = 3600,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a task for execution.
        
        Args:
            task_type: Type of task to execute
            task_data: Task data and parameters
            priority: Task priority
            timeout_seconds: Task timeout in seconds
            metadata: Additional task metadata
            
        Returns:
            Task ID
        """
        try:
            task_id = str(uuid.uuid4())
            
            task_info = TaskInfo(
                task_id=task_id,
                task_type=task_type,
                status=TaskStatus.PENDING,
                priority=priority,
                created_at=datetime.now(),
                timeout_seconds=timeout_seconds,
                metadata=metadata or {}
            )
            
            # Add task data to metadata
            task_info.metadata['task_data'] = task_data
            
            # Add to queue
            async with self.queue_lock:
                self.task_queue.append(task_info)
                # Sort by priority
                self.task_queue.sort(key=lambda t: self._priority_weight(t.priority), reverse=True)
            
            logger.info(f"Task {task_id} ({task_type}) submitted with priority {priority}")
            return task_id
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or pending task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was cancelled, False otherwise
        """
        try:
            # Check if task is in current tasks
            if task_id in self.current_tasks:
                task_info = self.current_tasks[task_id]
                if task_info.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                    task_info.status = TaskStatus.CANCELLED
                    task_info.completed_at = datetime.now()
                    task_info.message = "Task cancelled by user request"
                    
                    # Move to completed tasks
                    self.completed_tasks[task_id] = task_info
                    del self.current_tasks[task_id]
                    
                    self.stats['total_tasks_cancelled'] += 1
                    logger.info(f"Task {task_id} cancelled")
                    return True
            
            # Check if task is in queue
            async with self.queue_lock:
                for i, task_info in enumerate(self.task_queue):
                    if task_info.task_id == task_id:
                        task_info.status = TaskStatus.CANCELLED
                        task_info.completed_at = datetime.now()
                        task_info.message = "Task cancelled before execution"
                        
                        # Move to completed tasks
                        self.completed_tasks[task_id] = task_info
                        del self.task_queue[i]
                        
                        self.stats['total_tasks_cancelled'] += 1
                        logger.info(f"Queued task {task_id} cancelled")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Task cancellation failed: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information or None if not found
        """
        try:
            # Check current tasks
            if task_id in self.current_tasks:
                return asdict(self.current_tasks[task_id])
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return asdict(self.completed_tasks[task_id])
            
            # Check queue
            async with self.queue_lock:
                for task_info in self.task_queue:
                    if task_info.task_id == task_id:
                        return asdict(task_info)
            
            return None
            
        except Exception as e:
            logger.error(f"Task status retrieval failed: {e}")
            return None
    
    async def get_all_tasks(self, 
                           status_filter: Optional[TaskStatus] = None,
                           task_type_filter: Optional[str] = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all tasks with optional filtering.
        
        Args:
            status_filter: Filter by task status
            task_type_filter: Filter by task type
            limit: Maximum number of tasks to return
            
        Returns:
            List of task information
        """
        try:
            all_tasks = []
            
            # Add current tasks
            for task_info in self.current_tasks.values():
                if self._matches_filters(task_info, status_filter, task_type_filter):
                    all_tasks.append(asdict(task_info))
            
            # Add completed tasks
            for task_info in self.completed_tasks.values():
                if self._matches_filters(task_info, status_filter, task_type_filter):
                    all_tasks.append(asdict(task_info))
            
            # Add queued tasks
            async with self.queue_lock:
                for task_info in self.task_queue:
                    if self._matches_filters(task_info, status_filter, task_type_filter):
                        all_tasks.append(asdict(task_info))
            
            # Sort by creation time (newest first)
            all_tasks.sort(key=lambda t: t['created_at'], reverse=True)
            
            return all_tasks[:limit]
            
        except Exception as e:
            logger.error(f"Task listing failed: {e}")
            return []
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """
        Register a task handler for a specific task type.
        
        Args:
            task_type: Task type to handle
            handler: Async callable to handle the task
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def set_task_progress(self, task_id: str, progress: float, message: str = ""):
        """
        Update task progress.
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            message: Progress message
        """
        try:
            if task_id in self.current_tasks:
                task_info = self.current_tasks[task_id]
                task_info.progress = max(0.0, min(1.0, progress))
                task_info.message = message
                task_info.heartbeat_at = datetime.now()
                
        except Exception as e:
            logger.error(f"Task progress update failed: {e}")
    
    async def _task_processor(self):
        """Main task processing loop."""
        try:
            while not self.shutdown_requested:
                try:
                    # Get next task from queue
                    task_info = None
                    async with self.queue_lock:
                        if self.task_queue:
                            task_info = self.task_queue.pop(0)
                    
                    if task_info:
                        # Process task with concurrency control
                        async with self.task_limiter:
                            await self._execute_task(task_info)
                    else:
                        # No tasks available, wait a bit
                        await trio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Task processor error: {e}")
                    await trio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Task processor failed: {e}")
    
    async def _execute_task(self, task_info: TaskInfo):
        """
        Execute a single task.
        
        Args:
            task_info: Task information
        """
        task_id = task_info.task_id
        
        try:
            # Move task to current tasks
            self.current_tasks[task_id] = task_info
            
            # Update task status
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.now()
            task_info.worker_id = f"worker_{trio.lowlevel.current_task().name}"
            task_info.heartbeat_at = datetime.now()
            
            logger.info(f"Starting task {task_id} ({task_info.task_type})")
            
            # Check if handler exists
            if task_info.task_type not in self.task_handlers:
                raise ValueError(f"No handler registered for task type: {task_info.task_type}")
            
            handler = self.task_handlers[task_info.task_type]
            
            # Execute task with timeout
            with trio.move_on_after(task_info.timeout_seconds) as cancel_scope:
                # Get task data from metadata
                task_data = task_info.metadata.get('task_data', {})
                
                # Execute the handler
                result = await handler(task_id, task_data, self)
                
                # Task completed successfully
                task_info.status = TaskStatus.COMPLETED
                task_info.completed_at = datetime.now()
                task_info.progress = 1.0
                task_info.result = result
                task_info.message = "Task completed successfully"
                
                # Update statistics
                processing_time = (task_info.completed_at - task_info.started_at).total_seconds()
                self._update_processing_stats(processing_time)
                
                logger.info(f"Task {task_id} completed in {processing_time:.2f} seconds")
            
            # Check if task was cancelled due to timeout
            if cancel_scope.cancelled_caught:
                task_info.status = TaskStatus.TIMEOUT
                task_info.completed_at = datetime.now()
                task_info.error = f"Task timed out after {task_info.timeout_seconds} seconds"
                task_info.message = "Task timed out"
                
                self.stats['total_tasks_failed'] += 1
                logger.warning(f"Task {task_id} timed out")
                
        except Exception as e:
            # Task failed
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = datetime.now()
            task_info.error = str(e)
            task_info.message = f"Task failed: {str(e)}"
            
            self.stats['total_tasks_failed'] += 1
            logger.error(f"Task {task_id} failed: {e}")
            
            # Check if we should retry
            if task_info.retry_count < task_info.max_retries:
                task_info.retry_count += 1
                task_info.status = TaskStatus.PENDING
                task_info.started_at = None
                task_info.error = None
                
                # Re-queue the task
                async with self.queue_lock:
                    self.task_queue.append(task_info)
                    self.task_queue.sort(key=lambda t: self._priority_weight(t.priority), reverse=True)
                
                logger.info(f"Task {task_id} queued for retry ({task_info.retry_count}/{task_info.max_retries})")
                return
        
        finally:
            # Move task to completed tasks
            if task_id in self.current_tasks:
                self.completed_tasks[task_id] = self.current_tasks[task_id]
                del self.current_tasks[task_id]
                
                self.stats['total_tasks_processed'] += 1
    
    async def _health_monitor(self):
        """Monitor worker health and task timeouts."""
        try:
            while not self.shutdown_requested:
                try:
                    current_time = datetime.now()
                    
                    # Check for stale tasks
                    stale_tasks = []
                    for task_id, task_info in self.current_tasks.items():
                        if task_info.heartbeat_at:
                            time_since_heartbeat = (current_time - task_info.heartbeat_at).total_seconds()
                            if time_since_heartbeat > self.worker_heartbeat_timeout:
                                stale_tasks.append(task_id)
                    
                    # Handle stale tasks
                    for task_id in stale_tasks:
                        task_info = self.current_tasks[task_id]
                        task_info.status = TaskStatus.FAILED
                        task_info.completed_at = current_time
                        task_info.error = "Task worker became unresponsive"
                        task_info.message = "Worker heartbeat timeout"
                        
                        # Move to completed tasks
                        self.completed_tasks[task_id] = task_info
                        del self.current_tasks[task_id]
                        
                        self.stats['total_tasks_failed'] += 1
                        logger.warning(f"Task {task_id} marked as failed due to worker timeout")
                    
                    # Sleep before next check
                    await trio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Health monitor error: {e}")
                    await trio.sleep(60)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Health monitor failed: {e}")
    
    async def _cleanup_worker(self):
        """Clean up old completed tasks and maintain memory usage."""
        try:
            while not self.shutdown_requested:
                try:
                    current_time = datetime.now()
                    cleanup_threshold = current_time - timedelta(hours=24)  # Keep tasks for 24 hours
                    
                    # Clean up old completed tasks
                    tasks_to_remove = []
                    for task_id, task_info in self.completed_tasks.items():
                        if task_info.completed_at and task_info.completed_at < cleanup_threshold:
                            tasks_to_remove.append(task_id)
                    
                    for task_id in tasks_to_remove:
                        del self.completed_tasks[task_id]
                    
                    if tasks_to_remove:
                        logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
                    
                    # Sleep before next cleanup
                    await trio.sleep(3600)  # Clean up every hour
                    
                except Exception as e:
                    logger.error(f"Cleanup worker error: {e}")
                    await trio.sleep(3600)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Cleanup worker failed: {e}")
    
    async def _wait_for_shutdown(self):
        """Wait for shutdown signal."""
        # This would typically wait for a signal or event
        # For now, we'll just wait indefinitely
        while not self.shutdown_requested:
            await trio.sleep(1)
    
    def _priority_weight(self, priority: TaskPriority) -> int:
        """Get numeric weight for priority."""
        weights = {
            TaskPriority.LOW: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.HIGH: 3,
            TaskPriority.CRITICAL: 4
        }
        return weights.get(priority, 2)
    
    def _matches_filters(self, 
                        task_info: TaskInfo, 
                        status_filter: Optional[TaskStatus], 
                        task_type_filter: Optional[str]) -> bool:
        """Check if task matches the given filters."""
        if status_filter and task_info.status != status_filter:
            return False
        if task_type_filter and task_info.task_type != task_type_filter:
            return False
        return True
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing time statistics."""
        try:
            total_processed = self.stats['total_tasks_processed']
            current_avg = self.stats['average_processing_time']
            
            # Calculate new average
            new_avg = ((current_avg * total_processed) + processing_time) / (total_processed + 1)
            self.stats['average_processing_time'] = new_avg
            
        except Exception as e:
            logger.error(f"Stats update failed: {e}")
    
    async def get_executor_stats(self) -> Dict[str, Any]:
        """Get executor statistics and status."""
        try:
            current_time = datetime.now()
            uptime = (current_time - self.stats['uptime_start']).total_seconds()
            
            return {
                'status': 'running' if not self.shutdown_requested else 'shutting_down',
                'uptime_seconds': uptime,
                'concurrency_limits': {
                    'max_concurrent_tasks': self.MAX_CONCURRENT_TASKS,
                    'max_concurrent_chunk_builders': self.MAX_CONCURRENT_CHUNK_BUILDERS,
                    'max_concurrent_minio': self.MAX_CONCURRENT_MINIO
                },
                'current_usage': {
                    'active_tasks': len(self.current_tasks),
                    'queued_tasks': len(self.task_queue),
                    'completed_tasks': len(self.completed_tasks)
                },
                'statistics': self.stats.copy(),
                'task_handlers': list(self.task_handlers.keys()),
                'worker_heartbeat_timeout': self.worker_heartbeat_timeout
            }
            
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown the task executor."""
        try:
            logger.info("Initiating task executor shutdown")
            self.shutdown_requested = True
            
            # Wait for current tasks to complete (with timeout)
            shutdown_timeout = 300  # 5 minutes
            start_time = time.time()
            
            while self.current_tasks and (time.time() - start_time) < shutdown_timeout:
                await trio.sleep(1)
            
            # Cancel remaining tasks
            for task_id, task_info in self.current_tasks.items():
                task_info.status = TaskStatus.CANCELLED
                task_info.completed_at = datetime.now()
                task_info.message = "Task cancelled due to shutdown"
                
                self.completed_tasks[task_id] = task_info
            
            self.current_tasks.clear()
            
            logger.info("Task executor shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")


# Global instance
task_executor = TaskExecutor()