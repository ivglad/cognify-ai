"""
Advanced concurrency management for task execution with trio.
"""
import os
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import trio
from contextlib import asynccontextmanager

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ResourceType(str, Enum):
    """Resource type enumeration."""
    DOCUMENT_PROCESSING = "document_processing"
    CHUNK_BUILDING = "chunk_building"
    MINIO_OPERATIONS = "minio_operations"
    LLM_REQUESTS = "llm_requests"
    EMBEDDING_GENERATION = "embedding_generation"
    DATABASE_OPERATIONS = "database_operations"
    EXTERNAL_API = "external_api"


@dataclass
class ResourceLimit:
    """Resource limit configuration."""
    resource_type: ResourceType
    max_concurrent: int
    current_usage: int = 0
    queue_size: int = 0
    total_requests: int = 0
    total_completed: int = 0
    total_failed: int = 0
    average_duration: float = 0.0


@dataclass
class TaskResourceRequest:
    """Task resource request."""
    task_id: str
    resource_type: ResourceType
    requested_at: float
    acquired_at: Optional[float] = None
    released_at: Optional[float] = None
    metadata: Dict[str, Any] = None


class ConcurrencyManager:
    """
    Advanced concurrency management with resource limits and monitoring.
    """
    
    def __init__(self):
        # Load limits from environment variables
        self.resource_limits = {
            ResourceType.DOCUMENT_PROCESSING: ResourceLimit(
                ResourceType.DOCUMENT_PROCESSING,
                int(os.environ.get('MAX_CONCURRENT_TASKS', '5'))
            ),
            ResourceType.CHUNK_BUILDING: ResourceLimit(
                ResourceType.CHUNK_BUILDING,
                int(os.environ.get('MAX_CONCURRENT_CHUNK_BUILDERS', '1'))
            ),
            ResourceType.MINIO_OPERATIONS: ResourceLimit(
                ResourceType.MINIO_OPERATIONS,
                int(os.environ.get('MAX_CONCURRENT_MINIO', '10'))
            ),
            ResourceType.LLM_REQUESTS: ResourceLimit(
                ResourceType.LLM_REQUESTS,
                int(os.environ.get('MAX_CONCURRENT_LLM', '3'))
            ),
            ResourceType.EMBEDDING_GENERATION: ResourceLimit(
                ResourceType.EMBEDDING_GENERATION,
                int(os.environ.get('MAX_CONCURRENT_EMBEDDINGS', '5'))
            ),
            ResourceType.DATABASE_OPERATIONS: ResourceLimit(
                ResourceType.DATABASE_OPERATIONS,
                int(os.environ.get('MAX_CONCURRENT_DB', '20'))
            ),
            ResourceType.EXTERNAL_API: ResourceLimit(
                ResourceType.EXTERNAL_API,
                int(os.environ.get('MAX_CONCURRENT_EXTERNAL_API', '5'))
            )
        }
        
        # Create trio limiters for each resource type
        self.limiters = {
            resource_type: trio.CapacityLimiter(limit.max_concurrent)
            for resource_type, limit in self.resource_limits.items()
        }
        
        # Track active requests
        self.active_requests: Dict[str, TaskResourceRequest] = {}
        self.request_history: List[TaskResourceRequest] = []
        
        # Monitoring
        self.stats_lock = trio.Lock()
        
        # Graceful shutdown
        self.shutdown_requested = False
        self.active_tasks: Set[str] = set()
    
    @asynccontextmanager
    async def acquire_resource(self, 
                              task_id: str, 
                              resource_type: ResourceType,
                              timeout_seconds: Optional[float] = None,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        Acquire a resource with concurrency control.
        
        Args:
            task_id: Task ID requesting the resource
            resource_type: Type of resource to acquire
            timeout_seconds: Optional timeout for acquisition
            metadata: Additional metadata for the request
        """
        request = TaskResourceRequest(
            task_id=task_id,
            resource_type=resource_type,
            requested_at=time.time(),
            metadata=metadata or {}
        )
        
        try:
            # Update queue size
            async with self.stats_lock:
                self.resource_limits[resource_type].queue_size += 1
                self.resource_limits[resource_type].total_requests += 1
            
            logger.debug(f"Task {task_id} requesting {resource_type} resource")
            
            # Acquire the resource with optional timeout
            limiter = self.limiters[resource_type]
            
            if timeout_seconds:
                with trio.move_on_after(timeout_seconds) as cancel_scope:
                    async with limiter:
                        await self._handle_resource_acquired(request)
                        try:
                            yield request
                        finally:
                            await self._handle_resource_released(request)
                
                if cancel_scope.cancelled_caught:
                    await self._handle_resource_timeout(request)
                    raise trio.TooSlowError("Resource acquisition timed out")
            else:
                async with limiter:
                    await self._handle_resource_acquired(request)
                    try:
                        yield request
                    finally:
                        await self._handle_resource_released(request)
                        
        except Exception as e:
            await self._handle_resource_error(request, e)
            raise
    
    async def _handle_resource_acquired(self, request: TaskResourceRequest):
        """Handle resource acquisition."""
        try:
            request.acquired_at = time.time()
            
            async with self.stats_lock:
                resource_limit = self.resource_limits[request.resource_type]
                resource_limit.current_usage += 1
                resource_limit.queue_size -= 1
            
            self.active_requests[request.task_id] = request
            self.active_tasks.add(request.task_id)
            
            logger.debug(f"Task {request.task_id} acquired {request.resource_type} resource")
            
        except Exception as e:
            logger.error(f"Resource acquisition handling failed: {e}")
    
    async def _handle_resource_released(self, request: TaskResourceRequest):
        """Handle resource release."""
        try:
            request.released_at = time.time()
            
            # Calculate duration
            if request.acquired_at:
                duration = request.released_at - request.acquired_at
                
                async with self.stats_lock:
                    resource_limit = self.resource_limits[request.resource_type]
                    resource_limit.current_usage -= 1
                    resource_limit.total_completed += 1
                    
                    # Update average duration
                    total_completed = resource_limit.total_completed
                    current_avg = resource_limit.average_duration
                    new_avg = ((current_avg * (total_completed - 1)) + duration) / total_completed
                    resource_limit.average_duration = new_avg
            
            # Move to history
            if request.task_id in self.active_requests:
                del self.active_requests[request.task_id]
            
            self.active_tasks.discard(request.task_id)
            self.request_history.append(request)
            
            # Keep history size manageable
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-500:]
            
            logger.debug(f"Task {request.task_id} released {request.resource_type} resource")
            
        except Exception as e:
            logger.error(f"Resource release handling failed: {e}")
    
    async def _handle_resource_timeout(self, request: TaskResourceRequest):
        """Handle resource acquisition timeout."""
        try:
            async with self.stats_lock:
                resource_limit = self.resource_limits[request.resource_type]
                resource_limit.queue_size -= 1
                resource_limit.total_failed += 1
            
            logger.warning(f"Task {request.task_id} timed out waiting for {request.resource_type} resource")
            
        except Exception as e:
            logger.error(f"Resource timeout handling failed: {e}")
    
    async def _handle_resource_error(self, request: TaskResourceRequest, error: Exception):
        """Handle resource acquisition error."""
        try:
            async with self.stats_lock:
                resource_limit = self.resource_limits[request.resource_type]
                if resource_limit.queue_size > 0:
                    resource_limit.queue_size -= 1
                resource_limit.total_failed += 1
            
            logger.error(f"Task {request.task_id} failed to acquire {request.resource_type} resource: {error}")
            
        except Exception as e:
            logger.error(f"Resource error handling failed: {e}")
    
    async def cancel_task_resources(self, task_id: str) -> bool:
        """
        Cancel all resources for a specific task.
        
        Args:
            task_id: Task ID to cancel resources for
            
        Returns:
            True if resources were cancelled, False otherwise
        """
        try:
            cancelled = False
            
            # Check if task has active resources
            if task_id in self.active_requests:
                request = self.active_requests[task_id]
                
                # Mark as cancelled in metadata
                request.metadata['cancelled'] = True
                request.metadata['cancelled_at'] = time.time()
                
                # The actual resource will be released when the context manager exits
                cancelled = True
                
                logger.info(f"Marked resources for task {task_id} as cancelled")
            
            return cancelled
            
        except Exception as e:
            logger.error(f"Task resource cancellation failed: {e}")
            return False
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        try:
            async with self.stats_lock:
                stats = {}
                
                for resource_type, limit in self.resource_limits.items():
                    stats[resource_type.value] = {
                        'max_concurrent': limit.max_concurrent,
                        'current_usage': limit.current_usage,
                        'queue_size': limit.queue_size,
                        'utilization_percent': (limit.current_usage / limit.max_concurrent) * 100,
                        'total_requests': limit.total_requests,
                        'total_completed': limit.total_completed,
                        'total_failed': limit.total_failed,
                        'average_duration_seconds': limit.average_duration,
                        'success_rate_percent': (
                            (limit.total_completed / max(limit.total_requests, 1)) * 100
                        )
                    }
                
                return {
                    'resource_stats': stats,
                    'active_tasks': len(self.active_tasks),
                    'total_active_requests': len(self.active_requests),
                    'history_size': len(self.request_history),
                    'shutdown_requested': self.shutdown_requested
                }
                
        except Exception as e:
            logger.error(f"Resource stats retrieval failed: {e}")
            return {'error': str(e)}
    
    async def update_resource_limits(self, new_limits: Dict[ResourceType, int]):
        """
        Update resource limits dynamically.
        
        Args:
            new_limits: Dictionary of resource types and their new limits
        """
        try:
            async with self.stats_lock:
                for resource_type, new_limit in new_limits.items():
                    if resource_type in self.resource_limits:
                        old_limit = self.resource_limits[resource_type].max_concurrent
                        
                        # Update the limit
                        self.resource_limits[resource_type].max_concurrent = new_limit
                        
                        # Update the trio limiter
                        # Note: trio.CapacityLimiter doesn't support dynamic updates,
                        # so we need to create a new one
                        self.limiters[resource_type] = trio.CapacityLimiter(new_limit)
                        
                        logger.info(f"Updated {resource_type} limit from {old_limit} to {new_limit}")
                
        except Exception as e:
            logger.error(f"Resource limit update failed: {e}")
    
    async def wait_for_resource_availability(self, 
                                           resource_type: ResourceType,
                                           timeout_seconds: Optional[float] = None) -> bool:
        """
        Wait for a resource to become available.
        
        Args:
            resource_type: Type of resource to wait for
            timeout_seconds: Optional timeout
            
        Returns:
            True if resource became available, False if timed out
        """
        try:
            start_time = time.time()
            
            while True:
                async with self.stats_lock:
                    limit = self.resource_limits[resource_type]
                    if limit.current_usage < limit.max_concurrent:
                        return True
                
                # Check timeout
                if timeout_seconds and (time.time() - start_time) > timeout_seconds:
                    return False
                
                # Wait a bit before checking again
                await trio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Resource availability wait failed: {e}")
            return False
    
    async def get_resource_queue_info(self, resource_type: ResourceType) -> Dict[str, Any]:
        """
        Get detailed queue information for a resource type.
        
        Args:
            resource_type: Resource type to get info for
            
        Returns:
            Queue information
        """
        try:
            async with self.stats_lock:
                limit = self.resource_limits[resource_type]
                
                # Get active requests for this resource type
                active_for_resource = [
                    req for req in self.active_requests.values()
                    if req.resource_type == resource_type
                ]
                
                return {
                    'resource_type': resource_type.value,
                    'max_concurrent': limit.max_concurrent,
                    'current_usage': limit.current_usage,
                    'available_slots': limit.max_concurrent - limit.current_usage,
                    'queue_size': limit.queue_size,
                    'active_requests': len(active_for_resource),
                    'active_task_ids': [req.task_id for req in active_for_resource],
                    'average_wait_time': self._calculate_average_wait_time(resource_type),
                    'estimated_wait_time': self._estimate_wait_time(resource_type)
                }
                
        except Exception as e:
            logger.error(f"Resource queue info retrieval failed: {e}")
            return {'error': str(e)}
    
    def _calculate_average_wait_time(self, resource_type: ResourceType) -> float:
        """Calculate average wait time for a resource type."""
        try:
            relevant_requests = [
                req for req in self.request_history[-100:]  # Last 100 requests
                if req.resource_type == resource_type and req.acquired_at and req.requested_at
            ]
            
            if not relevant_requests:
                return 0.0
            
            wait_times = [req.acquired_at - req.requested_at for req in relevant_requests]
            return sum(wait_times) / len(wait_times)
            
        except Exception as e:
            logger.error(f"Average wait time calculation failed: {e}")
            return 0.0
    
    def _estimate_wait_time(self, resource_type: ResourceType) -> float:
        """Estimate wait time based on current queue and average processing time."""
        try:
            limit = self.resource_limits[resource_type]
            
            if limit.current_usage < limit.max_concurrent:
                return 0.0  # Resource available immediately
            
            # Estimate based on queue size and average duration
            if limit.average_duration > 0:
                return (limit.queue_size * limit.average_duration) / limit.max_concurrent
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Wait time estimation failed: {e}")
            return 0.0
    
    async def graceful_shutdown(self, timeout_seconds: float = 300):
        """
        Gracefully shutdown the concurrency manager.
        
        Args:
            timeout_seconds: Maximum time to wait for active tasks
        """
        try:
            logger.info("Starting graceful shutdown of concurrency manager")
            self.shutdown_requested = True
            
            start_time = time.time()
            
            # Wait for active tasks to complete
            while self.active_tasks and (time.time() - start_time) < timeout_seconds:
                logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
                await trio.sleep(1)
            
            # Force cleanup of remaining tasks
            if self.active_tasks:
                logger.warning(f"Force cleaning up {len(self.active_tasks)} remaining tasks")
                for task_id in list(self.active_tasks):
                    await self.cancel_task_resources(task_id)
            
            logger.info("Concurrency manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")


# Global instance
concurrency_manager = ConcurrencyManager()