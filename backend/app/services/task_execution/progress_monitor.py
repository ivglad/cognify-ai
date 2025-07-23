"""
Task progress monitoring and real-time status tracking with trio.
"""
import time
import json
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import trio
from collections import defaultdict, deque

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ProgressEventType(str, Enum):
    """Progress event type enumeration."""
    TASK_STARTED = "task_started"
    PROGRESS_UPDATE = "progress_update"
    STAGE_COMPLETED = "stage_completed"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    HEARTBEAT = "heartbeat"
    RESOURCE_ACQUIRED = "resource_acquired"
    RESOURCE_RELEASED = "resource_released"


@dataclass
class ProgressEvent:
    """Progress event data structure."""
    event_id: str
    task_id: str
    event_type: ProgressEventType
    timestamp: datetime
    progress: float
    message: str
    stage: Optional[str] = None
    metadata: Dict[str, Any] = None
    correlation_id: Optional[str] = None


@dataclass
class TaskProgress:
    """Task progress tracking data."""
    task_id: str
    task_type: str
    status: str
    progress: float
    current_stage: Optional[str]
    stages_completed: List[str]
    total_stages: int
    started_at: datetime
    last_update: datetime
    estimated_completion: Optional[datetime]
    processing_rate: float  # items per second
    error_count: int
    warning_count: int
    metadata: Dict[str, Any]
    correlation_id: Optional[str] = None


@dataclass
class WorkerHealth:
    """Worker health monitoring data."""
    worker_id: str
    task_id: Optional[str]
    status: str
    last_heartbeat: datetime
    cpu_usage: float
    memory_usage: float
    active_connections: int
    tasks_processed: int
    errors_count: int
    uptime_seconds: float


class ProgressMonitor:
    """
    Real-time task progress monitoring and health tracking system.
    """
    
    def __init__(self):
        # Progress tracking
        self.task_progress: Dict[str, TaskProgress] = {}
        self.progress_events: deque = deque(maxlen=10000)  # Keep last 10k events
        
        # Event subscribers
        self.event_subscribers: Dict[ProgressEventType, List[Callable]] = defaultdict(list)
        self.task_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Worker health monitoring
        self.worker_health: Dict[str, WorkerHealth] = {}
        self.health_check_interval = 30  # seconds
        
        # Performance metrics
        self.performance_metrics = {
            'total_events': 0,
            'events_per_second': 0.0,
            'average_task_duration': 0.0,
            'task_completion_rate': 0.0,
            'error_rate': 0.0
        }
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_nursery = None
        
        # Alerting
        self.alert_thresholds = {
            'high_error_rate': 0.1,  # 10% error rate
            'slow_progress_rate': 0.01,  # Less than 1% progress per minute
            'worker_timeout': 300,  # 5 minutes without heartbeat
            'high_memory_usage': 0.9,  # 90% memory usage
            'high_cpu_usage': 0.8  # 80% CPU usage
        }
        
        self.active_alerts: Set[str] = set()
    
    async def start_monitoring(self):
        """Start the progress monitoring system."""
        try:
            if self.monitoring_active:
                return
            
            logger.info("Starting progress monitoring system")
            self.monitoring_active = True
            
            # Start monitoring tasks
            async with trio.open_nursery() as nursery:
                self.monitoring_nursery = nursery
                
                # Start event processor
                nursery.start_soon(self._event_processor)
                
                # Start health monitor
                nursery.start_soon(self._health_monitor)
                
                # Start metrics calculator
                nursery.start_soon(self._metrics_calculator)
                
                # Start alert monitor
                nursery.start_soon(self._alert_monitor)
                
                # Keep monitoring active
                await self._monitoring_loop()
                
        except Exception as e:
            logger.error(f"Progress monitoring startup failed: {e}")
            self.monitoring_active = False
            raise
    
    async def record_progress_event(self, 
                                  task_id: str,
                                  event_type: ProgressEventType,
                                  progress: float,
                                  message: str,
                                  stage: Optional[str] = None,
                                  metadata: Optional[Dict[str, Any]] = None,
                                  correlation_id: Optional[str] = None):
        """
        Record a progress event.
        
        Args:
            task_id: Task ID
            event_type: Type of progress event
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            stage: Current processing stage
            metadata: Additional event metadata
            correlation_id: Correlation ID for tracking
        """
        try:
            import uuid
            
            event = ProgressEvent(
                event_id=str(uuid.uuid4()),
                task_id=task_id,
                event_type=event_type,
                timestamp=datetime.now(),
                progress=max(0.0, min(1.0, progress)),
                message=message,
                stage=stage,
                metadata=metadata or {},
                correlation_id=correlation_id
            )
            
            # Add to events queue
            self.progress_events.append(event)
            
            # Update task progress
            await self._update_task_progress(event)
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            # Update metrics
            self.performance_metrics['total_events'] += 1
            
            logger.debug(f"Recorded progress event for task {task_id}: {event_type} - {message}")
            
        except Exception as e:
            logger.error(f"Progress event recording failed: {e}")
    
    async def start_task_tracking(self, 
                                task_id: str,
                                task_type: str,
                                total_stages: int = 1,
                                correlation_id: Optional[str] = None):
        """
        Start tracking progress for a task.
        
        Args:
            task_id: Task ID
            task_type: Type of task
            total_stages: Total number of processing stages
            correlation_id: Correlation ID for tracking
        """
        try:
            progress = TaskProgress(
                task_id=task_id,
                task_type=task_type,
                status="started",
                progress=0.0,
                current_stage=None,
                stages_completed=[],
                total_stages=total_stages,
                started_at=datetime.now(),
                last_update=datetime.now(),
                estimated_completion=None,
                processing_rate=0.0,
                error_count=0,
                warning_count=0,
                metadata={},
                correlation_id=correlation_id
            )
            
            self.task_progress[task_id] = progress
            
            # Record start event
            await self.record_progress_event(
                task_id=task_id,
                event_type=ProgressEventType.TASK_STARTED,
                progress=0.0,
                message=f"Started {task_type} task",
                correlation_id=correlation_id
            )
            
            logger.info(f"Started tracking task {task_id} ({task_type})")
            
        except Exception as e:
            logger.error(f"Task tracking start failed: {e}")
    
    async def update_task_progress(self, 
                                 task_id: str,
                                 progress: float,
                                 message: str,
                                 stage: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None):
        """
        Update task progress.
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            stage: Current processing stage
            metadata: Additional metadata
        """
        try:
            if task_id not in self.task_progress:
                logger.warning(f"Task {task_id} not found in progress tracking")
                return
            
            # Record progress event
            await self.record_progress_event(
                task_id=task_id,
                event_type=ProgressEventType.PROGRESS_UPDATE,
                progress=progress,
                message=message,
                stage=stage,
                metadata=metadata,
                correlation_id=self.task_progress[task_id].correlation_id
            )
            
        except Exception as e:
            logger.error(f"Task progress update failed: {e}")
    
    async def complete_task_stage(self, 
                                task_id: str,
                                stage_name: str,
                                message: str = ""):
        """
        Mark a task stage as completed.
        
        Args:
            task_id: Task ID
            stage_name: Name of completed stage
            message: Completion message
        """
        try:
            if task_id not in self.task_progress:
                return
            
            progress_info = self.task_progress[task_id]
            
            # Add to completed stages
            if stage_name not in progress_info.stages_completed:
                progress_info.stages_completed.append(stage_name)
            
            # Calculate progress based on completed stages
            stage_progress = len(progress_info.stages_completed) / progress_info.total_stages
            
            # Record stage completion event
            await self.record_progress_event(
                task_id=task_id,
                event_type=ProgressEventType.STAGE_COMPLETED,
                progress=stage_progress,
                message=message or f"Completed stage: {stage_name}",
                stage=stage_name,
                correlation_id=progress_info.correlation_id
            )
            
        except Exception as e:
            logger.error(f"Task stage completion failed: {e}")
    
    async def complete_task(self, 
                          task_id: str,
                          message: str = "Task completed successfully",
                          result: Optional[Dict[str, Any]] = None):
        """
        Mark a task as completed.
        
        Args:
            task_id: Task ID
            message: Completion message
            result: Task result data
        """
        try:
            if task_id not in self.task_progress:
                return
            
            progress_info = self.task_progress[task_id]
            progress_info.status = "completed"
            progress_info.progress = 1.0
            progress_info.last_update = datetime.now()
            
            # Record completion event
            await self.record_progress_event(
                task_id=task_id,
                event_type=ProgressEventType.TASK_COMPLETED,
                progress=1.0,
                message=message,
                metadata={'result': result} if result else None,
                correlation_id=progress_info.correlation_id
            )
            
            logger.info(f"Task {task_id} completed")
            
        except Exception as e:
            logger.error(f"Task completion failed: {e}")
    
    async def fail_task(self, 
                       task_id: str,
                       error_message: str,
                       error_details: Optional[Dict[str, Any]] = None):
        """
        Mark a task as failed.
        
        Args:
            task_id: Task ID
            error_message: Error message
            error_details: Additional error details
        """
        try:
            if task_id not in self.task_progress:
                return
            
            progress_info = self.task_progress[task_id]
            progress_info.status = "failed"
            progress_info.error_count += 1
            progress_info.last_update = datetime.now()
            
            # Record failure event
            await self.record_progress_event(
                task_id=task_id,
                event_type=ProgressEventType.TASK_FAILED,
                progress=progress_info.progress,
                message=error_message,
                metadata={'error_details': error_details} if error_details else None,
                correlation_id=progress_info.correlation_id
            )
            
            logger.error(f"Task {task_id} failed: {error_message}")
            
        except Exception as e:
            logger.error(f"Task failure recording failed: {e}")
    
    async def record_worker_heartbeat(self, 
                                    worker_id: str,
                                    task_id: Optional[str] = None,
                                    cpu_usage: float = 0.0,
                                    memory_usage: float = 0.0,
                                    active_connections: int = 0):
        """
        Record worker heartbeat for health monitoring.
        
        Args:
            worker_id: Worker ID
            task_id: Current task ID (if any)
            cpu_usage: CPU usage percentage (0.0 to 1.0)
            memory_usage: Memory usage percentage (0.0 to 1.0)
            active_connections: Number of active connections
        """
        try:
            current_time = datetime.now()
            
            if worker_id in self.worker_health:
                health = self.worker_health[worker_id]
                health.last_heartbeat = current_time
                health.task_id = task_id
                health.cpu_usage = cpu_usage
                health.memory_usage = memory_usage
                health.active_connections = active_connections
            else:
                health = WorkerHealth(
                    worker_id=worker_id,
                    task_id=task_id,
                    status="active",
                    last_heartbeat=current_time,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    active_connections=active_connections,
                    tasks_processed=0,
                    errors_count=0,
                    uptime_seconds=0.0
                )
                self.worker_health[worker_id] = health
            
            # Record heartbeat event if task is active
            if task_id:
                await self.record_progress_event(
                    task_id=task_id,
                    event_type=ProgressEventType.HEARTBEAT,
                    progress=self.task_progress.get(task_id, TaskProgress(
                        task_id="", task_type="", status="", progress=0.0,
                        current_stage=None, stages_completed=[], total_stages=1,
                        started_at=current_time, last_update=current_time,
                        estimated_completion=None, processing_rate=0.0,
                        error_count=0, warning_count=0, metadata={}
                    )).progress,
                    message=f"Worker {worker_id} heartbeat",
                    metadata={
                        'worker_id': worker_id,
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'active_connections': active_connections
                    }
                )
            
        except Exception as e:
            logger.error(f"Worker heartbeat recording failed: {e}")
    
    def subscribe_to_events(self, 
                          event_type: ProgressEventType,
                          callback: Callable[[ProgressEvent], None]):
        """
        Subscribe to progress events.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Callback function to handle events
        """
        self.event_subscribers[event_type].append(callback)
        logger.info(f"Added subscriber for {event_type} events")
    
    def subscribe_to_task(self, 
                         task_id: str,
                         callback: Callable[[ProgressEvent], None]):
        """
        Subscribe to events for a specific task.
        
        Args:
            task_id: Task ID to subscribe to
            callback: Callback function to handle events
        """
        self.task_subscribers[task_id].append(callback)
        logger.info(f"Added subscriber for task {task_id}")
    
    async def get_task_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current progress for a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task progress information or None if not found
        """
        try:
            if task_id in self.task_progress:
                progress = self.task_progress[task_id]
                
                # Calculate estimated completion
                if progress.progress > 0 and progress.processing_rate > 0:
                    remaining_progress = 1.0 - progress.progress
                    estimated_seconds = remaining_progress / progress.processing_rate
                    estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
                else:
                    estimated_completion = None
                
                return {
                    **asdict(progress),
                    'estimated_completion': estimated_completion.isoformat() if estimated_completion else None,
                    'elapsed_time_seconds': (datetime.now() - progress.started_at).total_seconds(),
                    'stages_progress': f"{len(progress.stages_completed)}/{progress.total_stages}"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Task progress retrieval failed: {e}")
            return None
    
    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        try:
            current_time = datetime.now()
            
            # Calculate active tasks
            active_tasks = [
                task_id for task_id, progress in self.task_progress.items()
                if progress.status in ["started", "running"]
            ]
            
            # Calculate worker health summary
            healthy_workers = 0
            unhealthy_workers = 0
            
            for worker_id, health in self.worker_health.items():
                time_since_heartbeat = (current_time - health.last_heartbeat).total_seconds()
                if time_since_heartbeat < self.alert_thresholds['worker_timeout']:
                    healthy_workers += 1
                else:
                    unhealthy_workers += 1
            
            return {
                'monitoring_active': self.monitoring_active,
                'total_tasks_tracked': len(self.task_progress),
                'active_tasks': len(active_tasks),
                'active_task_ids': active_tasks,
                'total_events': len(self.progress_events),
                'performance_metrics': self.performance_metrics,
                'worker_health': {
                    'total_workers': len(self.worker_health),
                    'healthy_workers': healthy_workers,
                    'unhealthy_workers': unhealthy_workers
                },
                'event_subscribers': {
                    event_type.value: len(callbacks)
                    for event_type, callbacks in self.event_subscribers.items()
                },
                'task_subscribers': len(self.task_subscribers),
                'active_alerts': list(self.active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Monitoring stats retrieval failed: {e}")
            return {'error': str(e)}
    
    async def _update_task_progress(self, event: ProgressEvent):
        """Update task progress based on event."""
        try:
            if event.task_id not in self.task_progress:
                return
            
            progress = self.task_progress[event.task_id]
            progress.progress = event.progress
            progress.last_update = event.timestamp
            progress.current_stage = event.stage
            
            # Update processing rate
            elapsed_time = (event.timestamp - progress.started_at).total_seconds()
            if elapsed_time > 0:
                progress.processing_rate = event.progress / elapsed_time
            
            # Update metadata
            if event.metadata:
                progress.metadata.update(event.metadata)
            
        except Exception as e:
            logger.error(f"Task progress update failed: {e}")
    
    async def _notify_subscribers(self, event: ProgressEvent):
        """Notify event subscribers."""
        try:
            # Notify event type subscribers
            for callback in self.event_subscribers[event.event_type]:
                try:
                    await trio.to_thread.run_sync(callback, event)
                except Exception as e:
                    logger.error(f"Event subscriber callback failed: {e}")
            
            # Notify task-specific subscribers
            for callback in self.task_subscribers.get(event.task_id, []):
                try:
                    await trio.to_thread.run_sync(callback, event)
                except Exception as e:
                    logger.error(f"Task subscriber callback failed: {e}")
                    
        except Exception as e:
            logger.error(f"Subscriber notification failed: {e}")
    
    async def _event_processor(self):
        """Process events in background."""
        try:
            while self.monitoring_active:
                # Process any pending events
                # This could include batching, persistence, etc.
                await trio.sleep(1)
                
        except Exception as e:
            logger.error(f"Event processor failed: {e}")
    
    async def _health_monitor(self):
        """Monitor worker health."""
        try:
            while self.monitoring_active:
                current_time = datetime.now()
                
                # Check worker health
                for worker_id, health in list(self.worker_health.items()):
                    time_since_heartbeat = (current_time - health.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.alert_thresholds['worker_timeout']:
                        health.status = "timeout"
                        alert_key = f"worker_timeout_{worker_id}"
                        if alert_key not in self.active_alerts:
                            self.active_alerts.add(alert_key)
                            logger.warning(f"Worker {worker_id} timeout detected")
                    else:
                        health.status = "active"
                        alert_key = f"worker_timeout_{worker_id}"
                        self.active_alerts.discard(alert_key)
                
                await trio.sleep(self.health_check_interval)
                
        except Exception as e:
            logger.error(f"Health monitor failed: {e}")
    
    async def _metrics_calculator(self):
        """Calculate performance metrics."""
        try:
            last_event_count = 0
            last_calculation_time = time.time()
            
            while self.monitoring_active:
                current_time = time.time()
                current_event_count = self.performance_metrics['total_events']
                
                # Calculate events per second
                time_diff = current_time - last_calculation_time
                if time_diff > 0:
                    events_diff = current_event_count - last_event_count
                    self.performance_metrics['events_per_second'] = events_diff / time_diff
                
                # Calculate other metrics
                completed_tasks = [
                    p for p in self.task_progress.values()
                    if p.status == "completed"
                ]
                
                if completed_tasks:
                    durations = [
                        (p.last_update - p.started_at).total_seconds()
                        for p in completed_tasks
                    ]
                    self.performance_metrics['average_task_duration'] = sum(durations) / len(durations)
                
                # Update for next calculation
                last_event_count = current_event_count
                last_calculation_time = current_time
                
                await trio.sleep(60)  # Calculate every minute
                
        except Exception as e:
            logger.error(f"Metrics calculator failed: {e}")
    
    async def _alert_monitor(self):
        """Monitor for alert conditions."""
        try:
            while self.monitoring_active:
                # Check error rates, performance, etc.
                # This would implement alerting logic
                await trio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Alert monitor failed: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                await trio.sleep(1)
                
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
    
    async def stop_monitoring(self):
        """Stop the progress monitoring system."""
        try:
            logger.info("Stopping progress monitoring system")
            self.monitoring_active = False
            
            # Wait a bit for background tasks to finish
            await trio.sleep(2)
            
            logger.info("Progress monitoring stopped")
            
        except Exception as e:
            logger.error(f"Progress monitoring stop failed: {e}")


# Global instance
progress_monitor = ProgressMonitor()