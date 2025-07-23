"""
Production monitoring and alerting system.
"""
import time
import psutil
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import trio
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    load_average: List[float]
    timestamp: datetime


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    active_requests: int
    total_requests: int
    avg_response_time: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    vector_store_connections: int
    llm_requests_per_minute: int
    document_processing_queue: int
    timestamp: datetime


class ProductionMonitor:
    """Production monitoring and alerting system."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.system_metrics_history: List[SystemMetrics] = []
        self.app_metrics_history: List[ApplicationMetrics] = []
        
        # Prometheus metrics
        self.request_counter = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
        self.active_requests = Gauge('http_requests_active', 'Active HTTP requests')
        self.system_cpu = Gauge('system_cpu_percent', 'System CPU usage percentage')
        self.system_memory = Gauge('system_memory_percent', 'System memory usage percentage')
        self.system_disk = Gauge('system_disk_percent', 'System disk usage percentage')
        self.database_connections = Gauge('database_connections_active', 'Active database connections')
        self.cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate percentage')
        self.llm_requests = Counter('llm_requests_total', 'Total LLM requests', ['provider', 'model'])
        self.document_processing_queue = Gauge('document_processing_queue_size', 'Document processing queue size')
        
        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 2.0,
            'response_time_critical': 5.0,
            'error_rate_warning': 5.0,
            'error_rate_critical': 10.0,
            'cache_hit_rate_warning': 70.0,
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.last_metrics_collection = None
        
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        logger.info("Starting production monitoring system")
        
        # Start monitoring tasks
        async with trio.open_nursery() as nursery:
            nursery.start_soon(self._collect_metrics_loop)
            nursery.start_soon(self._check_alerts_loop)
            nursery.start_soon(self._cleanup_old_data_loop)
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        logger.info("Stopping production monitoring system")
    
    async def _collect_metrics_loop(self):
        """Continuously collect system and application metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Update Prometheus metrics
                self.system_cpu.set(system_metrics.cpu_percent)
                self.system_memory.set(system_metrics.memory_percent)
                self.system_disk.set(system_metrics.disk_percent)
                
                # Collect application metrics
                app_metrics = await self._collect_application_metrics()
                self.app_metrics_history.append(app_metrics)
                
                # Update Prometheus metrics
                self.active_requests.set(app_metrics.active_requests)
                self.cache_hit_rate.set(app_metrics.cache_hit_rate)
                self.database_connections.set(app_metrics.database_connections)
                self.document_processing_queue.set(app_metrics.document_processing_queue)
                
                self.last_metrics_collection = datetime.utcnow()
                
                # Wait before next collection
                await trio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await trio.sleep(60)  # Wait longer on error
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_total_gb = disk.total / (1024**3)
            
            # Load average
            load_average = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_total_gb=disk_total_gb,
                load_average=load_average,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                disk_percent=0.0,
                disk_used_gb=0.0,
                disk_total_gb=0.0,
                load_average=[0.0, 0.0, 0.0],
                timestamp=datetime.utcnow()
            )
    
    async def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        try:
            # These would be collected from actual application state
            # For now, using placeholder values
            return ApplicationMetrics(
                active_requests=0,  # Would be tracked by middleware
                total_requests=0,   # Would be tracked by middleware
                avg_response_time=0.0,  # Would be calculated from request history
                error_rate=0.0,     # Would be calculated from error tracking
                cache_hit_rate=0.0, # Would be retrieved from cache service
                database_connections=0,  # Would be retrieved from DB pool
                vector_store_connections=0,  # Would be retrieved from vector store
                llm_requests_per_minute=0,   # Would be tracked from LLM service
                document_processing_queue=0, # Would be retrieved from task queue
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return ApplicationMetrics(
                active_requests=0,
                total_requests=0,
                avg_response_time=0.0,
                error_rate=0.0,
                cache_hit_rate=0.0,
                database_connections=0,
                vector_store_connections=0,
                llm_requests_per_minute=0,
                document_processing_queue=0,
                timestamp=datetime.utcnow()
            )
    
    async def _check_alerts_loop(self):
        """Continuously check for alert conditions."""
        while self.monitoring_active:
            try:
                await self._check_system_alerts()
                await self._check_application_alerts()
                await trio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error checking alerts: {e}")
                await trio.sleep(120)  # Wait longer on error
    
    async def _check_system_alerts(self):
        """Check system resource alerts."""
        if not self.system_metrics_history:
            return
            
        latest_metrics = self.system_metrics_history[-1]
        
        # CPU alerts
        if latest_metrics.cpu_percent >= self.thresholds['cpu_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "High CPU Usage",
                f"CPU usage is {latest_metrics.cpu_percent:.1f}% (critical threshold: {self.thresholds['cpu_critical']}%)",
                "system"
            )
        elif latest_metrics.cpu_percent >= self.thresholds['cpu_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                "Elevated CPU Usage",
                f"CPU usage is {latest_metrics.cpu_percent:.1f}% (warning threshold: {self.thresholds['cpu_warning']}%)",
                "system"
            )
        
        # Memory alerts
        if latest_metrics.memory_percent >= self.thresholds['memory_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "High Memory Usage",
                f"Memory usage is {latest_metrics.memory_percent:.1f}% (critical threshold: {self.thresholds['memory_critical']}%)",
                "system"
            )
        elif latest_metrics.memory_percent >= self.thresholds['memory_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                "Elevated Memory Usage",
                f"Memory usage is {latest_metrics.memory_percent:.1f}% (warning threshold: {self.thresholds['memory_warning']}%)",
                "system"
            )
        
        # Disk alerts
        if latest_metrics.disk_percent >= self.thresholds['disk_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "High Disk Usage",
                f"Disk usage is {latest_metrics.disk_percent:.1f}% (critical threshold: {self.thresholds['disk_critical']}%)",
                "system"
            )
        elif latest_metrics.disk_percent >= self.thresholds['disk_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                "Elevated Disk Usage",
                f"Disk usage is {latest_metrics.disk_percent:.1f}% (warning threshold: {self.thresholds['disk_warning']}%)",
                "system"
            )
    
    async def _check_application_alerts(self):
        """Check application-specific alerts."""
        if not self.app_metrics_history:
            return
            
        latest_metrics = self.app_metrics_history[-1]
        
        # Response time alerts
        if latest_metrics.avg_response_time >= self.thresholds['response_time_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "High Response Time",
                f"Average response time is {latest_metrics.avg_response_time:.2f}s (critical threshold: {self.thresholds['response_time_critical']}s)",
                "application"
            )
        elif latest_metrics.avg_response_time >= self.thresholds['response_time_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                "Elevated Response Time",
                f"Average response time is {latest_metrics.avg_response_time:.2f}s (warning threshold: {self.thresholds['response_time_warning']}s)",
                "application"
            )
        
        # Error rate alerts
        if latest_metrics.error_rate >= self.thresholds['error_rate_critical']:
            await self._create_alert(
                AlertLevel.CRITICAL,
                "High Error Rate",
                f"Error rate is {latest_metrics.error_rate:.1f}% (critical threshold: {self.thresholds['error_rate_critical']}%)",
                "application"
            )
        elif latest_metrics.error_rate >= self.thresholds['error_rate_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                "Elevated Error Rate",
                f"Error rate is {latest_metrics.error_rate:.1f}% (warning threshold: {self.thresholds['error_rate_warning']}%)",
                "application"
            )
        
        # Cache hit rate alerts
        if latest_metrics.cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
            await self._create_alert(
                AlertLevel.WARNING,
                "Low Cache Hit Rate",
                f"Cache hit rate is {latest_metrics.cache_hit_rate:.1f}% (warning threshold: {self.thresholds['cache_hit_rate_warning']}%)",
                "application"
            )
    
    async def _create_alert(self, level: AlertLevel, title: str, message: str, component: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a new alert."""
        alert_id = f"{component}_{title.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Check if similar alert already exists and is not resolved
        existing_alert = None
        for alert in self.alerts:
            if (alert.component == component and 
                alert.title == title and 
                not alert.resolved and
                (datetime.utcnow() - alert.timestamp) < timedelta(hours=1)):
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert
            existing_alert.message = message
            existing_alert.timestamp = datetime.utcnow()
            logger.debug(f"Updated existing alert: {title}")
        else:
            # Create new alert
            alert = Alert(
                id=alert_id,
                level=level,
                title=title,
                message=message,
                component=component,
                timestamp=datetime.utcnow(),
                metadata=metadata
            )
            
            self.alerts.append(alert)
            logger.warning(f"Created {level.value} alert: {title} - {message}")
            
            # Send alert notification (would integrate with external systems)
            await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification to external systems."""
        try:
            # This would integrate with external alerting systems like:
            # - Slack/Discord webhooks
            # - Email notifications
            # - PagerDuty
            # - Prometheus Alertmanager
            
            logger.info(f"Alert notification sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    async def _cleanup_old_data_loop(self):
        """Clean up old metrics and resolved alerts."""
        while self.monitoring_active:
            try:
                await self._cleanup_old_metrics()
                await self._cleanup_old_alerts()
                await trio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                await trio.sleep(3600)
    
    async def _cleanup_old_metrics(self):
        """Remove old metrics data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep 24 hours of data
        
        # Clean system metrics
        self.system_metrics_history = [
            m for m in self.system_metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Clean application metrics
        self.app_metrics_history = [
            m for m in self.app_metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        logger.debug("Cleaned up old metrics data")
    
    async def _cleanup_old_alerts(self):
        """Remove old resolved alerts."""
        cutoff_time = datetime.utcnow() - timedelta(days=7)  # Keep resolved alerts for 7 days
        
        self.alerts = [
            alert for alert in self.alerts
            if not alert.resolved or (alert.resolved_at and alert.resolved_at > cutoff_time)
        ]
        
        logger.debug("Cleaned up old alerts")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and application metrics."""
        system_metrics = None
        app_metrics = None
        
        if self.system_metrics_history:
            system_metrics = asdict(self.system_metrics_history[-1])
        
        if self.app_metrics_history:
            app_metrics = asdict(self.app_metrics_history[-1])
        
        return {
            'system_metrics': system_metrics,
            'application_metrics': app_metrics,
            'monitoring_active': self.monitoring_active,
            'last_collection': self.last_metrics_collection.isoformat() if self.last_metrics_collection else None,
            'alerts_count': len([a for a in self.alerts if not a.resolved])
        }
    
    async def get_alerts(self, resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by resolved status."""
        filtered_alerts = self.alerts
        
        if resolved is not None:
            filtered_alerts = [a for a in self.alerts if a.resolved == resolved]
        
        return [asdict(alert) for alert in filtered_alerts]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert by ID."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                logger.info(f"Resolved alert: {alert.title}")
                return True
        
        return False
    
    async def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest().decode('utf-8')
    
    async def health_check(self) -> Dict[str, Any]:
        """Get monitoring system health status."""
        return {
            'monitoring_active': self.monitoring_active,
            'last_metrics_collection': self.last_metrics_collection.isoformat() if self.last_metrics_collection else None,
            'system_metrics_count': len(self.system_metrics_history),
            'app_metrics_count': len(self.app_metrics_history),
            'active_alerts': len([a for a in self.alerts if not a.resolved]),
            'total_alerts': len(self.alerts)
        }


# Global monitor instance
production_monitor = ProductionMonitor()