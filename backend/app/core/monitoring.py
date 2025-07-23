"""
Comprehensive system monitoring and health check system.
"""
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json

import trio
from sqlalchemy import text

from app.core.config import settings
from app.core.logging_config import get_logger, log_performance_metric, performance_monitor
from app.db.session import get_db_session
from app.core.cache import cache_manager
from app.core.elasticsearch_client import elasticsearch_client

logger = get_logger("ragflow.monitoring")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Types of system components."""
    DATABASE = "database"
    CACHE = "cache"
    SEARCH = "search"
    STORAGE = "storage"
    EXTERNAL_API = "external_api"
    COMPUTE = "compute"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    component_type: ComponentType
    status: HealthStatus
    response_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    disk_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, HealthCheckResult] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000
        self.check_interval = 30  # seconds
        self.running = False
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_health_check("database", self._check_database)
        self.register_health_check("cache", self._check_cache)
        self.register_health_check("elasticsearch", self._check_elasticsearch)
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("memory_usage", self._check_memory_usage)
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_health_check(self, component: str) -> HealthCheckResult:
        """Run a specific health check."""
        if component not in self.health_checks:
            return HealthCheckResult(
                component=component,
                component_type=ComponentType.COMPUTE,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=f"Health check not found: {component}"
            )
        
        start_time = time.time()
        try:
            result = await self.health_checks[component]()
            result.response_time = time.time() - start_time
            self.last_check_results[component] = result
            
            # Log performance metric
            log_performance_metric(
                "health_check", 
                component, 
                result.response_time, 
                result.status != HealthStatus.CRITICAL
            )
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            result = HealthCheckResult(
                component=component,
                component_type=ComponentType.COMPUTE,
                status=HealthStatus.CRITICAL,
                response_time=response_time,
                details={},
                error_message=str(e)
            )
            self.last_check_results[component] = result
            
            logger.error(f"Health check failed for {component}: {e}")
            return result
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        async with trio.open_nursery() as nursery:
            async def run_check(component: str):
                results[component] = await self.run_health_check(component)
            
            for component in self.health_checks:
                nursery.start_soon(run_check, component)
        
        return results
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_results = await self.run_all_health_checks()
        system_metrics = self._collect_system_metrics()
        
        # Determine overall health status
        overall_status = self._determine_overall_status(health_results)
        
        # Calculate uptime
        uptime = self._get_system_uptime()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "health_checks": {
                name: asdict(result) for name, result in health_results.items()
            },
            "system_metrics": asdict(system_metrics),
            "performance_summary": self._get_performance_summary()
        }
    
    def _determine_overall_status(self, health_results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system health status."""
        if not health_results:
            return HealthStatus.CRITICAL
        
        statuses = [result.status for result in health_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network I/O
            network_io = psutil.net_io_counters()._asdict()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()._asdict()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]  # Windows doesn't have load average
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                disk_io=disk_io,
                process_count=process_count,
                load_average=load_average
            )
            
            # Store in history
            self.system_metrics_history.append(metrics)
            if len(self.system_metrics_history) > self.max_history_size:
                self.system_metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                disk_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def _get_system_uptime(self) -> float:
        """Get system uptime in seconds."""
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            return 0.0
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent metrics."""
        if not self.system_metrics_history:
            return {}
        
        recent_metrics = self.system_metrics_history[-10:]  # Last 10 measurements
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "avg_memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            "avg_disk_percent": sum(m.disk_percent for m in recent_metrics) / len(recent_metrics),
            "avg_process_count": sum(m.process_count for m in recent_metrics) / len(recent_metrics),
            "measurements_count": len(recent_metrics)
        }
    
    # Health check implementations
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            
            async with get_db_session() as session:
                # Simple query to test connectivity
                result = await session.execute(text("SELECT 1"))
                await result.fetchone()
            
            response_time = time.time() - start_time
            
            if response_time > 5.0:
                status = HealthStatus.DEGRADED
                details = {"warning": "Database response time is slow"}
            elif response_time > 1.0:
                status = HealthStatus.DEGRADED
                details = {"warning": "Database response time is elevated"}
            else:
                status = HealthStatus.HEALTHY
                details = {"connection": "ok"}
            
            return HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=status,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=str(e)
            )
    
    async def _check_cache(self) -> HealthCheckResult:
        """Check cache (Redis) connectivity and performance."""
        try:
            start_time = time.time()
            
            # Test cache connectivity
            test_key = "health_check_test"
            test_value = "test_value"
            
            await cache_manager.set(test_key, test_value, ttl=10)
            retrieved_value = await cache_manager.get(test_key)
            await cache_manager.delete(test_key)
            
            response_time = time.time() - start_time
            
            if retrieved_value != test_value:
                return HealthCheckResult(
                    component="cache",
                    component_type=ComponentType.CACHE,
                    status=HealthStatus.UNHEALTHY,
                    response_time=response_time,
                    details={},
                    error_message="Cache read/write test failed"
                )
            
            if response_time > 1.0:
                status = HealthStatus.DEGRADED
                details = {"warning": "Cache response time is slow"}
            else:
                status = HealthStatus.HEALTHY
                details = {"connection": "ok", "read_write": "ok"}
            
            return HealthCheckResult(
                component="cache",
                component_type=ComponentType.CACHE,
                status=status,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="cache",
                component_type=ComponentType.CACHE,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=str(e)
            )
    
    async def _check_elasticsearch(self) -> HealthCheckResult:
        """Check Elasticsearch connectivity and cluster health."""
        try:
            start_time = time.time()
            
            # Check cluster health
            health_response = await elasticsearch_client.cluster.health()
            response_time = time.time() - start_time
            
            cluster_status = health_response.get('status', 'red')
            
            if cluster_status == 'green':
                status = HealthStatus.HEALTHY
            elif cluster_status == 'yellow':
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            details = {
                "cluster_status": cluster_status,
                "number_of_nodes": health_response.get('number_of_nodes', 0),
                "active_primary_shards": health_response.get('active_primary_shards', 0),
                "active_shards": health_response.get('active_shards', 0)
            }
            
            return HealthCheckResult(
                component="elasticsearch",
                component_type=ComponentType.SEARCH,
                status=status,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="elasticsearch",
                component_type=ComponentType.SEARCH,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=str(e)
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            metrics = self._collect_system_metrics()
            
            # Determine status based on resource usage
            if metrics.cpu_percent > 90 or metrics.memory_percent > 90:
                status = HealthStatus.CRITICAL
            elif metrics.cpu_percent > 80 or metrics.memory_percent > 80:
                status = HealthStatus.UNHEALTHY
            elif metrics.cpu_percent > 70 or metrics.memory_percent > 70:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            details = {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "load_average": metrics.load_average,
                "process_count": metrics.process_count
            }
            
            return HealthCheckResult(
                component="system_resources",
                component_type=ComponentType.COMPUTE,
                status=status,
                response_time=0.0,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                component_type=ComponentType.COMPUTE,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=str(e)
            )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space usage."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
            elif disk_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif disk_percent > 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            details = {
                "disk_percent": disk_percent,
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2)
            }
            
            return HealthCheckResult(
                component="disk_space",
                component_type=ComponentType.DISK,
                status=status,
                response_time=0.0,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="disk_space",
                component_type=ComponentType.DISK,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=str(e)
            )
    
    async def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage details."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
            elif memory.percent > 90:
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            details = {
                "memory_percent": memory.percent,
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "swap_percent": swap.percent,
                "swap_used_gb": round(swap.used / (1024**3), 2)
            }
            
            return HealthCheckResult(
                component="memory_usage",
                component_type=ComponentType.MEMORY,
                status=status,
                response_time=0.0,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="memory_usage",
                component_type=ComponentType.MEMORY,
                status=HealthStatus.CRITICAL,
                response_time=0.0,
                details={},
                error_message=str(e)
            )
    
    async def start_continuous_monitoring(self):
        """Start continuous health monitoring."""
        self.running = True
        logger.info("Starting continuous health monitoring")
        
        while self.running:
            try:
                health_status = await self.get_system_health()
                
                # Log overall health status
                logger.info(
                    "System health check completed",
                    overall_status=health_status["overall_status"],
                    uptime_seconds=health_status["uptime_seconds"],
                    component_count=len(health_status["health_checks"])
                )
                
                # Log critical issues
                for component, result in health_status["health_checks"].items():
                    if result["status"] in ["critical", "unhealthy"]:
                        logger.error(
                            "Component health issue detected",
                            component=component,
                            status=result["status"],
                            error_message=result.get("error_message"),
                            details=result["details"]
                        )
                
                await trio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await trio.sleep(self.check_interval)
    
    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring."""
        self.running = False
        logger.info("Stopping continuous health monitoring")
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get system metrics history for the specified number of hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        filtered_metrics = [
            asdict(metrics) for metrics in self.system_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        return filtered_metrics
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        if not self.last_check_results:
            return {"status": "unknown", "components": {}}
        
        component_statuses = {}
        overall_status = HealthStatus.HEALTHY
        
        for component, result in self.last_check_results.items():
            component_statuses[component] = {
                "status": result.status.value,
                "last_check": result.timestamp,
                "response_time": result.response_time
            }
            
            # Update overall status
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "overall_status": overall_status.value,
            "components": component_statuses,
            "last_update": max(result.timestamp for result in self.last_check_results.values()) if self.last_check_results else 0
        }


# Global health checker instance
health_checker = HealthChecker()


# Convenience functions
async def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return await health_checker.get_system_health()


async def run_health_check(component: str) -> HealthCheckResult:
    """Run a specific health check."""
    return await health_checker.run_health_check(component)


def get_health_summary() -> Dict[str, Any]:
    """Get health summary."""
    return health_checker.get_health_summary()


def get_metrics_history(hours: int = 1) -> List[Dict[str, Any]]:
    """Get metrics history."""
    return health_checker.get_metrics_history(hours)


async def start_monitoring():
    """Start continuous monitoring."""
    await health_checker.start_continuous_monitoring()


def stop_monitoring():
    """Stop continuous monitoring."""
    health_checker.stop_continuous_monitoring()