"""
API endpoints for system monitoring and health checks.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.core.monitoring import (
    get_system_health, run_health_check, get_health_summary,
    get_metrics_history, health_checker
)
from app.core.logging_config import (
    get_performance_metrics, reset_performance_metrics,
    get_logger, log_security_event
)
from app.core.production_monitoring import production_monitor
from app.api.dependencies.auth import get_current_user

router = APIRouter()
logger = get_logger("ragflow.monitoring.api")


@router.get(
    "/health",
    summary="Get system health status",
    description="Get comprehensive system health status including all components"
)
async def get_health_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive system health status."""
    try:
        health_status = await get_system_health()
        
        logger.info(
            "Health status requested",
            user_id=current_user.get("id"),
            overall_status=health_status["overall_status"]
        )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/health/summary",
    summary="Get health summary",
    description="Get a quick summary of system health status"
)
async def get_health_status_summary(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get quick health summary."""
    try:
        summary = get_health_summary()
        
        logger.info(
            "Health summary requested",
            user_id=current_user.get("id"),
            overall_status=summary.get("overall_status")
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"Health summary check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health summary failed: {str(e)}"
        )


@router.get(
    "/health/{component}",
    summary="Check specific component health",
    description="Run health check for a specific system component"
)
async def check_component_health(
    component: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Check health of a specific component."""
    try:
        result = await run_health_check(component)
        
        logger.info(
            "Component health check requested",
            user_id=current_user.get("id"),
            component=component,
            status=result.status.value
        )
        
        return {
            "component": result.component,
            "component_type": result.component_type.value,
            "status": result.status.value,
            "response_time": result.response_time,
            "details": result.details,
            "error_message": result.error_message,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Component health check failed for {component}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Component health check failed: {str(e)}"
        )


@router.get(
    "/metrics",
    summary="Get system metrics",
    description="Get current system resource metrics"
)
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current system metrics."""
    try:
        # Get latest system metrics from health checker
        if health_checker.system_metrics_history:
            latest_metrics = health_checker.system_metrics_history[-1]
            metrics_dict = {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "disk_percent": latest_metrics.disk_percent,
                "network_io": latest_metrics.network_io,
                "disk_io": latest_metrics.disk_io,
                "process_count": latest_metrics.process_count,
                "load_average": latest_metrics.load_average,
                "timestamp": latest_metrics.timestamp
            }
        else:
            metrics_dict = {
                "error": "No metrics available yet"
            }
        
        logger.info(
            "System metrics requested",
            user_id=current_user.get("id"),
            cpu_percent=metrics_dict.get("cpu_percent"),
            memory_percent=metrics_dict.get("memory_percent")
        )
        
        return metrics_dict
        
    except Exception as e:
        logger.error(f"System metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics retrieval failed: {str(e)}"
        )


@router.get(
    "/metrics/history",
    summary="Get metrics history",
    description="Get historical system metrics for the specified time period"
)
async def get_metrics_history_endpoint(
    hours: int = Query(1, ge=1, le=24, description="Number of hours of history to retrieve"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get historical system metrics."""
    try:
        history = get_metrics_history(hours)
        
        logger.info(
            "Metrics history requested",
            user_id=current_user.get("id"),
            hours=hours,
            data_points=len(history)
        )
        
        return {
            "hours": hours,
            "data_points": len(history),
            "metrics": history
        }
        
    except Exception as e:
        logger.error(f"Metrics history retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics history retrieval failed: {str(e)}"
        )


@router.get(
    "/performance",
    summary="Get performance metrics",
    description="Get application performance metrics and statistics"
)
async def get_performance_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get performance metrics."""
    try:
        performance_stats = get_performance_metrics()
        
        logger.info(
            "Performance metrics requested",
            user_id=current_user.get("id"),
            total_requests=performance_stats.get("request_count"),
            avg_response_time=performance_stats.get("avg_response_time")
        )
        
        return performance_stats
        
    except Exception as e:
        logger.error(f"Performance metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics retrieval failed: {str(e)}"
        )


@router.post(
    "/performance/reset",
    summary="Reset performance metrics",
    description="Reset performance metrics (admin only)"
)
async def reset_performance_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Reset performance metrics."""
    try:
        # Check if user has admin privileges
        if not current_user.get("is_admin", False):
            log_security_event(
                "unauthorized_admin_access",
                {
                    "user_id": current_user.get("id"),
                    "action": "reset_performance_metrics",
                    "ip_address": "unknown"  # Would need to get from request
                },
                "WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        reset_performance_metrics()
        
        logger.info(
            "Performance metrics reset",
            user_id=current_user.get("id"),
            admin_action=True
        )
        
        log_security_event(
            "admin_action",
            {
                "user_id": current_user.get("id"),
                "action": "reset_performance_metrics"
            },
            "INFO"
        )
        
        return {"message": "Performance metrics reset successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance metrics reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics reset failed: {str(e)}"
        )


@router.get(
    "/logs/recent",
    summary="Get recent log entries",
    description="Get recent log entries (admin only)"
)
async def get_recent_logs(
    lines: int = Query(100, ge=1, le=1000, description="Number of log lines to retrieve"),
    level: str = Query("INFO", description="Minimum log level"),
    component: Optional[str] = Query(None, description="Filter by component"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get recent log entries."""
    try:
        # Check if user has admin privileges
        if not current_user.get("is_admin", False):
            log_security_event(
                "unauthorized_admin_access",
                {
                    "user_id": current_user.get("id"),
                    "action": "access_logs",
                    "ip_address": "unknown"
                },
                "WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        # Read recent log entries from file
        log_entries = []
        try:
            with open("logs/ragflow.log", "r") as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    line = line.strip()
                    if line:
                        # Basic filtering by level and component
                        if level.upper() in line:
                            if component is None or component.lower() in line.lower():
                                log_entries.append(line)
        
        except FileNotFoundError:
            log_entries = ["Log file not found"]
        
        logger.info(
            "Recent logs requested",
            user_id=current_user.get("id"),
            lines_requested=lines,
            lines_returned=len(log_entries),
            level_filter=level,
            component_filter=component
        )
        
        log_security_event(
            "admin_action",
            {
                "user_id": current_user.get("id"),
                "action": "access_logs",
                "lines_requested": lines
            },
            "INFO"
        )
        
        return {
            "lines_requested": lines,
            "lines_returned": len(log_entries),
            "level_filter": level,
            "component_filter": component,
            "log_entries": log_entries
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recent logs retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Log retrieval failed: {str(e)}"
        )


@router.get(
    "/alerts",
    summary="Get system alerts",
    description="Get current system alerts and warnings"
)
async def get_system_alerts(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current system alerts."""
    try:
        alerts = []
        
        # Get current health status
        health_status = await get_system_health()
        
        # Generate alerts based on health status
        overall_status = health_status.get("overall_status")
        if overall_status in ["critical", "unhealthy"]:
            alerts.append({
                "type": "system_health",
                "severity": "high" if overall_status == "critical" else "medium",
                "message": f"System health is {overall_status}",
                "timestamp": health_status.get("timestamp"),
                "details": health_status.get("health_checks", {})
            })
        
        # Check individual components
        for component, result in health_status.get("health_checks", {}).items():
            if result["status"] in ["critical", "unhealthy"]:
                alerts.append({
                    "type": "component_health",
                    "severity": "high" if result["status"] == "critical" else "medium",
                    "message": f"Component {component} is {result['status']}",
                    "timestamp": result.get("timestamp"),
                    "component": component,
                    "error_message": result.get("error_message"),
                    "details": result.get("details", {})
                })
        
        # Check system metrics for alerts
        system_metrics = health_status.get("system_metrics", {})
        if system_metrics.get("cpu_percent", 0) > 90:
            alerts.append({
                "type": "resource_usage",
                "severity": "high",
                "message": f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                "timestamp": system_metrics.get("timestamp"),
                "metric": "cpu_percent",
                "value": system_metrics["cpu_percent"]
            })
        
        if system_metrics.get("memory_percent", 0) > 90:
            alerts.append({
                "type": "resource_usage",
                "severity": "high",
                "message": f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                "timestamp": system_metrics.get("timestamp"),
                "metric": "memory_percent",
                "value": system_metrics["memory_percent"]
            })
        
        if system_metrics.get("disk_percent", 0) > 90:
            alerts.append({
                "type": "resource_usage",
                "severity": "high",
                "message": f"High disk usage: {system_metrics['disk_percent']:.1f}%",
                "timestamp": system_metrics.get("timestamp"),
                "metric": "disk_percent",
                "value": system_metrics["disk_percent"]
            })
        
        logger.info(
            "System alerts requested",
            user_id=current_user.get("id"),
            alert_count=len(alerts)
        )
        
        return {
            "alert_count": len(alerts),
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System alerts retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alerts retrieval failed: {str(e)}"
        )


@router.get(
    "/status",
    summary="Get simple status check",
    description="Simple status endpoint for load balancers and monitoring tools"
)
async def get_simple_status():
    """Simple status check for load balancers."""
    try:
        # Quick health check
        summary = get_health_summary()
        overall_status = summary.get("overall_status", "unknown")
        
        if overall_status in ["healthy", "degraded"]:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ok",
                    "timestamp": datetime.now().isoformat(),
                    "health": overall_status
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "health": overall_status
                }
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Product
ion Monitoring Endpoints

@router.get(
    "/production/metrics",
    summary="Get production metrics",
    description="Get current production system and application metrics"
)
async def get_production_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current production metrics."""
    try:
        metrics = await production_monitor.get_current_metrics()
        
        logger.info(
            "Production metrics requested",
            user_id=current_user.get("id"),
            monitoring_active=metrics.get("monitoring_active")
        )
        
        return {
            "status": "success",
            "data": metrics,
            "timestamp": metrics.get("last_collection")
        }
        
    except Exception as e:
        logger.error(f"Production metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Production metrics retrieval failed: {str(e)}"
        )


@router.get(
    "/production/alerts",
    summary="Get production alerts",
    description="Get current production alerts and warnings"
)
async def get_production_alerts(
    resolved: Optional[bool] = Query(None, description="Filter by resolved status"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get production alerts."""
    try:
        alerts = await production_monitor.get_alerts(resolved=resolved)
        
        logger.info(
            "Production alerts requested",
            user_id=current_user.get("id"),
            alert_count=len(alerts),
            resolved_filter=resolved
        )
        
        return {
            "status": "success",
            "data": alerts,
            "count": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Production alerts retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Production alerts retrieval failed: {str(e)}"
        )


@router.post(
    "/production/alerts/{alert_id}/resolve",
    summary="Resolve production alert",
    description="Resolve a specific production alert"
)
async def resolve_production_alert(
    alert_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Resolve a production alert."""
    try:
        success = await production_monitor.resolve_alert(alert_id)
        
        if success:
            logger.info(
                "Production alert resolved",
                user_id=current_user.get("id"),
                alert_id=alert_id
            )
            
            return {
                "status": "success",
                "message": f"Alert {alert_id} resolved successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found or already resolved"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Production alert resolution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert resolution failed: {str(e)}"
        )


@router.get(
    "/production/dashboard",
    summary="Get production dashboard",
    description="Get comprehensive production monitoring dashboard data"
)
async def get_production_dashboard(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get production monitoring dashboard data."""
    try:
        # Get current metrics
        current_metrics = await production_monitor.get_current_metrics()
        
        # Get active alerts
        active_alerts = await production_monitor.get_alerts(resolved=False)
        
        # Get monitoring health
        monitor_health = await production_monitor.health_check()
        
        dashboard_data = {
            "system_status": {
                "cpu_percent": current_metrics.get("system_metrics", {}).get("cpu_percent", 0),
                "memory_percent": current_metrics.get("system_metrics", {}).get("memory_percent", 0),
                "disk_percent": current_metrics.get("system_metrics", {}).get("disk_percent", 0),
            },
            "application_status": {
                "active_requests": current_metrics.get("application_metrics", {}).get("active_requests", 0),
                "avg_response_time": current_metrics.get("application_metrics", {}).get("avg_response_time", 0),
                "error_rate": current_metrics.get("application_metrics", {}).get("error_rate", 0),
                "cache_hit_rate": current_metrics.get("application_metrics", {}).get("cache_hit_rate", 0),
            },
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.get("level") == "critical"]),
                "warning_count": len([a for a in active_alerts if a.get("level") == "warning"]),
                "recent_alerts": active_alerts[:5]  # Last 5 alerts
            },
            "monitoring": {
                "status": "active" if monitor_health.get("monitoring_active") else "inactive",
                "last_collection": current_metrics.get("last_collection"),
                "metrics_count": monitor_health.get("system_metrics_count", 0) + monitor_health.get("app_metrics_count", 0)
            }
        }
        
        logger.info(
            "Production dashboard requested",
            user_id=current_user.get("id"),
            monitoring_status=dashboard_data["monitoring"]["status"],
            active_alerts=dashboard_data["alerts"]["active_count"]
        )
        
        return {
            "status": "success",
            "data": dashboard_data,
            "timestamp": current_metrics.get("last_collection")
        }
        
    except Exception as e:
        logger.error(f"Production dashboard retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard data retrieval failed: {str(e)}"
        )


@router.get(
    "/production/prometheus",
    summary="Get Prometheus metrics",
    description="Get Prometheus metrics in text format"
)
async def get_prometheus_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get Prometheus metrics."""
    try:
        metrics_text = await production_monitor.get_prometheus_metrics()
        
        logger.info(
            "Prometheus metrics requested",
            user_id=current_user.get("id")
        )
        
        return JSONResponse(
            content=metrics_text,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Prometheus metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prometheus metrics retrieval failed: {str(e)}"
        )


@router.post(
    "/production/start",
    summary="Start production monitoring",
    description="Start the production monitoring system (admin only)"
)
async def start_production_monitoring(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Start production monitoring."""
    try:
        # Check if user has admin privileges
        if not current_user.get("is_admin", False):
            log_security_event(
                "unauthorized_admin_access",
                {
                    "user_id": current_user.get("id"),
                    "action": "start_production_monitoring",
                    "ip_address": "unknown"
                },
                "WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        await production_monitor.start_monitoring()
        
        logger.info(
            "Production monitoring started",
            user_id=current_user.get("id"),
            admin_action=True
        )
        
        log_security_event(
            "admin_action",
            {
                "user_id": current_user.get("id"),
                "action": "start_production_monitoring"
            },
            "INFO"
        )
        
        return {
            "status": "success",
            "message": "Production monitoring started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Production monitoring start failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Production monitoring start failed: {str(e)}"
        )


@router.post(
    "/production/stop",
    summary="Stop production monitoring",
    description="Stop the production monitoring system (admin only)"
)
async def stop_production_monitoring(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Stop production monitoring."""
    try:
        # Check if user has admin privileges
        if not current_user.get("is_admin", False):
            log_security_event(
                "unauthorized_admin_access",
                {
                    "user_id": current_user.get("id"),
                    "action": "stop_production_monitoring",
                    "ip_address": "unknown"
                },
                "WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        
        await production_monitor.stop_monitoring()
        
        logger.info(
            "Production monitoring stopped",
            user_id=current_user.get("id"),
            admin_action=True
        )
        
        log_security_event(
            "admin_action",
            {
                "user_id": current_user.get("id"),
                "action": "stop_production_monitoring"
            },
            "INFO"
        )
        
        return {
            "status": "success",
            "message": "Production monitoring stopped successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Production monitoring stop failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Production monitoring stop failed: {str(e)}"
        )