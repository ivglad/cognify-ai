"""
Comprehensive logging configuration with structured logging, correlation IDs, and performance tracking.
"""
import logging
import logging.handlers
import sys
import os
import uuid
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from contextvars import ContextVar
from functools import wraps

import structlog
from structlog.types import FilteringBoundLogger

from app.core.config import settings

# Context variables for correlation tracking
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

# Performance tracking
performance_metrics: Dict[str, Any] = {
    'request_count': 0,
    'avg_response_time': 0.0,
    'error_count': 0,
    'component_metrics': {}
}


class CorrelationIDProcessor:
    """Add correlation ID to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        correlation_id = correlation_id_var.get('')
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        
        user_id = user_id_var.get('')
        if user_id:
            event_dict['user_id'] = user_id
        
        request_id = request_id_var.get('')
        if request_id:
            event_dict['request_id'] = request_id
        
        return event_dict


class PerformanceProcessor:
    """Add performance metrics to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        # Add current performance metrics
        event_dict['performance'] = {
            'total_requests': performance_metrics['request_count'],
            'avg_response_time': performance_metrics['avg_response_time'],
            'error_rate': performance_metrics['error_count'] / max(performance_metrics['request_count'], 1)
        }
        
        return event_dict


class ComponentMetricsProcessor:
    """Add component-specific metrics to log records."""
    
    def __call__(self, logger, method_name, event_dict):
        component = event_dict.get('component', '')
        if component and component in performance_metrics['component_metrics']:
            event_dict['component_metrics'] = performance_metrics['component_metrics'][component]
        
        return event_dict


def setup_logging(log_level: str = "INFO", enable_json: bool = True, enable_file_logging: bool = True):
    """
    Setup comprehensive structured logging with correlation IDs and performance tracking.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Whether to use JSON formatting
        enable_file_logging: Whether to enable file logging
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        CorrelationIDProcessor(),
        PerformanceProcessor(),
        ComponentMetricsProcessor(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add appropriate renderer
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_obj,
    )
    
    # Set up file logging if enabled
    if enable_file_logging:
        setup_file_logging(log_level_obj)
    
    # Set specific log levels for third-party libraries
    setup_third_party_logging()
    
    # Create structured logger for application use
    logger = structlog.get_logger("ragflow")
    logger.info(
        "Comprehensive logging configuration completed",
        backend="trio",
        log_level=log_level,
        json_enabled=enable_json,
        file_logging_enabled=enable_file_logging
    )
    
    return logger


def setup_file_logging(log_level: int):
    """Setup file-based logging with rotation."""
    
    # Main application log
    main_handler = logging.handlers.RotatingFileHandler(
        "logs/ragflow.log",
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    main_handler.setLevel(log_level)
    main_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    
    # Error log (errors and above only)
    error_handler = logging.handlers.RotatingFileHandler(
        "logs/ragflow_errors.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d"
        )
    )
    
    # Performance log
    performance_handler = logging.handlers.RotatingFileHandler(
        "logs/ragflow_performance.log",
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=5
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - PERFORMANCE - %(message)s"
        )
    )
    
    # Security log
    security_handler = logging.handlers.RotatingFileHandler(
        "logs/ragflow_security.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    security_handler.setLevel(logging.WARNING)
    security_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - SECURITY - %(levelname)s - %(message)s"
        )
    )
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(main_handler)
    root_logger.addHandler(error_handler)
    
    # Add handlers to specific loggers
    performance_logger = logging.getLogger("ragflow.performance")
    performance_logger.addHandler(performance_handler)
    
    security_logger = logging.getLogger("ragflow.security")
    security_logger.addHandler(security_handler)


def setup_third_party_logging():
    """Configure logging levels for third-party libraries."""
    
    # Web framework logging
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Async framework logging
    logging.getLogger("trio").setLevel(logging.WARNING)
    logging.getLogger("trio._core").setLevel(logging.ERROR)
    
    # Database logging
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.INFO)
    
    # HTTP client logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # ML/AI library logging
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    # Other libraries
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id_var.get('')


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context."""
    correlation_id_var.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def set_user_context(user_id: str, request_id: Optional[str] = None):
    """Set user context for logging."""
    user_id_var.set(user_id)
    if request_id:
        request_id_var.set(request_id)


def clear_context():
    """Clear all context variables."""
    correlation_id_var.set('')
    user_id_var.set('')
    request_id_var.set('')


def log_performance_metric(component: str, operation: str, duration: float, success: bool = True):
    """Log performance metric for a component operation."""
    
    # Update global metrics
    performance_metrics['request_count'] += 1
    if not success:
        performance_metrics['error_count'] += 1
    
    # Update average response time
    current_avg = performance_metrics['avg_response_time']
    request_count = performance_metrics['request_count']
    new_avg = ((current_avg * (request_count - 1)) + duration) / request_count
    performance_metrics['avg_response_time'] = new_avg
    
    # Update component metrics
    if component not in performance_metrics['component_metrics']:
        performance_metrics['component_metrics'][component] = {
            'total_requests': 0,
            'avg_duration': 0.0,
            'error_count': 0,
            'operations': {}
        }
    
    comp_metrics = performance_metrics['component_metrics'][component]
    comp_metrics['total_requests'] += 1
    if not success:
        comp_metrics['error_count'] += 1
    
    # Update component average duration
    comp_avg = comp_metrics['avg_duration']
    comp_requests = comp_metrics['total_requests']
    comp_metrics['avg_duration'] = ((comp_avg * (comp_requests - 1)) + duration) / comp_requests
    
    # Update operation-specific metrics
    if operation not in comp_metrics['operations']:
        comp_metrics['operations'][operation] = {
            'count': 0,
            'avg_duration': 0.0,
            'error_count': 0
        }
    
    op_metrics = comp_metrics['operations'][operation]
    op_metrics['count'] += 1
    if not success:
        op_metrics['error_count'] += 1
    
    op_avg = op_metrics['avg_duration']
    op_count = op_metrics['count']
    op_metrics['avg_duration'] = ((op_avg * (op_count - 1)) + duration) / op_count
    
    # Log the metric
    logger = structlog.get_logger("ragflow.performance")
    logger.info(
        "Performance metric recorded",
        component=component,
        operation=operation,
        duration=duration,
        success=success,
        correlation_id=get_correlation_id()
    )


def performance_monitor(component: str, operation: str = None):
    """Decorator to monitor function performance."""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger = structlog.get_logger("ragflow.performance")
                logger.error(
                    "Performance monitored function failed",
                    component=component,
                    operation=op_name,
                    error=str(e),
                    correlation_id=get_correlation_id()
                )
                raise
            finally:
                duration = time.time() - start_time
                log_performance_metric(component, op_name, duration, success)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start_time = time.time()
            success = True
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                logger = structlog.get_logger("ragflow.performance")
                logger.error(
                    "Performance monitored function failed",
                    component=component,
                    operation=op_name,
                    error=str(e),
                    correlation_id=get_correlation_id()
                )
                raise
            finally:
                duration = time.time() - start_time
                log_performance_metric(component, op_name, duration, success)
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_security_event(event_type: str, details: Dict[str, Any], severity: str = "WARNING"):
    """Log security-related events."""
    
    logger = structlog.get_logger("ragflow.security")
    log_method = getattr(logger, severity.lower(), logger.warning)
    
    log_method(
        "Security event detected",
        event_type=event_type,
        details=details,
        correlation_id=get_correlation_id(),
        user_id=user_id_var.get(''),
        timestamp=time.time()
    )


def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics."""
    return performance_metrics.copy()


def reset_performance_metrics():
    """Reset performance metrics (useful for testing)."""
    global performance_metrics
    performance_metrics = {
        'request_count': 0,
        'avg_response_time': 0.0,
        'error_count': 0,
        'component_metrics': {}
    }


# Create logger instances for different purposes
def get_logger(name: str = "ragflow") -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def get_performance_logger() -> FilteringBoundLogger:
    """Get performance logger instance."""
    return structlog.get_logger("ragflow.performance")


def get_security_logger() -> FilteringBoundLogger:
    """Get security logger instance."""
    return structlog.get_logger("ragflow.security")


def get_component_logger(component: str) -> FilteringBoundLogger:
    """Get component-specific logger instance."""
    return structlog.get_logger(f"ragflow.{component}")


# Initialize logging on module import
if not hasattr(logging.getLogger(), '_ragflow_configured'):
    setup_logging()
    logging.getLogger()._ragflow_configured = True