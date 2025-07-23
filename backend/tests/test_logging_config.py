"""
Tests for logging configuration and structured logging functionality.
"""
import pytest
import logging
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from contextvars import copy_context

from app.core.logging_config import (
    setup_logging, get_correlation_id, set_correlation_id, generate_correlation_id,
    set_user_context, clear_context, log_performance_metric, performance_monitor,
    log_security_event, get_performance_metrics, reset_performance_metrics,
    get_logger, get_performance_logger, get_security_logger, get_component_logger,
    CorrelationIDProcessor, PerformanceProcessor, ComponentMetricsProcessor
)


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for log files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset logging state before each test."""
    reset_performance_metrics()
    clear_context()
    yield
    reset_performance_metrics()
    clear_context()


class TestLoggingSetup:
    """Test cases for logging setup and configuration."""
    
    def test_setup_logging_default(self, temp_log_dir):
        """Test default logging setup."""
        with patch('app.core.logging_config.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()
            
            logger = setup_logging()
            
            assert logger is not None
            assert hasattr(logger, 'info')
            assert hasattr(logger, 'error')
            assert hasattr(logger, 'warning')
    
    def test_setup_logging_custom_level(self, temp_log_dir):
        """Test logging setup with custom level."""
        with patch('app.core.logging_config.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()
            
            logger = setup_logging(log_level="DEBUG", enable_json=False)
            
            assert logger is not None
            # Check that logging level is set correctly
            root_logger = logging.getLogger()
            assert root_logger.level <= logging.DEBUG
    
    def test_setup_logging_file_disabled(self, temp_log_dir):
        """Test logging setup with file logging disabled."""
        with patch('app.core.logging_config.Path') as mock_path:
            mock_path.return_value.mkdir = Mock()
            
            logger = setup_logging(enable_file_logging=False)
            
            assert logger is not None
            # Should not create file handlers when disabled
            # This is hard to test directly, but we can verify the function completes
    
    def test_setup_file_logging(self, temp_log_dir):
        """Test file logging setup."""
        with patch('app.core.logging_config.logging.handlers.RotatingFileHandler') as mock_handler:
            mock_handler_instance = Mock()
            mock_handler.return_value = mock_handler_instance
            
            from app.core.logging_config import setup_file_logging
            setup_file_logging(logging.INFO)
            
            # Should create multiple file handlers
            assert mock_handler.call_count >= 4  # main, error, performance, security
    
    def test_setup_third_party_logging(self):
        """Test third-party library logging configuration."""
        from app.core.logging_config import setup_third_party_logging
        
        # Clear any existing handlers
        for logger_name in ['uvicorn', 'fastapi', 'trio', 'sqlalchemy.engine']:
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()
        
        setup_third_party_logging()
        
        # Check that specific loggers have appropriate levels
        assert logging.getLogger('uvicorn').level == logging.INFO
        assert logging.getLogger('trio').level == logging.WARNING
        assert logging.getLogger('sqlalchemy.engine').level == logging.WARNING


class TestCorrelationID:
    """Test cases for correlation ID functionality."""
    
    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = generate_correlation_id()
        
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0
        assert '-' in corr_id  # UUID format
        
        # Should generate unique IDs
        corr_id2 = generate_correlation_id()
        assert corr_id != corr_id2
    
    def test_set_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-id"
        
        # Initially should be empty
        assert get_correlation_id() == ""
        
        # Set and retrieve
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
        
        # Clear and check
        clear_context()
        assert get_correlation_id() == ""
    
    def test_set_user_context(self):
        """Test setting user context."""
        user_id = "test-user-123"
        request_id = "test-request-456"
        
        set_user_context(user_id, request_id)
        
        # Context variables are not directly accessible, but we can test
        # that the function completes without error
        assert True  # If we get here, the function worked
    
    def test_clear_context(self):
        """Test clearing all context."""
        set_correlation_id("test-id")
        set_user_context("test-user", "test-request")
        
        clear_context()
        
        # Should be cleared
        assert get_correlation_id() == ""


class TestPerformanceMetrics:
    """Test cases for performance metrics functionality."""
    
    def test_log_performance_metric(self):
        """Test logging performance metrics."""
        component = "test_component"
        operation = "test_operation"
        duration = 0.5
        
        initial_metrics = get_performance_metrics()
        initial_count = initial_metrics['request_count']
        
        log_performance_metric(component, operation, duration, success=True)
        
        updated_metrics = get_performance_metrics()
        
        assert updated_metrics['request_count'] == initial_count + 1
        assert updated_metrics['avg_response_time'] > 0
        assert component in updated_metrics['component_metrics']
        
        comp_metrics = updated_metrics['component_metrics'][component]
        assert comp_metrics['total_requests'] == 1
        assert comp_metrics['avg_duration'] == duration
        assert operation in comp_metrics['operations']
    
    def test_log_performance_metric_failure(self):
        """Test logging performance metrics for failures."""
        component = "test_component"
        operation = "test_operation"
        duration = 1.0
        
        initial_metrics = get_performance_metrics()
        initial_error_count = initial_metrics['error_count']
        
        log_performance_metric(component, operation, duration, success=False)
        
        updated_metrics = get_performance_metrics()
        
        assert updated_metrics['error_count'] == initial_error_count + 1
        
        comp_metrics = updated_metrics['component_metrics'][component]
        assert comp_metrics['error_count'] == 1
        assert comp_metrics['operations'][operation]['error_count'] == 1
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        metrics = get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'request_count' in metrics
        assert 'avg_response_time' in metrics
        assert 'error_count' in metrics
        assert 'component_metrics' in metrics
    
    def test_reset_performance_metrics(self):
        """Test resetting performance metrics."""
        # Add some metrics
        log_performance_metric("test", "op", 0.1)
        
        metrics_before = get_performance_metrics()
        assert metrics_before['request_count'] > 0
        
        reset_performance_metrics()
        
        metrics_after = get_performance_metrics()
        assert metrics_after['request_count'] == 0
        assert metrics_after['avg_response_time'] == 0.0
        assert metrics_after['error_count'] == 0
        assert metrics_after['component_metrics'] == {}
    
    @pytest.mark.asyncio
    async def test_performance_monitor_decorator_async(self):
        """Test performance monitor decorator for async functions."""
        @performance_monitor("test_component", "async_operation")
        async def test_async_function():
            await asyncio.sleep(0.01)  # Small delay
            return "success"
        
        initial_count = get_performance_metrics()['request_count']
        
        result = await test_async_function()
        
        assert result == "success"
        
        updated_metrics = get_performance_metrics()
        assert updated_metrics['request_count'] == initial_count + 1
        assert "test_component" in updated_metrics['component_metrics']
    
    def test_performance_monitor_decorator_sync(self):
        """Test performance monitor decorator for sync functions."""
        @performance_monitor("test_component", "sync_operation")
        def test_sync_function():
            time.sleep(0.01)  # Small delay
            return "success"
        
        initial_count = get_performance_metrics()['request_count']
        
        result = test_sync_function()
        
        assert result == "success"
        
        updated_metrics = get_performance_metrics()
        assert updated_metrics['request_count'] == initial_count + 1
        assert "test_component" in updated_metrics['component_metrics']
    
    @pytest.mark.asyncio
    async def test_performance_monitor_decorator_exception(self):
        """Test performance monitor decorator with exceptions."""
        @performance_monitor("test_component", "failing_operation")
        async def failing_function():
            raise ValueError("Test error")
        
        initial_error_count = get_performance_metrics()['error_count']
        
        with pytest.raises(ValueError):
            await failing_function()
        
        updated_metrics = get_performance_metrics()
        assert updated_metrics['error_count'] == initial_error_count + 1


class TestSecurityLogging:
    """Test cases for security logging functionality."""
    
    def test_log_security_event(self):
        """Test logging security events."""
        event_type = "unauthorized_access"
        details = {"user_id": "test-user", "ip": "192.168.1.1"}
        severity = "WARNING"
        
        # Mock the security logger
        with patch('app.core.logging_config.structlog.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_security_event(event_type, details, severity)
            
            # Should have called the logger
            mock_get_logger.assert_called_with("ragflow.security")
            mock_logger.warning.assert_called_once()
    
    def test_log_security_event_default_severity(self):
        """Test logging security events with default severity."""
        event_type = "test_event"
        details = {"key": "value"}
        
        with patch('app.core.logging_config.structlog.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            log_security_event(event_type, details)  # No severity specified
            
            mock_logger.warning.assert_called_once()  # Should default to warning


class TestLoggerInstances:
    """Test cases for logger instance creation."""
    
    def test_get_logger(self):
        """Test getting default logger."""
        logger = get_logger()
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
    
    def test_get_logger_with_name(self):
        """Test getting logger with custom name."""
        logger = get_logger("custom_name")
        
        assert logger is not None
        # The logger should be created successfully
    
    def test_get_performance_logger(self):
        """Test getting performance logger."""
        logger = get_performance_logger()
        
        assert logger is not None
        assert hasattr(logger, 'info')
    
    def test_get_security_logger(self):
        """Test getting security logger."""
        logger = get_security_logger()
        
        assert logger is not None
        assert hasattr(logger, 'warning')
    
    def test_get_component_logger(self):
        """Test getting component-specific logger."""
        component = "test_component"
        logger = get_component_logger(component)
        
        assert logger is not None
        # Should create logger with component-specific name


class TestProcessors:
    """Test cases for custom log processors."""
    
    def test_correlation_id_processor(self):
        """Test CorrelationIDProcessor."""
        processor = CorrelationIDProcessor()
        
        # Test without correlation ID
        event_dict = {"message": "test"}
        result = processor(None, None, event_dict)
        
        # Should not add correlation_id if not set
        assert "correlation_id" not in result
        
        # Test with correlation ID
        set_correlation_id("test-correlation-id")
        result = processor(None, None, event_dict)
        
        # Should add correlation_id
        assert result.get("correlation_id") == "test-correlation-id"
    
    def test_performance_processor(self):
        """Test PerformanceProcessor."""
        processor = PerformanceProcessor()
        
        # Add some performance metrics
        log_performance_metric("test", "op", 0.1)
        
        event_dict = {"message": "test"}
        result = processor(None, None, event_dict)
        
        assert "performance" in result
        assert "total_requests" in result["performance"]
        assert "avg_response_time" in result["performance"]
        assert "error_rate" in result["performance"]
    
    def test_component_metrics_processor(self):
        """Test ComponentMetricsProcessor."""
        processor = ComponentMetricsProcessor()
        
        # Add component metrics
        log_performance_metric("test_component", "op", 0.1)
        
        # Test without component in event
        event_dict = {"message": "test"}
        result = processor(None, None, event_dict)
        
        # Should not add component_metrics
        assert "component_metrics" not in result
        
        # Test with component in event
        event_dict = {"message": "test", "component": "test_component"}
        result = processor(None, None, event_dict)
        
        # Should add component_metrics
        assert "component_metrics" in result


class TestContextVariables:
    """Test cases for context variable behavior."""
    
    def test_context_isolation(self):
        """Test that context variables are properly isolated."""
        # Set context in current context
        set_correlation_id("context-1")
        assert get_correlation_id() == "context-1"
        
        # Create a copy of the context
        ctx = copy_context()
        
        def test_in_context():
            # Should see the same correlation ID
            assert get_correlation_id() == "context-1"
            
            # Change it in this context
            set_correlation_id("context-2")
            assert get_correlation_id() == "context-2"
        
        # Run in the copied context
        ctx.run(test_in_context)
        
        # Original context should be unchanged
        assert get_correlation_id() == "context-1"


class TestErrorHandling:
    """Test cases for error handling in logging functions."""
    
    def test_log_performance_metric_error_handling(self):
        """Test error handling in log_performance_metric."""
        # This should not raise an exception even with invalid input
        try:
            log_performance_metric("", "", -1, success=True)
            # Should complete without error
            assert True
        except Exception as e:
            pytest.fail(f"log_performance_metric raised an exception: {e}")
    
    def test_performance_monitor_with_invalid_component(self):
        """Test performance monitor with invalid component name."""
        @performance_monitor("", "")  # Empty strings
        def test_function():
            return "success"
        
        # Should not raise an exception
        try:
            result = test_function()
            assert result == "success"
        except Exception as e:
            pytest.fail(f"Performance monitor raised an exception: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])