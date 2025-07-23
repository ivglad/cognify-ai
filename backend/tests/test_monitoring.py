"""
Tests for system monitoring and health check functionality.
"""
import pytest
import trio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from app.core.monitoring import (
    HealthChecker, HealthStatus, ComponentType, HealthCheckResult, SystemMetrics,
    get_system_health, run_health_check, get_health_summary
)


@pytest.fixture
def health_checker():
    """Create health checker instance for testing."""
    checker = HealthChecker()
    return checker


@pytest.fixture
def mock_psutil():
    """Mock psutil for system metrics."""
    with patch('app.core.monitoring.psutil') as mock:
        # CPU metrics
        mock.cpu_percent.return_value = 25.5
        
        # Memory metrics
        mock_memory = Mock()
        mock_memory.percent = 45.2
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_memory.used = 4 * 1024**3  # 4GB
        mock.virtual_memory.return_value = mock_memory
        
        # Disk metrics
        mock_disk = Mock()
        mock_disk.percent = 60.0
        mock_disk.total = 100 * 1024**3  # 100GB
        mock_disk.used = 60 * 1024**3  # 60GB
        mock_disk.free = 40 * 1024**3  # 40GB
        mock.disk_usage.return_value = mock_disk
        
        # Network I/O
        mock_network = Mock()
        mock_network._asdict.return_value = {
            'bytes_sent': 1000000,
            'bytes_recv': 2000000,
            'packets_sent': 1000,
            'packets_recv': 2000
        }
        mock.net_io_counters.return_value = mock_network
        
        # Disk I/O
        mock_disk_io = Mock()
        mock_disk_io._asdict.return_value = {
            'read_bytes': 500000,
            'write_bytes': 300000,
            'read_count': 100,
            'write_count': 50
        }
        mock.disk_io_counters.return_value = mock_disk_io
        
        # Process count
        mock.pids.return_value = list(range(150))  # 150 processes
        
        # Load average
        mock.getloadavg.return_value = (0.5, 0.7, 0.9)
        
        # Boot time
        mock.boot_time.return_value = time.time() - 3600  # 1 hour ago
        
        yield mock


class TestHealthChecker:
    """Test cases for health checker."""
    
    def test_initialization(self, health_checker):
        """Test health checker initialization."""
        assert health_checker.health_checks is not None
        assert len(health_checker.health_checks) > 0
        assert health_checker.last_check_results == {}
        assert health_checker.system_metrics_history == []
        assert health_checker.running == False
    
    def test_register_health_check(self, health_checker):
        """Test registering custom health checks."""
        async def custom_check():
            return HealthCheckResult(
                component="custom",
                component_type=ComponentType.COMPUTE,
                status=HealthStatus.HEALTHY,
                response_time=0.1,
                details={"test": "ok"}
            )
        
        initial_count = len(health_checker.health_checks)
        health_checker.register_health_check("custom", custom_check)
        
        assert len(health_checker.health_checks) == initial_count + 1
        assert "custom" in health_checker.health_checks
    
    @pytest.mark.trio
    async def test_run_health_check_success(self, health_checker):
        """Test successful health check execution."""
        # Mock a successful health check
        async def mock_check():
            return HealthCheckResult(
                component="test_component",
                component_type=ComponentType.COMPUTE,
                status=HealthStatus.HEALTHY,
                response_time=0.0,
                details={"status": "ok"}
            )
        
        health_checker.register_health_check("test_component", mock_check)
        
        result = await health_checker.run_health_check("test_component")
        
        assert result.component == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time > 0  # Should be set by the checker
        assert "test_component" in health_checker.last_check_results
    
    @pytest.mark.trio
    async def test_run_health_check_failure(self, health_checker):
        """Test health check failure handling."""
        # Mock a failing health check
        async def failing_check():
            raise Exception("Test failure")
        
        health_checker.register_health_check("failing_component", failing_check)
        
        result = await health_checker.run_health_check("failing_component")
        
        assert result.component == "failing_component"
        assert result.status == HealthStatus.CRITICAL
        assert result.error_message == "Test failure"
        assert result.response_time > 0
    
    @pytest.mark.trio
    async def test_run_health_check_nonexistent(self, health_checker):
        """Test health check for non-existent component."""
        result = await health_checker.run_health_check("nonexistent")
        
        assert result.component == "nonexistent"
        assert result.status == HealthStatus.CRITICAL
        assert "not found" in result.error_message
    
    @pytest.mark.trio
    async def test_run_all_health_checks(self, health_checker):
        """Test running all health checks."""
        # Add a custom check
        async def custom_check():
            return HealthCheckResult(
                component="custom",
                component_type=ComponentType.COMPUTE,
                status=HealthStatus.HEALTHY,
                response_time=0.0,
                details={}
            )
        
        health_checker.register_health_check("custom", custom_check)
        
        # Mock the default checks to avoid external dependencies
        with patch.object(health_checker, '_check_database', new_callable=AsyncMock) as mock_db:
            with patch.object(health_checker, '_check_cache', new_callable=AsyncMock) as mock_cache:
                with patch.object(health_checker, '_check_elasticsearch', new_callable=AsyncMock) as mock_es:
                    # Setup mock returns
                    mock_db.return_value = HealthCheckResult(
                        component="database", component_type=ComponentType.DATABASE,
                        status=HealthStatus.HEALTHY, response_time=0.1, details={}
                    )
                    mock_cache.return_value = HealthCheckResult(
                        component="cache", component_type=ComponentType.CACHE,
                        status=HealthStatus.HEALTHY, response_time=0.1, details={}
                    )
                    mock_es.return_value = HealthCheckResult(
                        component="elasticsearch", component_type=ComponentType.SEARCH,
                        status=HealthStatus.HEALTHY, response_time=0.1, details={}
                    )
                    
                    results = await health_checker.run_all_health_checks()
                    
                    assert len(results) > 0
                    assert "custom" in results
                    assert all(isinstance(result, HealthCheckResult) for result in results.values())
    
    @pytest.mark.trio
    async def test_get_system_health(self, health_checker, mock_psutil):
        """Test getting comprehensive system health."""
        # Mock health checks
        with patch.object(health_checker, 'run_all_health_checks', new_callable=AsyncMock) as mock_checks:
            mock_checks.return_value = {
                "database": HealthCheckResult(
                    component="database", component_type=ComponentType.DATABASE,
                    status=HealthStatus.HEALTHY, response_time=0.1, details={}
                ),
                "cache": HealthCheckResult(
                    component="cache", component_type=ComponentType.CACHE,
                    status=HealthStatus.DEGRADED, response_time=0.5, details={}
                )
            }
            
            health_status = await health_checker.get_system_health()
            
            assert "overall_status" in health_status
            assert "timestamp" in health_status
            assert "uptime_seconds" in health_status
            assert "health_checks" in health_status
            assert "system_metrics" in health_status
            assert "performance_summary" in health_status
            
            # Should be degraded due to cache status
            assert health_status["overall_status"] == "degraded"
    
    def test_determine_overall_status(self, health_checker):
        """Test overall status determination logic."""
        # Test healthy status
        healthy_results = {
            "comp1": HealthCheckResult("comp1", ComponentType.COMPUTE, HealthStatus.HEALTHY, 0.1, {}),
            "comp2": HealthCheckResult("comp2", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, {})
        }
        assert health_checker._determine_overall_status(healthy_results) == HealthStatus.HEALTHY
        
        # Test degraded status
        degraded_results = {
            "comp1": HealthCheckResult("comp1", ComponentType.COMPUTE, HealthStatus.HEALTHY, 0.1, {}),
            "comp2": HealthCheckResult("comp2", ComponentType.DATABASE, HealthStatus.DEGRADED, 0.1, {})
        }
        assert health_checker._determine_overall_status(degraded_results) == HealthStatus.DEGRADED
        
        # Test unhealthy status
        unhealthy_results = {
            "comp1": HealthCheckResult("comp1", ComponentType.COMPUTE, HealthStatus.UNHEALTHY, 0.1, {}),
            "comp2": HealthCheckResult("comp2", ComponentType.DATABASE, HealthStatus.DEGRADED, 0.1, {})
        }
        assert health_checker._determine_overall_status(unhealthy_results) == HealthStatus.UNHEALTHY
        
        # Test critical status
        critical_results = {
            "comp1": HealthCheckResult("comp1", ComponentType.COMPUTE, HealthStatus.CRITICAL, 0.1, {}),
            "comp2": HealthCheckResult("comp2", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, {})
        }
        assert health_checker._determine_overall_status(critical_results) == HealthStatus.CRITICAL
        
        # Test empty results
        assert health_checker._determine_overall_status({}) == HealthStatus.CRITICAL
    
    def test_collect_system_metrics(self, health_checker, mock_psutil):
        """Test system metrics collection."""
        metrics = health_checker._collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 45.2
        assert metrics.disk_percent == 60.0
        assert metrics.process_count == 150
        assert metrics.load_average == [0.5, 0.7, 0.9]
        assert metrics.timestamp > 0
        
        # Check that metrics are stored in history
        assert len(health_checker.system_metrics_history) == 1
        assert health_checker.system_metrics_history[0] == metrics
    
    def test_get_system_uptime(self, health_checker, mock_psutil):
        """Test system uptime calculation."""
        uptime = health_checker._get_system_uptime()
        
        assert uptime > 0
        assert uptime <= 3700  # Should be around 1 hour (3600s) plus some margin
    
    def test_get_performance_summary(self, health_checker, mock_psutil):
        """Test performance summary generation."""
        # Add some metrics to history
        for _ in range(5):
            health_checker._collect_system_metrics()
        
        summary = health_checker._get_performance_summary()
        
        assert "avg_cpu_percent" in summary
        assert "avg_memory_percent" in summary
        assert "avg_disk_percent" in summary
        assert "avg_process_count" in summary
        assert "measurements_count" in summary
        
        assert summary["avg_cpu_percent"] == 25.5
        assert summary["avg_memory_percent"] == 45.2
        assert summary["measurements_count"] == 5
    
    @pytest.mark.trio
    async def test_database_health_check(self, health_checker):
        """Test database health check."""
        # Mock database session
        with patch('app.core.monitoring.get_db_session') as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock()
            mock_session.return_value.__aexit__ = AsyncMock()
            
            mock_session_instance = Mock()
            mock_result = Mock()
            mock_result.fetchone = AsyncMock(return_value=(1,))
            mock_session_instance.execute = AsyncMock(return_value=mock_result)
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            
            result = await health_checker._check_database()
            
            assert result.component == "database"
            assert result.component_type == ComponentType.DATABASE
            assert result.status == HealthStatus.HEALTHY
            assert result.response_time > 0
    
    @pytest.mark.trio
    async def test_cache_health_check(self, health_checker):
        """Test cache health check."""
        # Mock cache manager
        with patch('app.core.monitoring.cache_manager') as mock_cache:
            mock_cache.set = AsyncMock()
            mock_cache.get = AsyncMock(return_value="test_value")
            mock_cache.delete = AsyncMock()
            
            result = await health_checker._check_cache()
            
            assert result.component == "cache"
            assert result.component_type == ComponentType.CACHE
            assert result.status == HealthStatus.HEALTHY
            assert result.response_time > 0
    
    @pytest.mark.trio
    async def test_elasticsearch_health_check(self, health_checker):
        """Test Elasticsearch health check."""
        # Mock Elasticsearch client
        with patch('app.core.monitoring.elasticsearch_client') as mock_es:
            mock_es.cluster.health = AsyncMock(return_value={
                'status': 'green',
                'number_of_nodes': 3,
                'active_primary_shards': 5,
                'active_shards': 10
            })
            
            result = await health_checker._check_elasticsearch()
            
            assert result.component == "elasticsearch"
            assert result.component_type == ComponentType.SEARCH
            assert result.status == HealthStatus.HEALTHY
            assert result.details['cluster_status'] == 'green'
            assert result.details['number_of_nodes'] == 3
    
    @pytest.mark.trio
    async def test_system_resources_health_check(self, health_checker, mock_psutil):
        """Test system resources health check."""
        result = await health_checker._check_system_resources()
        
        assert result.component == "system_resources"
        assert result.component_type == ComponentType.COMPUTE
        assert result.status == HealthStatus.HEALTHY  # Based on mock values
        assert result.details['cpu_percent'] == 25.5
        assert result.details['memory_percent'] == 45.2
    
    @pytest.mark.trio
    async def test_disk_space_health_check(self, health_checker, mock_psutil):
        """Test disk space health check."""
        result = await health_checker._check_disk_space()
        
        assert result.component == "disk_space"
        assert result.component_type == ComponentType.DISK
        assert result.status == HealthStatus.HEALTHY  # 60% usage
        assert result.details['disk_percent'] == 60.0
        assert result.details['total_gb'] > 0
    
    @pytest.mark.trio
    async def test_memory_usage_health_check(self, health_checker, mock_psutil):
        """Test memory usage health check."""
        result = await health_checker._check_memory_usage()
        
        assert result.component == "memory_usage"
        assert result.component_type == ComponentType.MEMORY
        assert result.status == HealthStatus.HEALTHY  # 45.2% usage
        assert result.details['memory_percent'] == 45.2
        assert result.details['total_gb'] > 0
    
    def test_get_health_summary(self, health_checker):
        """Test health summary generation."""
        # Add some mock results
        health_checker.last_check_results = {
            "database": HealthCheckResult(
                "database", ComponentType.DATABASE, HealthStatus.HEALTHY, 0.1, {}, timestamp=time.time()
            ),
            "cache": HealthCheckResult(
                "cache", ComponentType.CACHE, HealthStatus.DEGRADED, 0.5, {}, timestamp=time.time()
            )
        }
        
        summary = health_checker.get_health_summary()
        
        assert summary["overall_status"] == "degraded"
        assert len(summary["components"]) == 2
        assert summary["components"]["database"]["status"] == "healthy"
        assert summary["components"]["cache"]["status"] == "degraded"
        assert summary["last_update"] > 0
    
    def test_get_metrics_history(self, health_checker, mock_psutil):
        """Test metrics history retrieval."""
        # Add some metrics to history
        for _ in range(10):
            health_checker._collect_system_metrics()
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get history for 1 hour (should include all)
        history = health_checker.get_metrics_history(hours=1)
        
        assert len(history) == 10
        assert all("cpu_percent" in metric for metric in history)
        assert all("timestamp" in metric for metric in history)
        
        # Get history for very short time (should include fewer)
        history_short = health_checker.get_metrics_history(hours=0.001)  # Very short time
        assert len(history_short) <= len(history)


class TestSystemMetrics:
    """Test cases for SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation."""
        metrics = SystemMetrics(
            cpu_percent=50.0,
            memory_percent=60.0,
            disk_percent=70.0,
            network_io={'bytes_sent': 1000},
            disk_io={'read_bytes': 500},
            process_count=100,
            load_average=[1.0, 1.5, 2.0]
        )
        
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 70.0
        assert metrics.network_io == {'bytes_sent': 1000}
        assert metrics.disk_io == {'read_bytes': 500}
        assert metrics.process_count == 100
        assert metrics.load_average == [1.0, 1.5, 2.0]
        assert metrics.timestamp > 0  # Should be set automatically


class TestHealthCheckResult:
    """Test cases for HealthCheckResult dataclass."""
    
    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        result = HealthCheckResult(
            component="test",
            component_type=ComponentType.COMPUTE,
            status=HealthStatus.HEALTHY,
            response_time=0.1,
            details={"key": "value"},
            error_message="test error"
        )
        
        assert result.component == "test"
        assert result.component_type == ComponentType.COMPUTE
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time == 0.1
        assert result.details == {"key": "value"}
        assert result.error_message == "test error"
        assert result.timestamp > 0  # Should be set automatically


class TestGlobalFunctions:
    """Test cases for global monitoring functions."""
    
    @pytest.mark.trio
    async def test_get_system_health_function(self):
        """Test global get_system_health function."""
        with patch('app.core.monitoring.health_checker') as mock_checker:
            mock_checker.get_system_health = AsyncMock(return_value={"status": "healthy"})
            
            result = await get_system_health()
            
            assert result == {"status": "healthy"}
            mock_checker.get_system_health.assert_called_once()
    
    @pytest.mark.trio
    async def test_run_health_check_function(self):
        """Test global run_health_check function."""
        with patch('app.core.monitoring.health_checker') as mock_checker:
            mock_result = HealthCheckResult(
                "test", ComponentType.COMPUTE, HealthStatus.HEALTHY, 0.1, {}
            )
            mock_checker.run_health_check = AsyncMock(return_value=mock_result)
            
            result = await run_health_check("test")
            
            assert result == mock_result
            mock_checker.run_health_check.assert_called_once_with("test")
    
    def test_get_health_summary_function(self):
        """Test global get_health_summary function."""
        with patch('app.core.monitoring.health_checker') as mock_checker:
            mock_checker.get_health_summary.return_value = {"status": "healthy"}
            
            result = get_health_summary()
            
            assert result == {"status": "healthy"}
            mock_checker.get_health_summary.assert_called_once()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])