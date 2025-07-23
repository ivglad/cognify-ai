"""
Integration tests for API endpoints.
"""
import pytest
import trio
import json
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.api.dependencies.auth import get_current_user


# Mock authentication for tests
def mock_get_current_user():
    """Mock user authentication for testing."""
    return {
        "id": "test-user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_admin": True,
        "is_active": True
    }


# Override the dependency
app.dependency_overrides[get_current_user] = mock_get_current_user


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client for API testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestHealthAPI:
    """Integration tests for health check API."""
    
    def test_health_status_endpoint(self, test_client):
        """Test health status endpoint."""
        with patch('app.api.v1.monitoring.get_system_health') as mock_health:
            mock_health.return_value = {
                "overall_status": "healthy",
                "timestamp": 1234567890,
                "uptime_seconds": 3600,
                "health_checks": {
                    "database": {
                        "status": "healthy",
                        "response_time": 0.1,
                        "details": {"connection": "ok"}
                    }
                },
                "system_metrics": {
                    "cpu_percent": 25.0,
                    "memory_percent": 45.0
                },
                "performance_summary": {
                    "avg_cpu_percent": 25.0
                }
            }
            
            response = test_client.get("/api/v1/monitoring/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["overall_status"] == "healthy"
            assert "health_checks" in data
            assert "system_metrics" in data
    
    def test_health_summary_endpoint(self, test_client):
        """Test health summary endpoint."""
        with patch('app.api.v1.monitoring.get_health_summary') as mock_summary:
            mock_summary.return_value = {
                "overall_status": "healthy",
                "components": {
                    "database": {"status": "healthy", "last_check": 1234567890}
                },
                "last_update": 1234567890
            }
            
            response = test_client.get("/api/v1/monitoring/health/summary")
            
            assert response.status_code == 200
            data = response.json()
            assert data["overall_status"] == "healthy"
            assert "components" in data
    
    def test_component_health_endpoint(self, test_client):
        """Test specific component health check."""
        with patch('app.api.v1.monitoring.run_health_check') as mock_check:
            from app.core.monitoring import HealthCheckResult, HealthStatus, ComponentType
            
            mock_result = HealthCheckResult(
                component="database",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                response_time=0.1,
                details={"connection": "ok"},
                timestamp=1234567890
            )
            mock_check.return_value = mock_result
            
            response = test_client.get("/api/v1/monitoring/health/database")
            
            assert response.status_code == 200
            data = response.json()
            assert data["component"] == "database"
            assert data["status"] == "healthy"
    
    def test_simple_status_endpoint(self, test_client):
        """Test simple status endpoint."""
        with patch('app.api.v1.monitoring.get_health_summary') as mock_summary:
            mock_summary.return_value = {"overall_status": "healthy"}
            
            response = test_client.get("/api/v1/monitoring/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["health"] == "healthy"
    
    def test_simple_status_unhealthy(self, test_client):
        """Test simple status endpoint when system is unhealthy."""
        with patch('app.api.v1.monitoring.get_health_summary') as mock_summary:
            mock_summary.return_value = {"overall_status": "critical"}
            
            response = test_client.get("/api/v1/monitoring/status")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "error"
            assert data["health"] == "critical"


class TestMetricsAPI:
    """Integration tests for metrics API."""
    
    def test_system_metrics_endpoint(self, test_client):
        """Test system metrics endpoint."""
        with patch('app.api.v1.monitoring.health_checker') as mock_checker:
            from app.core.monitoring import SystemMetrics
            
            mock_metrics = SystemMetrics(
                cpu_percent=25.0,
                memory_percent=45.0,
                disk_percent=60.0,
                network_io={"bytes_sent": 1000},
                disk_io={"read_bytes": 500},
                process_count=100,
                load_average=[1.0, 1.5, 2.0],
                timestamp=1234567890
            )
            mock_checker.system_metrics_history = [mock_metrics]
            
            response = test_client.get("/api/v1/monitoring/metrics")
            
            assert response.status_code == 200
            data = response.json()
            assert data["cpu_percent"] == 25.0
            assert data["memory_percent"] == 45.0
    
    def test_metrics_history_endpoint(self, test_client):
        """Test metrics history endpoint."""
        with patch('app.api.v1.monitoring.get_metrics_history') as mock_history:
            mock_history.return_value = [
                {
                    "cpu_percent": 25.0,
                    "memory_percent": 45.0,
                    "timestamp": 1234567890
                }
            ]
            
            response = test_client.get("/api/v1/monitoring/metrics/history?hours=1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["hours"] == 1
            assert data["data_points"] == 1
            assert len(data["metrics"]) == 1
    
    def test_performance_metrics_endpoint(self, test_client):
        """Test performance metrics endpoint."""
        with patch('app.api.v1.monitoring.get_performance_metrics') as mock_perf:
            mock_perf.return_value = {
                "request_count": 100,
                "avg_response_time": 0.5,
                "error_count": 5,
                "component_metrics": {
                    "search": {
                        "total_requests": 50,
                        "avg_duration": 0.3
                    }
                }
            }
            
            response = test_client.get("/api/v1/monitoring/performance")
            
            assert response.status_code == 200
            data = response.json()
            assert data["request_count"] == 100
            assert data["avg_response_time"] == 0.5
    
    def test_reset_performance_metrics_admin(self, test_client):
        """Test resetting performance metrics (admin only)."""
        with patch('app.api.v1.monitoring.reset_performance_metrics') as mock_reset:
            response = test_client.post("/api/v1/monitoring/performance/reset")
            
            assert response.status_code == 200
            data = response.json()
            assert "reset successfully" in data["message"]
            mock_reset.assert_called_once()
    
    def test_reset_performance_metrics_non_admin(self, test_client):
        """Test resetting performance metrics without admin privileges."""
        # Override with non-admin user
        def mock_non_admin_user():
            return {
                "id": "test-user-123",
                "username": "testuser",
                "is_admin": False
            }
        
        app.dependency_overrides[get_current_user] = mock_non_admin_user
        
        try:
            response = test_client.post("/api/v1/monitoring/performance/reset")
            assert response.status_code == 403
        finally:
            # Restore admin user
            app.dependency_overrides[get_current_user] = mock_get_current_user


class TestAlertsAPI:
    """Integration tests for alerts API."""
    
    def test_system_alerts_endpoint(self, test_client):
        """Test system alerts endpoint."""
        with patch('app.api.v1.monitoring.get_system_health') as mock_health:
            mock_health.return_value = {
                "overall_status": "critical",
                "timestamp": 1234567890,
                "health_checks": {
                    "database": {
                        "status": "critical",
                        "error_message": "Connection failed",
                        "timestamp": 1234567890,
                        "details": {}
                    }
                },
                "system_metrics": {
                    "cpu_percent": 95.0,
                    "memory_percent": 92.0,
                    "disk_percent": 85.0,
                    "timestamp": 1234567890
                }
            }
            
            response = test_client.get("/api/v1/monitoring/alerts")
            
            assert response.status_code == 200
            data = response.json()
            assert data["alert_count"] > 0
            assert len(data["alerts"]) > 0
            
            # Should have system health alert
            system_alerts = [a for a in data["alerts"] if a["type"] == "system_health"]
            assert len(system_alerts) > 0
            
            # Should have resource usage alerts
            resource_alerts = [a for a in data["alerts"] if a["type"] == "resource_usage"]
            assert len(resource_alerts) > 0  # High CPU and memory usage


class TestLogsAPI:
    """Integration tests for logs API."""
    
    def test_recent_logs_admin(self, test_client):
        """Test recent logs endpoint (admin only)."""
        # Mock log file content
        mock_log_content = """2023-01-01 10:00:00 - ragflow - INFO - Test log entry 1
2023-01-01 10:01:00 - ragflow - ERROR - Test error entry
2023-01-01 10:02:00 - ragflow - INFO - Test log entry 2"""
        
        with patch('builtins.open', mock_open_multiple_files({"logs/ragflow.log": mock_log_content})):
            response = test_client.get("/api/v1/monitoring/logs/recent?lines=10")
            
            assert response.status_code == 200
            data = response.json()
            assert data["lines_requested"] == 10
            assert len(data["log_entries"]) > 0
    
    def test_recent_logs_non_admin(self, test_client):
        """Test recent logs endpoint without admin privileges."""
        # Override with non-admin user
        def mock_non_admin_user():
            return {
                "id": "test-user-123",
                "username": "testuser",
                "is_admin": False
            }
        
        app.dependency_overrides[get_current_user] = mock_non_admin_user
        
        try:
            response = test_client.get("/api/v1/monitoring/logs/recent")
            assert response.status_code == 403
        finally:
            # Restore admin user
            app.dependency_overrides[get_current_user] = mock_get_current_user


class TestSearchAPI:
    """Integration tests for search API."""
    
    @pytest.mark.trio
    async def test_search_endpoint(self, async_client):
        """Test search endpoint."""
        with patch('app.api.v1.search.search_service') as mock_search:
            mock_search.search.return_value = {
                "query": "test query",
                "results": [
                    {
                        "chunk_id": "chunk_1",
                        "content": "Test content",
                        "score": 0.9,
                        "metadata": {"document_title": "Test Doc"}
                    }
                ],
                "total": 1,
                "search_time": 0.1,
                "success": True
            }
            
            search_request = {
                "query": "test query",
                "search_type": "hybrid",
                "top_k": 10
            }
            
            response = await async_client.post(
                "/api/v1/search",
                json=search_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "test query"
            assert len(data["results"]) == 1
            assert data["success"] == True


class TestDocumentAPI:
    """Integration tests for document API."""
    
    def test_list_documents_endpoint(self, test_client):
        """Test document listing endpoint."""
        with patch('app.api.v1.documents.document_service') as mock_service:
            mock_service.list_documents.return_value = {
                "documents": [
                    {
                        "id": "doc_1",
                        "title": "Test Document",
                        "description": "Test description",
                        "created_at": "2023-01-01T00:00:00Z"
                    }
                ],
                "total": 1,
                "page": 1,
                "per_page": 10
            }
            
            response = test_client.get("/api/v1/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["documents"]) == 1
            assert data["total"] == 1
    
    def test_get_document_endpoint(self, test_client):
        """Test get single document endpoint."""
        with patch('app.api.v1.documents.document_service') as mock_service:
            mock_service.get_document.return_value = {
                "id": "doc_1",
                "title": "Test Document",
                "description": "Test description",
                "content": "Test content",
                "metadata": {"author": "Test Author"},
                "created_at": "2023-01-01T00:00:00Z"
            }
            
            response = test_client.get("/api/v1/documents/doc_1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "doc_1"
            assert data["title"] == "Test Document"
    
    def test_delete_document_endpoint(self, test_client):
        """Test document deletion endpoint."""
        with patch('app.api.v1.documents.document_service') as mock_service:
            mock_service.delete_document.return_value = {
                "success": True,
                "message": "Document deleted successfully"
            }
            
            response = test_client.delete("/api/v1/documents/doc_1")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True


class TestCitationAPI:
    """Integration tests for citation API."""
    
    @pytest.mark.trio
    async def test_generate_citations_endpoint(self, async_client):
        """Test citation generation endpoint."""
        with patch('app.services.response.citation_generator.citation_generator') as mock_generator:
            mock_generator.generate_citations.return_value = {
                "original_response": "Test response",
                "formatted_response": "Test response [1]",
                "citation_mappings": [
                    {
                        "sentence": "Test response",
                        "sentence_index": 0,
                        "citations": [
                            {
                                "chunk_id": "chunk_1",
                                "document_id": "doc_1",
                                "confidence_score": 0.9
                            }
                        ]
                    }
                ],
                "total_citations": 1,
                "success": True
            }
            
            citation_request = {
                "response_text": "Test response",
                "source_chunks": [
                    {
                        "chunk_id": "chunk_1",
                        "document_id": "doc_1",
                        "content": "Test content"
                    }
                ]
            }
            
            response = await async_client.post(
                "/api/v1/citations/generate",
                json=citation_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_citations"] == 1
            assert data["success"] == True
    
    @pytest.mark.trio
    async def test_validate_citations_endpoint(self, async_client):
        """Test citation validation endpoint."""
        with patch('app.services.response.citation_validator.citation_validator') as mock_validator:
            mock_validator.validate_citations.return_value = {
                "total_citations": 1,
                "valid_citations": 1,
                "invalid_citations": 0,
                "overall_metrics": {
                    "avg_accuracy": 0.9,
                    "avg_relevance": 0.8
                },
                "success": True
            }
            
            validation_request = {
                "citations": [
                    {
                        "id": "cite_1",
                        "chunk_id": "chunk_1",
                        "document_id": "doc_1",
                        "cited_text": "Test citation",
                        "source_text": "Test source",
                        "confidence": 0.9
                    }
                ],
                "response_text": "Test response",
                "source_chunks": [
                    {
                        "chunk_id": "chunk_1",
                        "content": "Test content"
                    }
                ]
            }
            
            response = await async_client.post(
                "/api/v1/citations/validate",
                json=validation_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["valid_citations"] == 1
            assert data["success"] == True


# Helper functions
def mock_open_multiple_files(files_dict):
    """Mock open function for multiple files."""
    from unittest.mock import mock_open
    
    def open_func(filename, mode='r', **kwargs):
        if filename in files_dict:
            return mock_open(read_data=files_dict[filename]).return_value
        else:
            return mock_open(read_data="").return_value
    
    return open_func


class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_health_check_error_handling(self, test_client):
        """Test error handling in health check endpoints."""
        with patch('app.api.v1.monitoring.get_system_health') as mock_health:
            mock_health.side_effect = Exception("Test error")
            
            response = test_client.get("/api/v1/monitoring/health")
            
            assert response.status_code == 500
            data = response.json()
            assert "Health check failed" in data["detail"]
    
    def test_search_error_handling(self, test_client):
        """Test error handling in search endpoints."""
        with patch('app.api.v1.search.search_service') as mock_search:
            mock_search.search.side_effect = Exception("Search error")
            
            search_request = {
                "query": "test query",
                "search_type": "hybrid"
            }
            
            response = test_client.post("/api/v1/search", json=search_request)
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data["detail"].lower()


class TestAuthentication:
    """Integration tests for authentication."""
    
    def test_protected_endpoint_without_auth(self, test_client):
        """Test accessing protected endpoint without authentication."""
        # Remove the auth override temporarily
        if get_current_user in app.dependency_overrides:
            del app.dependency_overrides[get_current_user]
        
        try:
            response = test_client.get("/api/v1/monitoring/health")
            # Should return 401 or redirect to login
            assert response.status_code in [401, 403, 422]  # Depending on auth implementation
        finally:
            # Restore the auth override
            app.dependency_overrides[get_current_user] = mock_get_current_user
    
    def test_admin_endpoint_with_regular_user(self, test_client):
        """Test accessing admin endpoint with regular user."""
        def mock_regular_user():
            return {
                "id": "test-user-123",
                "username": "testuser",
                "is_admin": False
            }
        
        app.dependency_overrides[get_current_user] = mock_regular_user
        
        try:
            response = test_client.post("/api/v1/monitoring/performance/reset")
            assert response.status_code == 403
        finally:
            # Restore admin user
            app.dependency_overrides[get_current_user] = mock_get_current_user


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])