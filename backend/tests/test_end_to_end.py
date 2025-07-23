"""
End-to-end tests for complete RAGFlow workflows.
"""
import pytest
import trio
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.api.dependencies.auth import get_current_user


# Mock authentication for E2E tests
def mock_get_current_user():
    """Mock user authentication for testing."""
    return {
        "id": "e2e-test-user",
        "username": "e2euser",
        "email": "e2e@example.com",
        "is_admin": True,
        "is_active": True
    }


# Override the dependency
app.dependency_overrides[get_current_user] = mock_get_current_user


@pytest.fixture
def test_client():
    """Create test client for E2E testing."""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create async test client for E2E testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_document_file():
    """Create a sample document file for testing."""
    content = """
    # Introduction to Machine Learning
    
    Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    Supervised learning algorithms build a mathematical model of a set of data that contains both
    the inputs and the desired outputs. The data is known as training data.
    
    ### Unsupervised Learning
    Unsupervised learning algorithms take a set of data that contains only inputs, and find
    structure in the data, like grouping or clustering of data points.
    
    ### Reinforcement Learning
    Reinforcement learning is an area of machine learning concerned with how software agents
    ought to take actions in an environment in order to maximize the notion of cumulative reward.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return Path(f.name)


class TestCompleteDocumentWorkflow:
    """End-to-end tests for complete document processing workflow."""
    
    @pytest.mark.trio
    async def test_document_upload_to_search_workflow(self, async_client, sample_document_file):
        """Test complete workflow from document upload to search."""
        
        # Mock all the services that would be involved
        with patch('app.services.document_processing.document_processor') as mock_processor, \
             patch('app.services.search.search_service') as mock_search, \
             patch('app.services.embeddings.embedding_service') as mock_embeddings, \
             patch('app.core.storage.minio_client') as mock_storage:
            
            # Setup mocks
            mock_processor.process_document.return_value = {
                "document_id": "test_doc_1",
                "status": "completed",
                "chunks_created": 5,
                "processing_time": 2.5
            }
            
            mock_search.search.return_value = {
                "query": "machine learning",
                "results": [
                    {
                        "chunk_id": "chunk_1",
                        "document_id": "test_doc_1",
                        "content": "Machine learning is a method of data analysis...",
                        "score": 0.95,
                        "metadata": {"document_title": "Introduction to Machine Learning"}
                    }
                ],
                "total": 1,
                "search_time": 0.1,
                "success": True
            }
            
            mock_storage.put_object.return_value = True
            mock_embeddings.generate_embeddings.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            # Step 1: Upload document
            with open(sample_document_file, 'rb') as f:
                files = {"file": ("test_document.txt", f, "text/plain")}
                upload_response = await async_client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data={"title": "Test ML Document", "description": "Test document for E2E testing"}
                )
            
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert upload_data["success"] == True
            document_id = upload_data["document_id"]
            
            # Step 2: Wait for processing (simulate)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Step 3: Check document status
            status_response = await async_client.get(f"/api/v1/documents/{document_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["processing_status"] == "completed"
            
            # Step 4: Search for content
            search_request = {
                "query": "machine learning",
                "search_type": "hybrid",
                "top_k": 10
            }
            
            search_response = await async_client.post(
                "/api/v1/search",
                json=search_request
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["success"] == True
            assert len(search_data["results"]) > 0
            assert search_data["results"][0]["document_id"] == document_id
    
    @pytest.mark.trio
    async def test_document_processing_with_chunking(self, async_client, sample_document_file):
        """Test document processing with different chunking strategies."""
        
        with patch('app.services.document_processing.document_processor') as mock_processor, \
             patch('app.services.chunking.chunking_manager') as mock_chunking:
            
            # Mock chunking with different strategies
            mock_chunking.chunk_document.return_value = {
                "chunks": [
                    {
                        "chunk_id": "chunk_1",
                        "content": "Introduction to Machine Learning section",
                        "chunk_index": 0,
                        "metadata": {"section": "introduction", "strategy": "hierarchical"}
                    },
                    {
                        "chunk_id": "chunk_2", 
                        "content": "Types of Machine Learning section",
                        "chunk_index": 1,
                        "metadata": {"section": "types", "strategy": "hierarchical"}
                    }
                ],
                "strategy_used": "hierarchical",
                "total_chunks": 2
            }
            
            mock_processor.process_document.return_value = {
                "document_id": "test_doc_2",
                "status": "completed",
                "chunks_created": 2,
                "chunking_strategy": "hierarchical"
            }
            
            # Upload document with specific chunking strategy
            with open(sample_document_file, 'rb') as f:
                files = {"file": ("ml_document.txt", f, "text/plain")}
                data = {
                    "title": "ML Document with Chunking",
                    "description": "Testing chunking strategies",
                    "chunking_strategy": "hierarchical"
                }
                
                response = await async_client.post(
                    "/api/v1/documents/upload",
                    files=files,
                    data=data
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            
            document_id = data["document_id"]
            
            # Check chunks were created
            chunks_response = await async_client.get(f"/api/v1/documents/{document_id}/chunks")
            assert chunks_response.status_code == 200
            chunks_data = chunks_response.json()
            assert len(chunks_data["chunks"]) == 2
            assert chunks_data["chunks"][0]["metadata"]["strategy"] == "hierarchical"
    
    @pytest.mark.trio
    async def test_search_with_citation_generation(self, async_client):
        """Test search workflow with citation generation."""
        
        with patch('app.services.search.search_service') as mock_search, \
             patch('app.services.response.answer_generator') as mock_generator, \
             patch('app.services.response.citation_generator') as mock_citations:
            
            # Mock search results
            mock_search.search.return_value = {
                "query": "what is supervised learning",
                "results": [
                    {
                        "chunk_id": "chunk_supervised",
                        "document_id": "ml_doc",
                        "content": "Supervised learning algorithms build a mathematical model using training data that contains both inputs and desired outputs.",
                        "score": 0.92,
                        "metadata": {"document_title": "ML Guide", "page_number": 2}
                    }
                ],
                "total": 1,
                "success": True
            }
            
            # Mock answer generation
            mock_generator.generate_answer.return_value = {
                "answer": "Supervised learning is a type of machine learning where algorithms learn from labeled training data to make predictions on new data.",
                "confidence": 0.9,
                "sources_used": ["chunk_supervised"],
                "generation_time": 1.2
            }
            
            # Mock citation generation
            mock_citations.generate_citations.return_value = {
                "original_response": "Supervised learning is a type of machine learning where algorithms learn from labeled training data.",
                "formatted_response": "Supervised learning is a type of machine learning where algorithms learn from labeled training data [1].",
                "citations": [
                    {
                        "id": "cite_1",
                        "chunk_id": "chunk_supervised",
                        "document_id": "ml_doc",
                        "document_title": "ML Guide",
                        "confidence_score": 0.95,
                        "cited_text": "algorithms learn from labeled training data",
                        "source_text": "Supervised learning algorithms build a mathematical model using training data"
                    }
                ],
                "total_citations": 1,
                "success": True
            }
            
            # Perform search with answer generation
            search_request = {
                "query": "what is supervised learning",
                "search_type": "hybrid",
                "generate_answer": True,
                "include_citations": True,
                "top_k": 5
            }
            
            response = await async_client.post(
                "/api/v1/search/answer",
                json=search_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "answer" in data
            assert "citations" in data
            assert len(data["citations"]) > 0
            assert data["citations"][0]["document_title"] == "ML Guide"
    
    @pytest.mark.trio
    async def test_knowledge_graph_integration_workflow(self, async_client):
        """Test workflow with knowledge graph integration."""
        
        with patch('app.services.kg.entity_extractor') as mock_entities, \
             patch('app.services.kg.relation_extractor') as mock_relations, \
             patch('app.services.kg.graph_builder') as mock_graph, \
             patch('app.services.search.search_service') as mock_search:
            
            # Mock entity extraction
            mock_entities.extract_entities.return_value = {
                "entities": [
                    {"name": "Machine Learning", "type": "concept", "confidence": 0.95},
                    {"name": "Supervised Learning", "type": "concept", "confidence": 0.92},
                    {"name": "Training Data", "type": "concept", "confidence": 0.88}
                ],
                "total_entities": 3
            }
            
            # Mock relation extraction
            mock_relations.extract_relations.return_value = {
                "relations": [
                    {
                        "subject": "Supervised Learning",
                        "predicate": "is_type_of",
                        "object": "Machine Learning",
                        "confidence": 0.9
                    },
                    {
                        "subject": "Supervised Learning",
                        "predicate": "uses",
                        "object": "Training Data",
                        "confidence": 0.85
                    }
                ],
                "total_relations": 2
            }
            
            # Mock graph building
            mock_graph.build_graph.return_value = {
                "nodes": 3,
                "edges": 2,
                "communities": 1,
                "graph_metrics": {"density": 0.67, "clustering_coefficient": 0.8}
            }
            
            # Mock enhanced search with KG
            mock_search.search_with_kg.return_value = {
                "query": "machine learning types",
                "results": [
                    {
                        "chunk_id": "chunk_types",
                        "content": "Types of machine learning include supervised and unsupervised learning.",
                        "score": 0.94,
                        "kg_enhanced": True,
                        "related_entities": ["Supervised Learning", "Unsupervised Learning"]
                    }
                ],
                "kg_context": {
                    "related_concepts": ["Supervised Learning", "Training Data"],
                    "concept_relationships": ["is_type_of", "uses"]
                },
                "success": True
            }
            
            # Test KG-enhanced search
            search_request = {
                "query": "machine learning types",
                "search_type": "hybrid",
                "use_knowledge_graph": True,
                "top_k": 10
            }
            
            response = await async_client.post(
                "/api/v1/search/kg-enhanced",
                json=search_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "kg_context" in data
            assert len(data["kg_context"]["related_concepts"]) > 0
            assert data["results"][0]["kg_enhanced"] == True
    
    @pytest.mark.trio
    async def test_multi_document_analysis_workflow(self, async_client):
        """Test workflow analyzing multiple documents together."""
        
        with patch('app.services.document_processing.document_processor') as mock_processor, \
             patch('app.services.search.search_service') as mock_search, \
             patch('app.services.analysis.document_analyzer') as mock_analyzer:
            
            # Mock processing multiple documents
            mock_processor.process_batch.return_value = {
                "processed_documents": [
                    {"document_id": "doc_1", "title": "ML Basics", "status": "completed"},
                    {"document_id": "doc_2", "title": "Deep Learning", "status": "completed"},
                    {"document_id": "doc_3", "title": "NLP Techniques", "status": "completed"}
                ],
                "total_processed": 3,
                "batch_processing_time": 15.2
            }
            
            # Mock cross-document analysis
            mock_analyzer.analyze_documents.return_value = {
                "document_similarities": [
                    {"doc1": "doc_1", "doc2": "doc_2", "similarity": 0.78},
                    {"doc1": "doc_1", "doc2": "doc_3", "similarity": 0.65},
                    {"doc1": "doc_2", "doc2": "doc_3", "similarity": 0.72}
                ],
                "common_themes": ["machine learning", "algorithms", "data processing"],
                "topic_distribution": {
                    "machine_learning": 0.45,
                    "deep_learning": 0.30,
                    "natural_language": 0.25
                }
            }
            
            # Mock search across multiple documents
            mock_search.search_multi_document.return_value = {
                "query": "compare machine learning approaches",
                "results": [
                    {
                        "chunk_id": "chunk_ml_basic",
                        "document_id": "doc_1",
                        "content": "Traditional machine learning uses feature engineering...",
                        "score": 0.89
                    },
                    {
                        "chunk_id": "chunk_dl_approach",
                        "document_id": "doc_2", 
                        "content": "Deep learning automatically learns features...",
                        "score": 0.87
                    }
                ],
                "cross_document_insights": {
                    "comparison_points": ["feature engineering", "automation", "complexity"],
                    "contrasts": ["manual vs automatic", "shallow vs deep"]
                },
                "success": True
            }
            
            # Test batch document upload
            batch_request = {
                "documents": [
                    {"title": "ML Basics", "content": "Basic ML content"},
                    {"title": "Deep Learning", "content": "Deep learning content"},
                    {"title": "NLP Techniques", "content": "NLP content"}
                ],
                "analyze_relationships": True
            }
            
            upload_response = await async_client.post(
                "/api/v1/documents/batch-upload",
                json=batch_request
            )
            
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert upload_data["total_processed"] == 3
            
            # Test cross-document analysis
            analysis_response = await async_client.post(
                "/api/v1/analysis/cross-document",
                json={"document_ids": ["doc_1", "doc_2", "doc_3"]}
            )
            
            assert analysis_response.status_code == 200
            analysis_data = analysis_response.json()
            assert len(analysis_data["document_similarities"]) > 0
            assert len(analysis_data["common_themes"]) > 0
            
            # Test multi-document search
            search_request = {
                "query": "compare machine learning approaches",
                "document_ids": ["doc_1", "doc_2", "doc_3"],
                "search_type": "cross_document",
                "include_comparisons": True
            }
            
            search_response = await async_client.post(
                "/api/v1/search/multi-document",
                json=search_request
            )
            
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["success"] == True
            assert "cross_document_insights" in search_data
            assert len(search_data["results"]) > 1  # Results from multiple documents


class TestPerformanceWorkflows:
    """End-to-end tests for performance-critical workflows."""
    
    @pytest.mark.trio
    async def test_large_document_processing_workflow(self, async_client):
        """Test processing workflow for large documents."""
        
        with patch('app.services.document_processing.document_processor') as mock_processor, \
             patch('app.services.chunking.chunking_manager') as mock_chunking, \
             patch('app.core.monitoring.health_checker') as mock_health:
            
            # Mock processing large document
            mock_processor.process_large_document.return_value = {
                "document_id": "large_doc_1",
                "status": "completed",
                "file_size_mb": 50,
                "chunks_created": 500,
                "processing_time": 45.2,
                "memory_usage_peak": "2.1GB",
                "performance_metrics": {
                    "parsing_time": 12.3,
                    "chunking_time": 18.7,
                    "embedding_time": 14.2
                }
            }
            
            # Mock chunking with streaming
            mock_chunking.chunk_large_document.return_value = {
                "chunks_processed": 500,
                "streaming_enabled": True,
                "memory_efficient": True,
                "chunk_size_avg": 512,
                "processing_rate": "11.1 chunks/sec"
            }
            
            # Mock system health during processing
            mock_health.get_system_health.return_value = {
                "overall_status": "healthy",
                "system_metrics": {
                    "cpu_percent": 75.0,
                    "memory_percent": 68.0,
                    "disk_percent": 45.0
                }
            }
            
            # Test large document upload
            large_doc_request = {
                "title": "Large Technical Manual",
                "description": "50MB technical documentation",
                "file_size": 52428800,  # 50MB
                "processing_options": {
                    "enable_streaming": True,
                    "chunk_size": 512,
                    "memory_limit": "4GB",
                    "priority": "high"
                }
            }
            
            response = await async_client.post(
                "/api/v1/documents/upload-large",
                json=large_doc_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["chunks_created"] == 500
            assert data["performance_metrics"]["processing_time"] < 60  # Under 1 minute
            
            # Check system health during processing
            health_response = await async_client.get("/api/v1/monitoring/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["overall_status"] == "healthy"
            assert health_data["system_metrics"]["memory_percent"] < 80  # Memory under control
    
    @pytest.mark.trio
    async def test_concurrent_processing_workflow(self, async_client):
        """Test concurrent document processing workflow."""
        
        with patch('app.services.document_processing.document_processor') as mock_processor:
            
            # Mock concurrent processing
            mock_processor.process_concurrent.return_value = {
                "batch_id": "batch_001",
                "documents_processed": 10,
                "concurrent_workers": 4,
                "total_processing_time": 25.6,
                "avg_time_per_document": 2.56,
                "success_rate": 1.0,
                "failed_documents": []
            }
            
            # Test concurrent upload
            concurrent_request = {
                "documents": [
                    {"title": f"Document {i}", "content": f"Content for document {i}"}
                    for i in range(10)
                ],
                "processing_options": {
                    "concurrent_workers": 4,
                    "batch_size": 10,
                    "timeout_per_document": 30
                }
            }
            
            response = await async_client.post(
                "/api/v1/documents/batch-concurrent",
                json=concurrent_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["documents_processed"] == 10
            assert data["success_rate"] == 1.0
            assert len(data["failed_documents"]) == 0
            assert data["avg_time_per_document"] < 5.0  # Efficient processing


class TestErrorRecoveryWorkflows:
    """End-to-end tests for error recovery and resilience."""
    
    @pytest.mark.trio
    async def test_processing_failure_recovery(self, async_client):
        """Test recovery from processing failures."""
        
        with patch('app.services.document_processing.document_processor') as mock_processor, \
             patch('app.services.recovery.recovery_manager') as mock_recovery:
            
            # Mock initial processing failure
            mock_processor.process_document.side_effect = [
                Exception("Processing failed - network timeout"),
                {  # Successful retry
                    "document_id": "recovered_doc",
                    "status": "completed",
                    "retry_count": 1,
                    "recovery_time": 5.2
                }
            ]
            
            # Mock recovery manager
            mock_recovery.recover_failed_processing.return_value = {
                "recovery_successful": True,
                "retry_attempts": 1,
                "recovery_strategy": "exponential_backoff",
                "final_status": "completed"
            }
            
            # Test document upload with failure and recovery
            upload_request = {
                "title": "Document with Recovery",
                "content": "Content that initially fails to process",
                "retry_options": {
                    "max_retries": 3,
                    "backoff_strategy": "exponential",
                    "timeout": 30
                }
            }
            
            response = await async_client.post(
                "/api/v1/documents/upload-with-retry",
                json=upload_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["retry_count"] == 1
            assert data["recovery_time"] > 0
    
    @pytest.mark.trio
    async def test_system_degradation_handling(self, async_client):
        """Test handling of system degradation scenarios."""
        
        with patch('app.core.monitoring.health_checker') as mock_health, \
             patch('app.services.search.search_service') as mock_search:
            
            # Mock degraded system state
            mock_health.get_system_health.return_value = {
                "overall_status": "degraded",
                "health_checks": {
                    "database": {"status": "healthy"},
                    "cache": {"status": "degraded", "response_time": 2.5},
                    "elasticsearch": {"status": "healthy"}
                },
                "system_metrics": {
                    "cpu_percent": 85.0,
                    "memory_percent": 78.0
                }
            }
            
            # Mock search with degraded performance
            mock_search.search.return_value = {
                "query": "test query",
                "results": [
                    {
                        "chunk_id": "chunk_1",
                        "content": "Test content",
                        "score": 0.8
                    }
                ],
                "search_time": 3.2,  # Slower than normal
                "degraded_performance": True,
                "fallback_used": True,
                "success": True
            }
            
            # Test search during system degradation
            search_request = {
                "query": "test query during degradation",
                "search_type": "hybrid",
                "fallback_enabled": True
            }
            
            response = await async_client.post(
                "/api/v1/search",
                json=search_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["degraded_performance"] == True
            assert data["fallback_used"] == True
            assert data["search_time"] > 3.0  # Slower performance
            
            # Check system health
            health_response = await async_client.get("/api/v1/monitoring/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["overall_status"] == "degraded"


class TestSecurityWorkflows:
    """End-to-end tests for security-related workflows."""
    
    @pytest.mark.trio
    async def test_secure_document_processing(self, async_client):
        """Test secure document processing workflow."""
        
        with patch('app.services.security.document_scanner') as mock_scanner, \
             patch('app.services.document_processing.document_processor') as mock_processor:
            
            # Mock security scanning
            mock_scanner.scan_document.return_value = {
                "scan_result": "clean",
                "threats_detected": [],
                "scan_time": 1.2,
                "security_score": 0.95,
                "safe_to_process": True
            }
            
            # Mock secure processing
            mock_processor.process_secure.return_value = {
                "document_id": "secure_doc_1",
                "status": "completed",
                "security_validated": True,
                "encryption_applied": True,
                "access_controls": ["user_group_1", "admin"]
            }
            
            # Test secure document upload
            secure_request = {
                "title": "Confidential Document",
                "content": "Sensitive content requiring security validation",
                "security_options": {
                    "scan_for_threats": True,
                    "encrypt_at_rest": True,
                    "access_control": ["user_group_1", "admin"],
                    "audit_logging": True
                }
            }
            
            response = await async_client.post(
                "/api/v1/documents/upload-secure",
                json=secure_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["security_validated"] == True
            assert data["encryption_applied"] == True
            assert "admin" in data["access_controls"]
    
    @pytest.mark.trio
    async def test_audit_trail_workflow(self, async_client):
        """Test audit trail generation workflow."""
        
        with patch('app.services.audit.audit_logger') as mock_audit:
            
            # Mock audit logging
            mock_audit.log_activity.return_value = {
                "audit_id": "audit_001",
                "activity": "document_search",
                "user_id": "e2e-test-user",
                "timestamp": "2023-01-01T12:00:00Z",
                "details": {
                    "query": "sensitive information",
                    "results_count": 3,
                    "access_granted": True
                }
            }
            
            # Test audited search
            search_request = {
                "query": "sensitive information",
                "search_type": "hybrid",
                "audit_required": True
            }
            
            response = await async_client.post(
                "/api/v1/search/audited",
                json=search_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "audit_id" in data
            assert data["audit_id"] == "audit_001"
            
            # Check audit log
            audit_response = await async_client.get(
                f"/api/v1/audit/logs/{data['audit_id']}"
            )
            
            assert audit_response.status_code == 200
            audit_data = audit_response.json()
            assert audit_data["activity"] == "document_search"
            assert audit_data["user_id"] == "e2e-test-user"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])