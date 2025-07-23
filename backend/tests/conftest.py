"""
Pytest configuration and fixtures for comprehensive testing.
"""
import pytest
import trio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Generator, AsyncGenerator
import json
import os

import trio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.db.session import Base
from app.models.document import Document, Chunk, ChunkEnrichment
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.search.retriever import Retriever
from app.services.response.citation_generator import CitationGenerator
from app.services.response.citation_validator import CitationValidator
from app.core.logging_config import reset_performance_metrics


# Test configuration
@pytest.fixture(scope="session")
def test_settings():
    """Test settings configuration."""
    return Settings(
        database_url="sqlite:///:memory:",
        redis_url="redis://localhost:6379/15",  # Use different DB for tests
        elasticsearch_url="http://localhost:9200",
        test_mode=True,
        log_level="DEBUG"
    )


# Database fixtures
@pytest.fixture(scope="session")
def test_engine(test_settings):
    """Create test database engine."""
    engine = create_engine(
        test_settings.database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def test_db_session(test_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_engine
    )
    
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir):
    """Create temporary file for tests."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("Test file content for testing purposes.")
    return temp_file_path


# Mock service fixtures
@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    service = Mock(spec=EmbeddingService)
    service._initialized = True
    service.initialize = AsyncMock()
    service.generate_embeddings = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
    service.generate_batch_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return service


@pytest.fixture
def mock_retriever():
    """Mock retriever service."""
    service = Mock(spec=Retriever)
    service._initialized = True
    service.initialize = AsyncMock()
    service.search = AsyncMock(return_value={
        'results': [
            {
                'chunk_id': 'chunk_1',
                'document_id': 'doc_1',
                'content': 'Test chunk content',
                'score': 0.9,
                'metadata': {'page_number': 1}
            }
        ],
        'total': 1
    })
    return service


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager."""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    cache.exists = AsyncMock(return_value=False)
    return cache


@pytest.fixture
def mock_document_store():
    """Mock document store."""
    store = Mock()
    store._initialized = True
    store.initialize = AsyncMock()
    store.get_document = AsyncMock(return_value={
        'id': 'doc_1',
        'title': 'Test Document',
        'content': 'Test document content',
        'metadata': {'author': 'Test Author'}
    })
    store.search_documents_by_metadata = AsyncMock(return_value=[])
    store.count_documents_by_metadata = AsyncMock(return_value=0)
    return store


# Sample data fixtures
@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        'id': 'test_doc_1',
        'title': 'Test Document',
        'description': 'A test document for unit testing',
        'content': 'This is the content of the test document. It contains multiple sentences for testing purposes.',
        'metadata': {
            'author': 'Test Author',
            'created_at': '2023-01-01T00:00:00Z',
            'file_type': 'txt',
            'file_size': 1024
        },
        'file_path': '/test/path/document.txt',
        'processing_status': 'completed'
    }


@pytest.fixture
def sample_chunk_data():
    """Sample chunk data for testing."""
    return [
        {
            'chunk_id': 'chunk_1',
            'document_id': 'test_doc_1',
            'content': 'This is the first chunk of the test document.',
            'chunk_index': 0,
            'start_char': 0,
            'end_char': 50,
            'metadata': {
                'page_number': 1,
                'section': 'introduction'
            },
            'embeddings': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        {
            'chunk_id': 'chunk_2',
            'document_id': 'test_doc_1',
            'content': 'This is the second chunk with different content.',
            'chunk_index': 1,
            'start_char': 51,
            'end_char': 100,
            'metadata': {
                'page_number': 1,
                'section': 'body'
            },
            'embeddings': [0.6, 0.7, 0.8, 0.9, 1.0]
        }
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return {
        'results': [
            {
                'chunk_id': 'chunk_1',
                'document_id': 'test_doc_1',
                'content': 'This is relevant content for the search query.',
                'score': 0.95,
                'metadata': {
                    'page_number': 1,
                    'document_title': 'Test Document'
                }
            },
            {
                'chunk_id': 'chunk_2',
                'document_id': 'test_doc_1',
                'content': 'This is also relevant but with lower score.',
                'score': 0.85,
                'metadata': {
                    'page_number': 2,
                    'document_title': 'Test Document'
                }
            }
        ],
        'total': 2,
        'query': 'test query',
        'search_time': 0.1
    }


@pytest.fixture
def sample_response_text():
    """Sample response text for testing."""
    return """
    Machine learning is a powerful technique for analyzing data and making predictions.
    It uses algorithms to automatically learn patterns from datasets.
    Deep learning, a subset of machine learning, employs neural networks with multiple layers.
    Python is widely used for implementing machine learning solutions due to its rich ecosystem.
    """


# Service instance fixtures
@pytest.fixture
def citation_generator_service(mock_cache_manager, mock_document_store, mock_retriever):
    """Citation generator service with mocked dependencies."""
    with patch('app.services.response.citation_generator.cache_manager', mock_cache_manager):
        with patch('app.services.response.citation_generator.document_store', mock_document_store):
            with patch('app.services.response.citation_generator.basic_retriever', mock_retriever):
                service = CitationGenerator()
                service._initialized = True
                return service


@pytest.fixture
def citation_validator_service(mock_cache_manager, mock_document_store, mock_embedding_service):
    """Citation validator service with mocked dependencies."""
    with patch('app.services.response.citation_validator.cache_manager', mock_cache_manager):
        with patch('app.services.response.citation_validator.document_store', mock_document_store):
            with patch('app.services.response.citation_validator.embedding_service', mock_embedding_service):
                service = CitationValidator()
                service._initialized = True
                return service


# Test data generators
@pytest.fixture
def generate_test_documents():
    """Generate test documents."""
    def _generate(count: int = 5) -> List[Dict[str, Any]]:
        documents = []
        for i in range(count):
            documents.append({
                'id': f'test_doc_{i+1}',
                'title': f'Test Document {i+1}',
                'description': f'Description for test document {i+1}',
                'content': f'Content of test document {i+1}. This contains test information.',
                'metadata': {
                    'author': f'Author {i+1}',
                    'category': 'test',
                    'created_at': f'2023-01-{i+1:02d}T00:00:00Z'
                },
                'file_path': f'/test/path/document_{i+1}.txt',
                'processing_status': 'completed'
            })
        return documents
    return _generate


@pytest.fixture
def generate_test_chunks():
    """Generate test chunks."""
    def _generate(document_id: str, count: int = 3) -> List[Dict[str, Any]]:
        chunks = []
        for i in range(count):
            chunks.append({
                'chunk_id': f'chunk_{document_id}_{i+1}',
                'document_id': document_id,
                'content': f'Chunk {i+1} content for document {document_id}.',
                'chunk_index': i,
                'start_char': i * 50,
                'end_char': (i + 1) * 50,
                'metadata': {
                    'page_number': (i // 2) + 1,
                    'section': f'section_{i+1}'
                },
                'embeddings': [float(j) * 0.1 for j in range(5)]
            })
        return chunks
    return _generate


# Performance testing fixtures
@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset performance metrics before each test."""
    reset_performance_metrics()
    yield
    reset_performance_metrics()


# Async testing support
@pytest.fixture
def trio_mode():
    """Enable trio mode for async tests."""
    return True


# Mock external services
@pytest.fixture
def mock_elasticsearch():
    """Mock Elasticsearch client."""
    client = Mock()
    client.search = AsyncMock(return_value={
        'hits': {
            'total': {'value': 1},
            'hits': [
                {
                    '_id': 'chunk_1',
                    '_score': 0.9,
                    '_source': {
                        'content': 'Test content',
                        'document_id': 'doc_1',
                        'metadata': {}
                    }
                }
            ]
        }
    })
    client.index = AsyncMock()
    client.delete = AsyncMock()
    client.indices.create = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.cluster.health = AsyncMock(return_value={
        'status': 'green',
        'number_of_nodes': 1,
        'active_primary_shards': 1,
        'active_shards': 1
    })
    return client


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    client = Mock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.exists = AsyncMock(return_value=False)
    client.ping = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_minio():
    """Mock MinIO client."""
    client = Mock()
    client.bucket_exists = Mock(return_value=True)
    client.make_bucket = Mock()
    client.put_object = Mock()
    client.get_object = Mock()
    client.remove_object = Mock()
    client.list_objects = Mock(return_value=[])
    return client


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# File handling fixtures
@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"


@pytest.fixture
def sample_docx_content():
    """Sample DOCX content for testing."""
    # Minimal DOCX structure
    return b"PK\x03\x04\x14\x00\x00\x00\x08\x00\x00\x00!\x00"


@pytest.fixture
def sample_text_files(temp_dir):
    """Create sample text files for testing."""
    files = {}
    
    # Create different types of text files
    files['simple'] = temp_dir / "simple.txt"
    files['simple'].write_text("This is a simple text file for testing.")
    
    files['multiline'] = temp_dir / "multiline.txt"
    files['multiline'].write_text("""This is a multiline text file.
It contains multiple lines of text.
Each line has different content.
This is useful for testing text processing.""")
    
    files['structured'] = temp_dir / "structured.txt"
    files['structured'].write_text("""# Title
This is a structured document.

## Section 1
Content of section 1.

## Section 2
Content of section 2.

### Subsection 2.1
Content of subsection 2.1.""")
    
    return files


# Error simulation fixtures
@pytest.fixture
def simulate_database_error():
    """Simulate database connection errors."""
    def _simulate():
        return Exception("Database connection failed")
    return _simulate


@pytest.fixture
def simulate_network_error():
    """Simulate network errors."""
    def _simulate():
        return Exception("Network connection timeout")
    return _simulate


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Perform any necessary cleanup
    # This runs after each test


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# Async test support
@pytest.fixture(scope="session")
def anyio_backend():
    """Use trio as the async backend for tests."""
    return "trio"