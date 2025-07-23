"""
Integration tests for database operations and data persistence.
"""
import pytest
import trio
from typing import List, Dict, Any
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db_session
from app.models.document import Document, Chunk, ChunkEnrichment
from app.core.config import Settings


@pytest.fixture(scope="session")
def test_db_engine():
    """Create test database engine."""
    engine = create_engine(
        "sqlite:///:memory:",
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
def test_db_session(test_db_engine):
    """Create test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine
    )
    
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
async def async_test_db_session(test_db_engine):
    """Create async test database session."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # For testing, we'll use a sync session wrapped in async context
    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=test_db_engine
    )
    
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()


class TestDocumentModel:
    """Integration tests for Document model operations."""
    
    def test_create_document(self, test_db_session):
        """Test creating a document in the database."""
        document = Document(
            id="test_doc_1",
            title="Test Document",
            description="A test document for integration testing",
            file_path="/test/path/document.txt",
            file_type="txt",
            file_size=1024,
            processing_status="pending",
            metadata_={"author": "Test Author", "category": "test"}
        )
        
        test_db_session.add(document)
        test_db_session.commit()
        
        # Retrieve and verify
        retrieved_doc = test_db_session.query(Document).filter(Document.id == "test_doc_1").first()
        
        assert retrieved_doc is not None
        assert retrieved_doc.title == "Test Document"
        assert retrieved_doc.description == "A test document for integration testing"
        assert retrieved_doc.file_type == "txt"
        assert retrieved_doc.processing_status == "pending"
        assert retrieved_doc.metadata_["author"] == "Test Author"
    
    def test_update_document(self, test_db_session):
        """Test updating a document in the database."""
        # Create document
        document = Document(
            id="test_doc_2",
            title="Original Title",
            description="Original description",
            file_path="/test/path/document.txt",
            file_type="txt",
            file_size=1024,
            processing_status="pending"
        )
        
        test_db_session.add(document)
        test_db_session.commit()
        
        # Update document
        document.title = "Updated Title"
        document.description = "Updated description"
        document.processing_status = "completed"
        test_db_session.commit()
        
        # Retrieve and verify
        retrieved_doc = test_db_session.query(Document).filter(Document.id == "test_doc_2").first()
        
        assert retrieved_doc.title == "Updated Title"
        assert retrieved_doc.description == "Updated description"
        assert retrieved_doc.processing_status == "completed"
    
    def test_delete_document(self, test_db_session):
        """Test deleting a document from the database."""
        # Create document
        document = Document(
            id="test_doc_3",
            title="Document to Delete",
            description="This document will be deleted",
            file_path="/test/path/document.txt",
            file_type="txt",
            file_size=1024,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.commit()
        
        # Verify it exists
        retrieved_doc = test_db_session.query(Document).filter(Document.id == "test_doc_3").first()
        assert retrieved_doc is not None
        
        # Delete document
        test_db_session.delete(document)
        test_db_session.commit()
        
        # Verify it's deleted
        deleted_doc = test_db_session.query(Document).filter(Document.id == "test_doc_3").first()
        assert deleted_doc is None
    
    def test_document_with_chunks(self, test_db_session):
        """Test document with associated chunks."""
        # Create document
        document = Document(
            id="test_doc_4",
            title="Document with Chunks",
            description="Document that has chunks",
            file_path="/test/path/document.txt",
            file_type="txt",
            file_size=2048,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.flush()  # Flush to get the document ID
        
        # Create chunks
        chunk1 = Chunk(
            id="chunk_1",
            document_id=document.id,
            content="This is the first chunk of the document.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata_={"page_number": 1, "section": "intro"}
        )
        
        chunk2 = Chunk(
            id="chunk_2",
            document_id=document.id,
            content="This is the second chunk of the document.",
            chunk_index=1,
            start_char=51,
            end_char=100,
            metadata_={"page_number": 1, "section": "body"}
        )
        
        test_db_session.add_all([chunk1, chunk2])
        test_db_session.commit()
        
        # Retrieve document with chunks
        retrieved_doc = test_db_session.query(Document).filter(Document.id == "test_doc_4").first()
        
        assert retrieved_doc is not None
        assert len(retrieved_doc.chunks) == 2
        assert retrieved_doc.chunks[0].content == "This is the first chunk of the document."
        assert retrieved_doc.chunks[1].content == "This is the second chunk of the document."


class TestChunkModel:
    """Integration tests for Chunk model operations."""
    
    def test_create_chunk(self, test_db_session):
        """Test creating a chunk in the database."""
        # Create parent document first
        document = Document(
            id="test_doc_5",
            title="Parent Document",
            description="Parent document for chunk testing",
            file_path="/test/path/document.txt",
            file_type="txt",
            file_size=1024,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.flush()
        
        # Create chunk
        chunk = Chunk(
            id="test_chunk_1",
            document_id=document.id,
            content="This is a test chunk with some content for testing purposes.",
            chunk_index=0,
            start_char=0,
            end_char=65,
            metadata_={"page_number": 1, "section": "test", "confidence": 0.95}
        )
        
        test_db_session.add(chunk)
        test_db_session.commit()
        
        # Retrieve and verify
        retrieved_chunk = test_db_session.query(Chunk).filter(Chunk.id == "test_chunk_1").first()
        
        assert retrieved_chunk is not None
        assert retrieved_chunk.content == "This is a test chunk with some content for testing purposes."
        assert retrieved_chunk.chunk_index == 0
        assert retrieved_chunk.start_char == 0
        assert retrieved_chunk.end_char == 65
        assert retrieved_chunk.metadata_["page_number"] == 1
        assert retrieved_chunk.metadata_["confidence"] == 0.95
        assert retrieved_chunk.document_id == document.id
    
    def test_chunk_with_enrichments(self, test_db_session):
        """Test chunk with enrichments."""
        # Create parent document
        document = Document(
            id="test_doc_6",
            title="Document for Enrichment Testing",
            description="Document to test chunk enrichments",
            file_path="/test/path/document.txt",
            file_type="txt",
            file_size=1024,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.flush()
        
        # Create chunk
        chunk = Chunk(
            id="test_chunk_2",
            document_id=document.id,
            content="This chunk will have enrichments like keywords and questions.",
            chunk_index=0,
            start_char=0,
            end_char=70,
            metadata_={"page_number": 1}
        )
        
        test_db_session.add(chunk)
        test_db_session.flush()
        
        # Create enrichments
        keyword_enrichment = ChunkEnrichment(
            id="enrichment_1",
            chunk_id=chunk.id,
            enrichment_type="keywords",
            enrichment_data={
                "keywords": ["chunk", "enrichments", "keywords", "questions"],
                "confidence": 0.9
            }
        )
        
        question_enrichment = ChunkEnrichment(
            id="enrichment_2",
            chunk_id=chunk.id,
            enrichment_type="questions",
            enrichment_data={
                "questions": [
                    "What are enrichments?",
                    "How do keywords work?"
                ],
                "confidence": 0.85
            }
        )
        
        test_db_session.add_all([keyword_enrichment, question_enrichment])
        test_db_session.commit()
        
        # Retrieve chunk with enrichments
        retrieved_chunk = test_db_session.query(Chunk).filter(Chunk.id == "test_chunk_2").first()
        
        assert retrieved_chunk is not None
        assert len(retrieved_chunk.enrichments) == 2
        
        # Check keyword enrichment
        keyword_enr = next((e for e in retrieved_chunk.enrichments if e.enrichment_type == "keywords"), None)
        assert keyword_enr is not None
        assert "keywords" in keyword_enr.enrichment_data
        assert len(keyword_enr.enrichment_data["keywords"]) == 4
        
        # Check question enrichment
        question_enr = next((e for e in retrieved_chunk.enrichments if e.enrichment_type == "questions"), None)
        assert question_enr is not None
        assert "questions" in question_enr.enrichment_data
        assert len(question_enr.enrichment_data["questions"]) == 2


class TestDatabaseQueries:
    """Integration tests for complex database queries."""
    
    def test_search_documents_by_title(self, test_db_session):
        """Test searching documents by title."""
        # Create test documents
        documents = [
            Document(
                id="search_doc_1",
                title="Machine Learning Basics",
                description="Introduction to machine learning",
                file_path="/test/ml_basics.txt",
                file_type="txt",
                file_size=1024,
                processing_status="completed"
            ),
            Document(
                id="search_doc_2",
                title="Deep Learning Advanced",
                description="Advanced deep learning techniques",
                file_path="/test/dl_advanced.txt",
                file_type="txt",
                file_size=2048,
                processing_status="completed"
            ),
            Document(
                id="search_doc_3",
                title="Python Programming",
                description="Python programming guide",
                file_path="/test/python.txt",
                file_type="txt",
                file_size=1536,
                processing_status="completed"
            )
        ]
        
        test_db_session.add_all(documents)
        test_db_session.commit()
        
        # Search for documents with "Learning" in title
        results = test_db_session.query(Document).filter(
            Document.title.contains("Learning")
        ).all()
        
        assert len(results) == 2
        titles = [doc.title for doc in results]
        assert "Machine Learning Basics" in titles
        assert "Deep Learning Advanced" in titles
        assert "Python Programming" not in titles
    
    def test_get_chunks_by_document(self, test_db_session):
        """Test retrieving chunks for a specific document."""
        # Create document
        document = Document(
            id="chunk_test_doc",
            title="Document for Chunk Testing",
            description="Testing chunk retrieval",
            file_path="/test/chunk_test.txt",
            file_type="txt",
            file_size=1024,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.flush()
        
        # Create multiple chunks
        chunks = []
        for i in range(5):
            chunk = Chunk(
                id=f"chunk_test_{i}",
                document_id=document.id,
                content=f"This is chunk number {i} with test content.",
                chunk_index=i,
                start_char=i * 50,
                end_char=(i + 1) * 50,
                metadata_={"page_number": (i // 2) + 1}
            )
            chunks.append(chunk)
        
        test_db_session.add_all(chunks)
        test_db_session.commit()
        
        # Retrieve chunks for the document
        retrieved_chunks = test_db_session.query(Chunk).filter(
            Chunk.document_id == document.id
        ).order_by(Chunk.chunk_index).all()
        
        assert len(retrieved_chunks) == 5
        for i, chunk in enumerate(retrieved_chunks):
            assert chunk.chunk_index == i
            assert f"chunk number {i}" in chunk.content
    
    def test_filter_documents_by_status(self, test_db_session):
        """Test filtering documents by processing status."""
        # Create documents with different statuses
        documents = [
            Document(
                id="status_doc_1",
                title="Pending Document",
                description="Document in pending status",
                file_path="/test/pending.txt",
                file_type="txt",
                file_size=1024,
                processing_status="pending"
            ),
            Document(
                id="status_doc_2",
                title="Processing Document",
                description="Document being processed",
                file_path="/test/processing.txt",
                file_type="txt",
                file_size=1024,
                processing_status="processing"
            ),
            Document(
                id="status_doc_3",
                title="Completed Document",
                description="Document processing completed",
                file_path="/test/completed.txt",
                file_type="txt",
                file_size=1024,
                processing_status="completed"
            ),
            Document(
                id="status_doc_4",
                title="Another Completed Document",
                description="Another completed document",
                file_path="/test/completed2.txt",
                file_type="txt",
                file_size=1024,
                processing_status="completed"
            )
        ]
        
        test_db_session.add_all(documents)
        test_db_session.commit()
        
        # Filter by completed status
        completed_docs = test_db_session.query(Document).filter(
            Document.processing_status == "completed"
        ).all()
        
        assert len(completed_docs) == 2
        
        # Filter by pending status
        pending_docs = test_db_session.query(Document).filter(
            Document.processing_status == "pending"
        ).all()
        
        assert len(pending_docs) == 1
        assert pending_docs[0].title == "Pending Document"
    
    def test_aggregate_queries(self, test_db_session):
        """Test aggregate queries on documents and chunks."""
        # Create documents with chunks
        for doc_idx in range(3):
            document = Document(
                id=f"agg_doc_{doc_idx}",
                title=f"Aggregate Test Document {doc_idx}",
                description=f"Document {doc_idx} for aggregate testing",
                file_path=f"/test/agg_{doc_idx}.txt",
                file_type="txt",
                file_size=1024 * (doc_idx + 1),
                processing_status="completed"
            )
            
            test_db_session.add(document)
            test_db_session.flush()
            
            # Create chunks for each document
            for chunk_idx in range(doc_idx + 2):  # Different number of chunks per document
                chunk = Chunk(
                    id=f"agg_chunk_{doc_idx}_{chunk_idx}",
                    document_id=document.id,
                    content=f"Chunk {chunk_idx} of document {doc_idx}",
                    chunk_index=chunk_idx,
                    start_char=chunk_idx * 50,
                    end_char=(chunk_idx + 1) * 50,
                    metadata_={"page_number": 1}
                )
                test_db_session.add(chunk)
        
        test_db_session.commit()
        
        # Count total documents
        total_docs = test_db_session.query(Document).count()
        assert total_docs >= 3  # At least the 3 we just created
        
        # Count total chunks
        total_chunks = test_db_session.query(Chunk).count()
        assert total_chunks >= 6  # 2 + 3 + 4 = 9 chunks from our test documents
        
        # Count chunks per document
        from sqlalchemy import func
        chunk_counts = test_db_session.query(
            Document.id,
            Document.title,
            func.count(Chunk.id).label('chunk_count')
        ).outerjoin(Chunk).group_by(Document.id, Document.title).all()
        
        # Find our test documents in the results
        test_doc_counts = [row for row in chunk_counts if row.id.startswith('agg_doc_')]
        assert len(test_doc_counts) == 3
        
        # Verify chunk counts
        for row in test_doc_counts:
            if row.id == 'agg_doc_0':
                assert row.chunk_count == 2
            elif row.id == 'agg_doc_1':
                assert row.chunk_count == 3
            elif row.id == 'agg_doc_2':
                assert row.chunk_count == 4


class TestDatabaseTransactions:
    """Integration tests for database transactions."""
    
    def test_transaction_rollback(self, test_db_session):
        """Test transaction rollback functionality."""
        # Create a document
        document = Document(
            id="rollback_doc",
            title="Document for Rollback Test",
            description="This document should be rolled back",
            file_path="/test/rollback.txt",
            file_type="txt",
            file_size=1024,
            processing_status="pending"
        )
        
        test_db_session.add(document)
        
        # Verify it's in the session but not committed
        assert test_db_session.query(Document).filter(Document.id == "rollback_doc").first() is not None
        
        # Rollback the transaction
        test_db_session.rollback()
        
        # Verify it's no longer in the session
        assert test_db_session.query(Document).filter(Document.id == "rollback_doc").first() is None
    
    def test_transaction_commit(self, test_db_session):
        """Test transaction commit functionality."""
        # Create a document
        document = Document(
            id="commit_doc",
            title="Document for Commit Test",
            description="This document should be committed",
            file_path="/test/commit.txt",
            file_type="txt",
            file_size=1024,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.commit()
        
        # Verify it's persisted
        retrieved_doc = test_db_session.query(Document).filter(Document.id == "commit_doc").first()
        assert retrieved_doc is not None
        assert retrieved_doc.title == "Document for Commit Test"
    
    def test_cascade_delete(self, test_db_session):
        """Test cascade delete functionality."""
        # Create document with chunks and enrichments
        document = Document(
            id="cascade_doc",
            title="Document for Cascade Test",
            description="Testing cascade delete",
            file_path="/test/cascade.txt",
            file_type="txt",
            file_size=1024,
            processing_status="completed"
        )
        
        test_db_session.add(document)
        test_db_session.flush()
        
        # Create chunk
        chunk = Chunk(
            id="cascade_chunk",
            document_id=document.id,
            content="Chunk that should be cascade deleted",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata_={"page_number": 1}
        )
        
        test_db_session.add(chunk)
        test_db_session.flush()
        
        # Create enrichment
        enrichment = ChunkEnrichment(
            id="cascade_enrichment",
            chunk_id=chunk.id,
            enrichment_type="keywords",
            enrichment_data={"keywords": ["test", "cascade"]}
        )
        
        test_db_session.add(enrichment)
        test_db_session.commit()
        
        # Verify all objects exist
        assert test_db_session.query(Document).filter(Document.id == "cascade_doc").first() is not None
        assert test_db_session.query(Chunk).filter(Chunk.id == "cascade_chunk").first() is not None
        assert test_db_session.query(ChunkEnrichment).filter(ChunkEnrichment.id == "cascade_enrichment").first() is not None
        
        # Delete the document (should cascade to chunks and enrichments)
        test_db_session.delete(document)
        test_db_session.commit()
        
        # Verify cascade delete worked
        assert test_db_session.query(Document).filter(Document.id == "cascade_doc").first() is None
        assert test_db_session.query(Chunk).filter(Chunk.id == "cascade_chunk").first() is None
        assert test_db_session.query(ChunkEnrichment).filter(ChunkEnrichment.id == "cascade_enrichment").first() is None


class TestDatabasePerformance:
    """Integration tests for database performance."""
    
    def test_bulk_insert_performance(self, test_db_session):
        """Test bulk insert performance."""
        import time
        
        # Create many documents
        documents = []
        for i in range(100):
            document = Document(
                id=f"bulk_doc_{i}",
                title=f"Bulk Document {i}",
                description=f"Document {i} for bulk testing",
                file_path=f"/test/bulk_{i}.txt",
                file_type="txt",
                file_size=1024,
                processing_status="completed"
            )
            documents.append(document)
        
        # Measure bulk insert time
        start_time = time.time()
        test_db_session.add_all(documents)
        test_db_session.commit()
        end_time = time.time()
        
        bulk_time = end_time - start_time
        
        # Verify all documents were inserted
        count = test_db_session.query(Document).filter(
            Document.id.like("bulk_doc_%")
        ).count()
        
        assert count == 100
        assert bulk_time < 5.0  # Should complete within 5 seconds
    
    def test_query_performance(self, test_db_session):
        """Test query performance with indexes."""
        import time
        
        # Create documents with various attributes for testing
        documents = []
        for i in range(50):
            document = Document(
                id=f"perf_doc_{i}",
                title=f"Performance Document {i}",
                description=f"Document {i} for performance testing",
                file_path=f"/test/perf_{i}.txt",
                file_type="txt" if i % 2 == 0 else "pdf",
                file_size=1024 * (i + 1),
                processing_status="completed" if i % 3 == 0 else "pending"
            )
            documents.append(document)
        
        test_db_session.add_all(documents)
        test_db_session.commit()
        
        # Test query performance
        start_time = time.time()
        
        # Complex query with multiple filters
        results = test_db_session.query(Document).filter(
            Document.processing_status == "completed",
            Document.file_type == "txt",
            Document.file_size > 5000
        ).all()
        
        end_time = time.time()
        query_time = end_time - start_time
        
        # Should complete quickly even with multiple filters
        assert query_time < 1.0  # Should complete within 1 second
        assert len(results) > 0  # Should find some matching documents


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])