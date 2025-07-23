"""
Search API endpoints.
"""
import logging
from typing import List, Optional, Dict, Any
import time

import trio
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from app.services.search.retriever import basic_retriever
from app.services.response.answer_generator import answer_generator
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    search_type: str = Field("hybrid", description="Search type: sparse, dense, or hybrid")
    top_k: int = Field(20, description="Number of results to return", ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    generate_answer: bool = Field(True, description="Whether to generate an answer")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Chat history for context")


class SearchResult(BaseModel):
    """Search result model."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    search_type: str
    highlights: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class Citation(BaseModel):
    """Citation model."""
    chunk_id: str
    document_id: str
    content_preview: str
    source_number: int
    metadata: Dict[str, Any] = {}
    implicit: bool = False


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total: int
    max_score: float
    search_time: float
    search_type: str
    answer: Optional[str] = None
    citations: Optional[List[Citation]] = None
    confidence: Optional[float] = None
    generation_time: Optional[float] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., description="User question", min_length=1, max_length=1000)
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Previous messages")
    search_options: Optional[Dict[str, Any]] = Field(None, description="Search configuration")
    generation_config: Optional[Dict[str, Any]] = Field(None, description="LLM generation config")


class ChatResponse(BaseModel):
    """Chat response model."""
    query: str
    answer: str
    citations: List[Citation]
    confidence: float
    search_results_count: int
    total_time: float
    search_time: float
    generation_time: float
    error: Optional[str] = None


# Dependency functions
async def get_retriever():
    """Get initialized retriever."""
    if not basic_retriever._initialized:
        await basic_retriever.initialize()
    return basic_retriever


async def get_answer_generator():
    """Get initialized answer generator."""
    if not answer_generator._initialized:
        await answer_generator.initialize()
    return answer_generator


# API Endpoints
@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    retriever = Depends(get_retriever)
):
    """
    Search documents with optional answer generation.
    
    This endpoint performs document search using the specified method
    (sparse, dense, or hybrid) and optionally generates an answer
    based on the retrieved context.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Search request: query='{request.query[:50]}...', type={request.search_type}")
        
        # Perform search
        search_results = await retriever.search(
            query=request.query,
            document_ids=request.document_ids,
            search_type=request.search_type,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Convert results to response format
        results = []
        for result in search_results.get("results", []):
            results.append(SearchResult(**result))
        
        response_data = {
            "query": request.query,
            "results": results,
            "total": search_results.get("total", 0),
            "max_score": search_results.get("max_score", 0.0),
            "search_time": search_results.get("search_time", 0.0),
            "search_type": request.search_type
        }
        
        # Generate answer if requested
        if request.generate_answer and results:
            try:
                generator = await get_answer_generator()
                
                # Convert results back to dict format for answer generation
                context_chunks = [result.dict() for result in results]
                
                answer_data = await generator.generate_answer(
                    query=request.query,
                    context_chunks=context_chunks,
                    chat_history=request.chat_history
                )
                
                response_data.update({
                    "answer": answer_data.get("answer"),
                    "citations": [Citation(**citation) for citation in answer_data.get("citations", [])],
                    "confidence": answer_data.get("confidence"),
                    "generation_time": answer_data.get("generation_time")
                })
                
            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                response_data["error"] = f"Answer generation failed: {str(e)}"
        
        total_time = time.time() - start_time
        logger.info(f"Search completed in {total_time:.3f}s, {len(results)} results")
        
        return SearchResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    retriever = Depends(get_retriever),
    generator = Depends(get_answer_generator)
):
    """
    Chat with documents using RAG (Retrieval-Augmented Generation).
    
    This endpoint combines document search with answer generation
    to provide conversational responses based on document content.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Chat request: query='{request.query[:50]}...'")
        
        # Default search options
        search_options = request.search_options or {}
        search_type = search_options.get("search_type", "hybrid")
        top_k = search_options.get("top_k", 10)
        
        # Perform search
        search_results = await retriever.search(
            query=request.query,
            document_ids=request.document_ids,
            search_type=search_type,
            top_k=top_k
        )
        
        search_time = search_results.get("search_time", 0.0)
        context_chunks = search_results.get("results", [])
        
        # Generate answer
        answer_data = await generator.generate_answer(
            query=request.query,
            context_chunks=context_chunks,
            chat_history=request.chat_history,
            generation_config=request.generation_config
        )
        
        generation_time = answer_data.get("generation_time", 0.0)
        total_time = time.time() - start_time
        
        response = ChatResponse(
            query=request.query,
            answer=answer_data.get("answer", ""),
            citations=[Citation(**citation) for citation in answer_data.get("citations", [])],
            confidence=answer_data.get("confidence", 0.0),
            search_results_count=len(context_chunks),
            total_time=total_time,
            search_time=search_time,
            generation_time=generation_time,
            error=answer_data.get("error")
        )
        
        logger.info(f"Chat completed in {total_time:.3f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )


@router.get("/suggest")
async def get_search_suggestions(
    query: str = Query(..., description="Partial query for suggestions", min_length=1),
    size: int = Query(5, description="Number of suggestions", ge=1, le=20),
    retriever = Depends(get_retriever)
):
    """
    Get search suggestions based on indexed content.
    
    This endpoint provides query suggestions to help users
    formulate better search queries.
    """
    try:
        suggestions = await retriever.sparse_retriever.suggest(query, size)
        
        return {
            "query": query,
            "suggestions": suggestions,
            "count": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"Suggestions failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Suggestions failed: {str(e)}"
        )


@router.get("/analyze")
async def analyze_query(
    query: str = Query(..., description="Query to analyze", min_length=1),
    retriever = Depends(get_retriever)
):
    """
    Analyze query using different text analyzers.
    
    This endpoint shows how the query is processed by different
    analyzers, which can be useful for debugging search issues.
    """
    try:
        analysis = await retriever.sparse_retriever.analyze_query(query)
        
        return {
            "query": query,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Query analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query analysis failed: {str(e)}"
        )


@router.get("/stats")
async def get_search_stats(
    retriever = Depends(get_retriever),
    generator = Depends(get_answer_generator)
):
    """
    Get search system statistics.
    
    This endpoint provides information about the search system
    status, performance metrics, and configuration.
    """
    try:
        retriever_stats = await retriever.get_retriever_stats()
        generator_stats = await generator.get_generator_stats()
        
        return {
            "retriever": retriever_stats,
            "generator": generator_stats,
            "configuration": {
                "hybrid_search_enabled": settings.HYBRID_SEARCH_ENABLED,
                "max_search_results": settings.MAX_SEARCH_RESULTS,
                "reranking_enabled": settings.RERANKING_ENABLED
            }
        }
        
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Stats retrieval failed: {str(e)}"
        )


@router.post("/explain")
async def explain_search_score(
    chunk_id: str = Query(..., description="Chunk ID to explain"),
    query: str = Query(..., description="Search query"),
    retriever = Depends(get_retriever)
):
    """
    Explain why a specific chunk got its search score.
    
    This endpoint provides detailed information about how
    the search score was calculated for a specific chunk.
    """
    try:
        explanation = await retriever.sparse_retriever.explain_score(query, chunk_id)
        
        return {
            "chunk_id": chunk_id,
            "query": query,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"Score explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Score explanation failed: {str(e)}"
        )