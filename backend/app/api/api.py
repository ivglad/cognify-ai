"""
Main API router for RAGFlow implementation.
"""
from fastapi import APIRouter

from app.api.v1 import documents, search, health

api_router = APIRouter()

# Include all API routes
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(search.router, prefix="/search", tags=["search"])