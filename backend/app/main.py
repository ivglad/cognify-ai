"""
Main FastAPI application with trio async support.
Based on RAGFlow architecture with Yandex Cloud ML integration.
"""
import logging
import time
from contextlib import asynccontextmanager

import trio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with trio support."""
    logger.info("Starting RAGFlow implementation with trio async...")
    
    # Import services
    from app.db.session import init_db, close_db
    from app.db.infinity_client import infinity_client
    from app.core.cache import cache_manager
    
    # Initialize trio-based services
    try:
        # Initialize database
        await init_db()
        
        # Initialize Infinity DB
        await infinity_client.connect()
        
        # Initialize Infinity collections
        from app.db.infinity_schemas import create_infinity_collections
        await create_infinity_collections(infinity_client)
        
        # Initialize Redis cache
        await cache_manager.connect()
        
        logger.info("All services initialized successfully")
        logger.info("Application startup completed")
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    finally:
        # Cleanup services
        try:
            await cache_manager.disconnect()
            await infinity_client.disconnect()
            await close_db()
            logger.info("Application shutdown completed")
        except Exception as e:
            logger.error(f"Application shutdown error: {e}")


# Create FastAPI app with trio-compatible configuration
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="RAGFlow-based document processing and search system with trio async",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with processing time."""
    start_time = trio.current_time()
    
    response = await call_next(request)
    
    process_time = trio.current_time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.4f}s"
    )
    
    return response


# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": trio.current_time(),
        "version": "1.0.0",
        "async_backend": "trio"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Note: For production, use a trio-compatible ASGI server
    # or configure uvicorn to work with trio
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )