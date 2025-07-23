"""
Health check endpoints with trio support.
"""
import logging
from typing import Dict, Any

import trio
from fastapi import APIRouter

from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Comprehensive system health check."""
    
    start_time = trio.current_time()
    
    health_status = {
        "status": "healthy",
        "timestamp": start_time,
        "version": "1.0.0",
        "async_backend": "trio",
        "components": {}
    }
    
    # Import services
    from app.db.infinity_client import infinity_client
    from app.core.cache import cache_manager
    
    # Check database connectivity
    try:
        from app.db.session import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        health_status["components"]["database"] = {
            "status": "healthy",
            "type": "postgresql"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Infinity DB
    try:
        infinity_health = await infinity_client.health_check()
        health_status["components"]["vector_db"] = infinity_health
        if infinity_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Infinity DB health check failed: {e}")
        health_status["components"]["vector_db"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        redis_health = await cache_manager.health_check()
        health_status["components"]["cache"] = redis_health
        if redis_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["components"]["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Add performance metrics
    health_status["performance"] = {
        "response_time_ms": (trio.current_time() - start_time) * 1000,
        "trio_current_time": trio.current_time()
    }
    
    return health_status


@router.get("/components")
async def component_health() -> Dict[str, Any]:
    """Detailed component health status."""
    return {
        "database": {"status": "pending", "message": "Not implemented yet"},
        "vector_db": {"status": "pending", "message": "Not implemented yet"},
        "search_engine": {"status": "pending", "message": "Not implemented yet"},
        "cache": {"status": "pending", "message": "Not implemented yet"},
        "deepdoc_models": {"status": "pending", "message": "Not implemented yet"}
    }