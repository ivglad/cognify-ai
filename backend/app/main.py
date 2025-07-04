import os
# Настройки gRPC для лучшей совместимости с uvloop
os.environ.setdefault("GRPC_POLL_STRATEGY", "poll")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR") 
os.environ.setdefault("GRPC_TRACE", "")

from fastapi import FastAPI, Request
from app.api.api import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.db.infinity_client import infinity_client
import time
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Инициализация централизованного Infinity клиента при старте
logger.info("Application startup: Infinity client initialized")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Smart search for your documents.",
    version="0.1.0",
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.monotonic()
    response = await call_next(request)
    end_time = time.monotonic()
    
    # Correctly log processing time for specific endpoints
    # Note: /chat expects a trailing slash, so we match both cases.
    if request.url.path == f"{settings.API_V1_STR}/documents/upload" or \
       request.url.path == f"{settings.API_V1_STR}/chat/" or \
       request.url.path == f"{settings.API_V1_STR}/chat":
        
        processing_time = (end_time - start_time) * 1000
        logger.info(f"Request {request.method} {request.url.path} completed in {processing_time:.2f}ms. Status code: {response.status_code}")
    
    return response

app.include_router(api_router, prefix=settings.API_V1_STR)