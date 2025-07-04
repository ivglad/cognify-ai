from fastapi import APIRouter

from app.api.v1 import documents, chat
 
api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"]) 