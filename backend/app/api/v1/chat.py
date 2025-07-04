from fastapi import APIRouter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.search_service import search_service

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def handle_chat_request(request: ChatRequest):
    """
    Handles a user's chat query, performs a search, and returns an answer
    along with the updated conversation history.
    """
    answer, history = await search_service.search(request)
    return ChatResponse(answer=answer, sources=[], history=history) 