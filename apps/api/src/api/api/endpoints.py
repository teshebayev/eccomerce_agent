from fastapi import FastAPI, Request, APIRouter
from pydantic import BaseModel
from .models import RagRequest, RagResponse, RagUsedContext
from api.core.config import config
import logging
from api.agents.retrieval_generation import rag_pipeline_wrapper
from api.agents.graph import rag_agent_wrapper

from qdrant_client import QdrantClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def rag(
    request: Request,
    payload: RagRequest
) -> RagResponse:

    # answer = rag_pipeline_wrapper(payload.query)
    answer = rag_agent_wrapper(payload.query)
    return RagResponse(request_id=request.state.request_id, 
                       answer =  answer["answer"], 
                       used_context = [RagUsedContext(**item) for item in answer["used_context"]])

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])



