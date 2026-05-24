from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from rag_engine import RAGEngine

app = FastAPI(title="RAG Search API")
engine: Optional[RAGEngine] = None


class SearchRequest(BaseModel):
    query: str
    top_k_retrieve: Optional[int] = None
    top_k_rerank: Optional[int] = None


class SearchResult(BaseModel):
    rank: int
    score: float
    content: str
    source: str


@app.on_event("startup")
def startup():
    global engine
    engine = RAGEngine()


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(400, "query 不能为空")
    return engine.search(req.query, req.top_k_retrieve, req.top_k_rerank)


@app.post("/rebuild")
def rebuild():
    engine.rebuild_index()
    return {"status": "ok"}
