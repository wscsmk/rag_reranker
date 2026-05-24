import os
from dataclasses import dataclass, field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class RAGConfig:
    embedding_model_path: str = os.path.join(BASE_DIR, "../models", "rag", "acge_text_embedding")
    reranker_model_path: str = os.path.join(BASE_DIR, "../models", "rag", "Qwen3-Reranker-4B")
    knowledge_dir: str = os.path.join(BASE_DIR, "../knowledge")
    vector_db_path: str = os.path.join(BASE_DIR, "../vector_db")

    device: str = "cuda"
    top_k_retrieve: int = 5
    top_k_rerank: int = 3

    chunk_size: int = 200
    chunk_overlap: int = 30
    separators: list = field(default_factory=lambda: ["\n\n", "\n", "。", "！", "？", "；", " ", ""])
