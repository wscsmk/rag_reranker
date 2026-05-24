from typing import List, Dict, Any
from config import RAGConfig
from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from reranker import Reranker


class RAGEngine:
    """RAG 引擎统一入口，供 CLI / API / 其他项目调用"""

    def __init__(self, config: RAGConfig = None):
        self.cfg = config or RAGConfig()
        self.loader = DocumentLoader(self.cfg)
        self.vs_manager = VectorStoreManager(self.cfg, self.loader)
        self.vector_store = self.vs_manager.load_or_build()
        self.reranker = Reranker(self.cfg)

    def search(self, query: str, top_k_retrieve: int = None, top_k_rerank: int = None) -> List[Dict[str, Any]]:
        """检索 + 重排，返回标准化结果"""
        k1 = top_k_retrieve or self.cfg.top_k_retrieve
        k2 = top_k_rerank or self.cfg.top_k_rerank

        retrieved = self.vector_store.similarity_search_with_score(query, k=k1)
        docs = [doc for doc, _ in retrieved]
        reranked = self.reranker.rerank(query, docs, top_k=k2)

        return [
            {
                "rank": i,
                "score": float(score),
                "content": doc.page_content,
                "source": doc.metadata.get("source", "N/A"),
                "metadata": doc.metadata,
            }
            for i, (doc, score) in enumerate(reranked, 1)
        ]

    def rebuild_index(self):
        """对外暴露：增量更新知识库后重建索引"""
        self.vector_store = self.vs_manager.rebuild()
