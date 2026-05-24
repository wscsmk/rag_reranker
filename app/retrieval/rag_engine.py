"""RAG 引擎 - 统一检索 + 重排入口"""
from typing import List, Dict, Any, Optional

from app.core.config import get_settings, AppSettings
from app.core.logger import get_logger
from app.retrieval.document_loader import DocumentLoader
from app.retrieval.vector_store import VectorStoreManager
from app.retrieval.reranker import Reranker

logger = get_logger(__name__)


class RAGEngine:
    """RAG 检索引擎：嵌入 → 向量召回 → 重排序"""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()
        s = self.settings

        # 初始化各子模块
        self.loader = DocumentLoader(s.knowledge)
        self.vs_manager = VectorStoreManager(s.model, s.knowledge, self.loader)
        self.vector_store = self.vs_manager.load_or_build()
        self.reranker = Reranker(s.model)

        logger.info("RAG 引擎初始化完成")

    def search(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """检索 + 重排，返回标准化结果列表"""
        k1 = top_k_retrieve or self.settings.knowledge.top_k_retrieve
        k2 = top_k_rerank or self.settings.knowledge.top_k_rerank

        # 初步召回
        retrieved = self.vector_store.similarity_search_with_score(query, k=k1)
        docs = [doc for doc, _ in retrieved]
        logger.debug("初步召回 %d 条", len(docs))

        # 重排序
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

    async def async_search(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """异步检索（将同步 FAISS 操作放到线程池）"""
        import asyncio
        return await asyncio.to_thread(
            self.search, query, top_k_retrieve, top_k_rerank
        )

    def rebuild_index(self) -> None:
        """重建向量库索引"""
        self.vector_store = self.vs_manager.rebuild()
        logger.info("向量库索引重建完成")
