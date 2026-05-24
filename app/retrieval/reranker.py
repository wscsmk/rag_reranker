"""重排序模块 - 基于 CrossEncoder / Transformers 的二次精排"""
import torch
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document

from app.core.config import ModelSettings
from app.core.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """对初步检索结果做精排，返回按相关性降序的 (Document, score) 列表"""

    def __init__(self, cfg: ModelSettings):
        self.cfg = cfg
        self.device = cfg.device
        self.top_k = 3  # 默认返回数量，可被调用方覆盖

        logger.info("加载重排序模型: %s (device=%s)", cfg.reranker_model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.reranker_model_path), trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(cfg.reranker_model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

    @torch.no_grad()
    def _score_single(self, query: str, passage: str) -> float:
        """对单条 (query, passage) 计算相关性分数"""
        inputs = self.tokenizer(
            query, passage,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        logits = self.model(**inputs).logits

        if logits.shape[-1] == 1:
            return logits[0, 0].item()
        # 多分类 → 取正类概率
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()

    def rerank(
        self, query: str, docs: List[Document], top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """对文档列表做重排序，返回 top_k 个最相关结果"""
        k = top_k or self.top_k
        valid = [d for d in docs if d.page_content.strip()]
        if not valid:
            logger.warning("所有检索结果均为空文档，无法重排序")
            return []

        scored = [(doc, self._score_single(query, doc.page_content)) for doc in valid]
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info("重排序完成，取 Top %d / %d 条有效结果", k, len(valid))
        return scored[:k]