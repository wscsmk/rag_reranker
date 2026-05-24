import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document


class Reranker:
    def __init__(self, config):
        self.cfg = config
        self.device = config.device
        print("加载重排序模型 ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.reranker_model_path, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.reranker_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

    @torch.no_grad()
    def _score(self, query: str, passage: str) -> float:
        inputs = self.tokenizer(
            query, passage,
            return_tensors="pt", truncation=True, max_length=512, padding=True,
        ).to(self.device)
        logits = self.model(**inputs).logits
        if logits.shape[-1] == 1:
            return logits[0, 0].item()
        return torch.softmax(logits, dim=-1)[0, 1].item()

    def rerank(self, query: str, docs: List[Document], top_k: int = None) -> List[Tuple[Document, float]]:
        valid = [d for d in docs if d.page_content.strip()]
        if not valid:
            return []
        scored = [(doc, self._score(query, doc.page_content)) for doc in valid]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: top_k or self.cfg.top_k_rerank]
