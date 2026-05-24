"""向量库管理模块 - FAISS 索引构建 / 加载 / 重建"""
import os
import shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.core.config import ModelSettings, KnowledgeSettings
from app.core.logger import get_logger
from app.retrieval.document_loader import DocumentLoader

logger = get_logger(__name__)


class VectorStoreManager:
    """FAISS 向量库生命周期管理"""

    def __init__(self, model_cfg: ModelSettings, kb_cfg: KnowledgeSettings, loader: DocumentLoader):
        self.model_cfg = model_cfg
        self.kb_cfg = kb_cfg
        self.loader = loader
        self.embeddings = self._init_embeddings()

    # ── 嵌入模型 ────────────────────────────────────────

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        self._check_model(self.model_cfg.embedding_model_path, "嵌入模型")
        logger.info("加载嵌入模型: %s (device=%s)", self.model_cfg.embedding_model_path, self.model_cfg.device)
        return HuggingFaceEmbeddings(
            model_name=str(self.model_cfg.embedding_model_path),
            model_kwargs={"device": self.model_cfg.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @staticmethod
    def _check_model(path: Path, name: str) -> None:
        """校验模型目录存在且包含配置文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} 路径不存在: {path}\n请先运行 scripts/download_models.py 下载模型"
            )
        files = os.listdir(path)
        if not any(f.startswith("config") for f in files):
            raise FileNotFoundError(f"{name} 缺少 config.json: {path}\n目录内容: {files}")

    # ── 向量库加载 / 构建 ───────────────────────────────

    def _index_exists(self) -> bool:
        db_path = str(self.kb_cfg.vector_db_path)
        return all(
            os.path.exists(os.path.join(db_path, f)) for f in ("index.faiss", "index.pkl")
        )

    def load_or_build(self) -> FAISS:
        """加载已有索引，不存在则自动构建"""
        if self._index_exists():
            logger.info("加载已有向量库: %s", self.kb_cfg.vector_db_path)
            return FAISS.load_local(
                str(self.kb_cfg.vector_db_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self.rebuild()

    def rebuild(self) -> FAISS:
        """重建向量库（清理旧索引 → 切分文档 → 构建新索引）"""
        logger.info("构建新的向量库 ...")
        db_path = str(self.kb_cfg.vector_db_path)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            logger.info("已清理旧目录: %s", db_path)

        chunks = self.loader.load_and_split()
        if not chunks:
            raise ValueError(f"知识库为空: {self.kb_cfg.knowledge_dir}")

        vs = FAISS.from_documents(chunks, self.embeddings)
        vs.save_local(db_path)
        logger.info("向量库已保存: %s (共 %d 向量)", db_path, vs.index.ntotal)
        return vs