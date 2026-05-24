"""文档解析模块 - 支持 TXT / DOCX / Markdown 多格式"""
import os
from typing import List, Dict, Callable

import chardet
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import KnowledgeSettings
from app.core.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """多格式文档加载 + 自动切分"""

    # 格式 → 加载方法映射
    _LOADERS: Dict[str, str] = {
        ".txt": "_load_txt",
        ".docx": "_load_docx",
        ".md": "_load_markdown",
    }

    def __init__(self, cfg: KnowledgeSettings):
        self.cfg = cfg
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.separators,
        )

    # ── 单格式加载 ─────────────────────────────────────

    @staticmethod
    def _detect_encoding(file_path: str) -> str:
        """自动检测文件编码"""
        with open(file_path, "rb") as f:
            raw = f.read()
        result = chardet.detect(raw)
        encoding = result.get("encoding", "utf-8")
        logger.debug("检测编码 %s → %s (置信度 %.2f)", file_path, encoding, result.get("confidence", 0))
        return encoding

    def _load_txt(self, file_path: str) -> List[Document]:
        encoding = self._detect_encoding(file_path)
        return TextLoader(file_path, encoding=encoding, autodetect_encoding=False).load()

    def _load_docx(self, file_path: str) -> List[Document]:
        return Docx2txtLoader(file_path).load()

    def _load_markdown(self, file_path: str) -> List[Document]:
        """加载 Markdown 文件，保留标题层级信息"""
        return UnstructuredMarkdownLoader(file_path).load()

    # ── 批量加载 ────────────────────────────────────────

    def load_all(self) -> List[Document]:
        """遍历知识库目录，加载所有支持格式的文档"""
        documents: List[Document] = []
        knowledge_dir = str(self.cfg.knowledge_dir)

        if not os.path.isdir(knowledge_dir):
            logger.warning("知识库目录不存在: %s", knowledge_dir)
            return documents

        logger.info("开始加载知识库目录: %s", knowledge_dir)
        for filename in sorted(os.listdir(knowledge_dir)):
            file_path = os.path.join(knowledge_dir, filename)
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(filename)[1].lower()
            method_name = self._LOADERS.get(ext)
            if method_name is None:
                if ext not in self.cfg.supported_extensions:
                    logger.debug("跳过不支持的格式: %s", filename)
                continue

            try:
                docs = getattr(self, method_name)(file_path)
                documents.extend(docs)
                logger.info("已加载 [%s]: %s (%d 片段)", ext, filename, len(docs))
            except Exception as exc:
                logger.error("加载失败 %s: %s", filename, exc)

        logger.info("共加载文档片段: %d", len(documents))
        return documents

    # ── 加载 + 切分 ────────────────────────────────────

    def load_and_split(self) -> List[Document]:
        """加载文档并切分为适合检索的文本块"""
        documents = self.load_all()
        if not documents:
            logger.warning("知识库为空，无文档可切分")
            return []

        chunks = self.splitter.split_documents(documents)
        logger.info("切分为 %d 个文本块 (chunk_size=%d, overlap=%d)",
                     len(chunks), self.cfg.chunk_size, self.cfg.chunk_overlap)
        return chunks