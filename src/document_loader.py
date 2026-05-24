import os
import chardet
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentLoader:
    LOADERS = {
        ".txt": "_load_txt",
        ".docx": "_load_docx",
    }

    def __init__(self, config):
        self.cfg = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )

    @staticmethod
    def _detect_encoding(file_path: str) -> str:
        with open(file_path, "rb") as f:
            return chardet.detect(f.read()).get("encoding") or "utf-8"

    def _load_txt(self, file_path: str) -> List[Document]:
        encoding = self._detect_encoding(file_path)
        return TextLoader(file_path, encoding=encoding, autodetect_encoding=False).load()

    def _load_docx(self, file_path: str) -> List[Document]:
        return Docx2txtLoader(file_path).load()

    def load_all(self) -> List[Document]:
        documents = []
        for filename in os.listdir(self.cfg.knowledge_dir):
            file_path = os.path.join(self.cfg.knowledge_dir, filename)
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(filename)[1].lower()
            method_name = self.LOADERS.get(ext)
            if not method_name:
                print(f"  跳过不支持的文件: {filename}")
                continue

            try:
                docs = getattr(self, method_name)(file_path)
                documents.extend(docs)
                print(f"  已加载 [{ext}]: {filename}")
            except Exception as e:
                print(f"  加载失败 {filename}: {e}")

        print(f"共加载文档片段: {len(documents)}")
        return documents

    def load_and_split(self) -> List[Document]:
        documents = self.load_all()
        chunks = self.splitter.split_documents(documents)
        print(f"切分成 {len(chunks)} 个文本块")
        return chunks
