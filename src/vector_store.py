import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    def __init__(self, config, document_loader):
        self.cfg = config
        self.loader = document_loader
        self.embeddings = self._init_embeddings()

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        self._check_model(self.cfg.embedding_model_path, "嵌入模型")
        print("加载嵌入模型 ...")
        return HuggingFaceEmbeddings(
            model_name=self.cfg.embedding_model_path,
            model_kwargs={"device": self.cfg.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @staticmethod
    def _check_model(path: str, name: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} 路径不存在: {path}，请先运行 dow.py 下载")
        if not any(f.startswith("config") for f in os.listdir(path)):
            raise FileNotFoundError(f"{name} 缺少 config.json: {path}")

    def _exists(self) -> bool:
        return all(
            os.path.exists(os.path.join(self.cfg.vector_db_path, f))
            for f in ("index.faiss", "index.pkl")
        )

    def load_or_build(self) -> FAISS:
        if self._exists():
            print(f"加载已有向量库: {self.cfg.vector_db_path}")
            return FAISS.load_local(
                self.cfg.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self.rebuild()

    def rebuild(self) -> FAISS:
        print("构建新的向量库 ...")
        if os.path.exists(self.cfg.vector_db_path):
            shutil.rmtree(self.cfg.vector_db_path)

        chunks = self.loader.load_and_split()
        if not chunks:
            raise ValueError(f"知识库为空: {self.cfg.knowledge_dir}")

        vs = FAISS.from_documents(chunks, self.embeddings)
        vs.save_local(self.cfg.vector_db_path)
        print(f"向量库已保存: {self.cfg.vector_db_path}")
        return vs
