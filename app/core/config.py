"""统一配置模块 - 支持环境变量覆盖，企业级配置管理"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


# ── 项目根目录 ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelSettings(BaseSettings):
    """模型路径配置 - Docker 构建时 /models/rag/ 和 /knowledge/ 为空目录"""

    embedding_model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "rag" / "acge_text_embedding",
        description="嵌入模型本地路径",
    )
    reranker_model_path: Path = Field(
        default=PROJECT_ROOT / "models" / "rag" / "Qwen3-Reranker-4B",
        description="重排序模型本地路径",
    )
    device: str = Field(
        default="cuda",
        description="推理设备 (cuda / cpu)",
    )

    model_config = {"env_prefix": "RAG_MODEL_"}


class KnowledgeSettings(BaseSettings):
    """知识库 & 向量库配置"""

    knowledge_dir: Path = Field(
        default=PROJECT_ROOT / "knowledge",
        description="法律文档目录",
    )
    vector_db_path: Path = Field(
        default=PROJECT_ROOT / "vector_db",
        description="FAISS 向量库存储路径",
    )

    chunk_size: int = Field(default=200, description="文本切分块大小")
    chunk_overlap: int = Field(default=30, description="切分重叠字符数")
    separators: List[str] = Field(
        default=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        description="文本切分分隔符优先级",
    )
    supported_extensions: List[str] = Field(
        default=[".txt", ".docx", ".md"],
        description="支持的文档格式",
    )

    top_k_retrieve: int = Field(default=5, description="初步检索召回数量")
    top_k_rerank: int = Field(default=3, description="重排序后返回数量")

    model_config = {"env_prefix": "RAG_KB_"}


class LLMSettings(BaseSettings):
    """LLM API 配置"""

    api_key: str = Field(
        default="",
        description="OpenAI-compatible API Key",
    )
    base_url: str = Field(
        default="http://106.55.99.69:3000/v1",
        description="API 基础 URL",
    )
    judge_model: str = Field(
        default="deepseek/deepseek-v4-flash",
        description="意图判断模型（轻量级）",
    )
    chat_model: str = Field(
        default="deepseek/deepseek-v4-pro",
        description="对话生成模型",
    )
    max_history_rounds: int = Field(
        default=6,
        description="携带到判断模型的最近对话条数",
    )
    system_prompt: str = Field(
        default="你是一个严谨的法律助手，优先使用提供的法律知识。",
        description="系统提示词",
    )

    model_config = {"env_prefix": "RAG_LLM_"}


class APISettings(BaseSettings):
    """FastAPI 服务配置"""

    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8000, description="服务监听端口")
    cors_origins: List[str] = Field(
        default=["*"],
        description="允许跨域来源",
    )

    model_config = {"env_prefix": "RAG_API_"}


class AppSettings(BaseSettings):
    """应用全局配置聚合"""

    model: ModelSettings = Field(default_factory=ModelSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)

    log_level: str = Field(default="INFO", description="日志级别")


# ── 单例 ────────────────────────────────────────────────
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """获取全局配置单例，首次调用时初始化"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings