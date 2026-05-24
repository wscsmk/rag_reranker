"""统一日志模块 - 企业级结构化日志"""
import logging
import sys
from pathlib import Path

from app.core.config import get_settings


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging() -> None:
    """初始化全局日志配置，只执行一次"""
    global _initialized
    if _initialized:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # 控制台 handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))

    # 根 logger
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)

    # 第三方库降噪
    for noisy in ("urllib3", "httpx", "httpcore", "transformers", "torch", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger，自动初始化"""
    setup_logging()
    return logging.getLogger(name)
