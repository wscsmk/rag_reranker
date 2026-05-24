import os
from modelscope import snapshot_download

MODEL_DIR = os.path.abspath("models/rag")
os.makedirs(MODEL_DIR, exist_ok=True)

# 嵌入模型：魔搭上的官方 ID
EMBEDDING_MODEL_ID = "BAAI/bge-small-zh-v1.5"
# 重排序模型
RERANKER_MODEL_ID = "BAAI/bge-reranker-base"


def download(model_id: str):
    target = os.path.join(MODEL_DIR, model_id.split("/")[-1])
    if os.path.exists(target) and os.listdir(target):
        print(f"✅ 已存在: {target}")
        return target

    print(f"⬇️  正在下载 {model_id} ...")
    local_path = snapshot_download(
        model_id=model_id,
        local_dir=target,   # 直接下到这个目录
    )
    print(f"✅ 下载完成: {local_path}")
    return local_path


if __name__ == "__main__":
    emb_path = download(EMBEDDING_MODEL_ID)
    rer_path = download(RERANKER_MODEL_ID)

    print("\n📁 模型最终路径：")
    print(f"  嵌入模型:   {emb_path}")
    print(f"  重排序模型: {rer_path}")
