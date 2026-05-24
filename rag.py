import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "models", "rag", "bge-small-zh-v1.5")
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "rag", "bge-reranker-base")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
TOP_K_RETRIEVE = 5
TOP_K_RERANK = 3

def check_model(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} 路径不存在: {path}\n"
            f"得先下载模型，dow.py"
        )
    files = os.listdir(path)
    if not any(f.startswith("config") for f in files):
        raise FileNotFoundError(
            f"{name} 目录下找不到 config.json: {path}\n"
            f"目录内容: {files}\n"
        )
    print(f"{name} 路径正常: {path}")


# def load_and_split_documents():
#     print(f"加载知识库目录: {KNOWLEDGE_DIR}")
#     loader = DirectoryLoader(
#         KNOWLEDGE_DIR,
#         glob="**/*.txt",
#         loader_cls=TextLoader,
#         loader_kwargs={"encoding": "utf-8"},
#     )
#     docs = loader.load()
#     print(f"共加载 {len(docs)} 个文件")
#
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=200,
#         chunk_overlap=30,
#         separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
#     )
#     chunks = splitter.split_documents(docs)
#     print(f"切分成 {len(chunks)} 个文本块")
#     return chunks

from langchain_community.document_loaders import TextLoader, Docx2txtLoader


def load_and_split_documents():
    print(f"加载知识库目录: {KNOWLEDGE_DIR}")
    documents = []

    # 遍历目录下所有文件
    for file in os.listdir(KNOWLEDGE_DIR):
        file_path = os.path.join(KNOWLEDGE_DIR, file)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file)[1].lower()
        try:
            if ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)
                print(f"  已加载 TXT: {file}")
            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"  已加载 DOCX: {file}")
            else:
                print(f"  跳过不支持的文件: {file}")
        except Exception as e:
            print(f"  加载失败 {file}: {e}")

    print(f"共加载 {len(documents)} 个文件（片段）")

    # 文本切分器不变
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"切分成 {len(chunks)} 个文本块")
    return chunks


def build_or_load_vector_store(embeddings):
    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    pkl_file = os.path.join(VECTOR_DB_PATH, "index.pkl")
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        print(f"加载已有向量库: {VECTOR_DB_PATH}")
        return FAISS.load_local(
            VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True
        )
    print("构建新的向量库 ...")
    if os.path.exists(VECTOR_DB_PATH):
        import shutil
        shutil.rmtree(VECTOR_DB_PATH)
        print(f"已清理无效目录: {VECTOR_DB_PATH}")
    chunks = load_and_split_documents()
    if not chunks:
        raise ValueError(
            f"知识库为空！请检查 {KNOWLEDGE_DIR} 目录下是否有 .txt 文件"
        )

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(VECTOR_DB_PATH)
    print(f"已保存到: {VECTOR_DB_PATH}")
    return vs

def reranker_results(query):
    check_model(EMBEDDING_MODEL_PATH, "嵌入模型")
    check_model(RERANKER_MODEL_PATH, "重排序模型")
    print("\n加载嵌入模型 ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = build_or_load_vector_store(embeddings)
    print("加载重排序模型 ...")
    reranker = CrossEncoder(RERANKER_MODEL_PATH, device="cpu")
    results = vector_store.similarity_search_with_score(query, k=TOP_K_RETRIEVE)
    # print(f"\n--- 初步召回 Top {len(results)} （分数越小越相似）---")
    # for i, (doc, score) in enumerate(results, 1):
    #     print(f"[{i}] score={score:.4f}  {doc.page_content}")
    pairs = [[query, doc.page_content] for doc, _ in results]
    rerank_scores = reranker.predict(pairs).tolist()
    reranked = sorted(
        zip(results, rerank_scores),
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_K_RERANK]

    return reranked

def main():
    print("\n" + "=" * 60)
    print("RAG 检索系统已就绪！输入 'q' 退出")
    print("=" * 60)

    while True:
        query = input("\n🔍 请输入查询关键词: ").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("👋 再见")
            break
        if not query:
            continue
        reranked = reranker_results(query)
        print(f"\n--- 重排序后 Top {TOP_K_RERANK} （分数越高越相关）---")
        for i, ((doc, _), rscore) in enumerate(reranked, 1):
            print(f"[{i}] rerank_score={rscore:.4f}")
            print(f"内容: {doc.page_content}")
            print(f"来源: {doc.metadata.get('source', 'N/A')}")
if __name__ == "__main__":
    main()
