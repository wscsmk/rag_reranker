import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from rag_engine import RAGEngine


def main():
    engine = RAGEngine()
    print("\n" + "=" * 60)
    print("RAG 检索系统已就绪！输入 'q' 退出")
    print("=" * 60)

    while True:
        query = input("\n🔍 请输入查询关键词: ").strip()
        if query.lower() in ("q", "quit", "exit"):
            print("退出")
            break
        if not query:
            continue

        results = engine.search(query)
        print(f"\n--- 重排序后 Top {len(results)} ---")
        for r in results:
            print(f"\n[{r['rank']}] score={r['score']:.4f}")
            print(f"内容: {r['content']}")
            print(f"来源: {r['source']}")


if __name__ == "__main__":
    main()
