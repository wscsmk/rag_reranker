import asyncio
from openai import AsyncOpenAI
import os
from rag import reranker_results

api_key = os.getenv('key')
if api_key is None:
    print('key不存在')
    exit(0)

client = AsyncOpenAI(
    api_key=api_key,
    base_url='http://106.55.99.69:3000/v1'
)

JUDGE_MODEL = "deepseek/deepseek-v4-flash"      # 示例：轻量模型

CHAT_MODEL = "deepseek/deepseek-v4-pro"

class LegalKnowledgeChat:
    def __init__(self, system_prompt: str = "你是一个专业的法律助手。"):
        self.system_prompt = system_prompt
        # 只保存原始对话（不含检索到的知识）
        self.raw_history = []   # 元素为 {"role": "user"/"assistant", "content": str}

    async def _is_legal_query(self, query: str) -> bool:
        judge_messages = [
            {"role": "system", "content":
                "你是一个路由助手。判断用户当前问题是否属于「法律相关」问题。"
                "法律相关问题包括：咨询法律法规、合同条款、诉讼程序、权利义务、法律后果、司法案例、法律条款等。"
                "日常问候、闲聊、数学、天气、编程等技术问题不属于法律问题。"
                "只回答 YES 或 NO，不要输出其他内容。"}
        ]
        # 添加最近3轮历史（帮助理解指代）
        for msg in self.raw_history[-6:]:   # 最多6条（3轮）
            judge_messages.append(msg)
        # 添加当前问题
        judge_messages.append({"role": "user", "content": query})

        try:
            response = await client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=judge_messages,
                temperature=0,
                max_tokens=5
            )
            answer = response.choices[0].message.content.strip().upper()
            print(answer)
            return "YES" in answer
        except Exception as e:
            print(f"\n判断模型调用失败: {e}，默认检索知识库")
            return True   # 安全起见，失败时检索

    async def ask(self, user_input: str):
        print("正在分析问题类型...", end="", flush=True)
        need_retrieve = await self._is_legal_query(user_input)
        print(" 需要检索" if need_retrieve else " 无需检索")

        context_str = ""
        if need_retrieve:
            print("正在检索法律知识库...", end="", flush=True)
            reranked = await asyncio.to_thread(reranker_results, user_input)
            if reranked:
                top_chunks = []
                for (doc, _), rerank_score in reranked[:3]:
                    content = doc.page_content.strip()
                    source = doc.metadata.get('source', '未知来源')
                    top_chunks.append(f"[来源：{source}]\n{content}")
                context_str = "\n\n---\n\n".join(top_chunks)
                print(f" 召回 {len(top_chunks)} 条相关知识")
            else:
                print(" 未找到相关知识")
        else:
            print("💬 直接使用通用对话能力（不检索知识库）")


        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        messages.extend(self.raw_history)


        if context_str:
            augmented_user_msg = (
                f"请严格依据下面提供的法律知识片段回答用户问题。\n"
                f"如果知识片段不足以回答问题，请说“根据现有法律知识无法回答”。\n\n"
                f"【法律知识片段】\n{context_str}\n\n"
                f"【用户问题】\n{user_input}"
            )
        else:
            augmented_user_msg = user_input

        messages.append({"role": "user", "content": augmented_user_msg})


        stream = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
        )


        full_response = []
        print("助手: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response.append(content)
        print()


        self.raw_history.append({"role": "user", "content": user_input})
        self.raw_history.append({"role": "assistant", "content": "".join(full_response)})

async def async_input(prompt: str) -> str:

    print(prompt, end="", flush=True)
    return await asyncio.to_thread(input)

async def background_task():
    i = 0
    while True:
        await asyncio.sleep(200)
        i += 200
        print(f"\n[后台] 第{i}次心跳 - 还在努力工作...")

async def main():
    chat = LegalKnowledgeChat(system_prompt="你是一个严谨的法律助手，优先使用提供的法律知识。")
    task = asyncio.create_task(background_task())
    print("多轮对话开始（输入 quit 退出）\n")
    try:
        while True:
            user = await async_input("你: ")
            if user.lower() in ("quit", "exit"):
                break
            if not user.strip():
                continue
            await chat.ask(user)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print('正常退出')

if __name__ == "__main__":
    asyncio.run(main())