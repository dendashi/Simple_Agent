from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:8b",  # 这里改成你在 `ollama list` 里看到的实际模型名
    base_url="http://127.0.0.1:11434",  # 明确指定本地服务
)

resp = llm.invoke("用一句话告诉我：这是一个测试。")
print(resp.content)
