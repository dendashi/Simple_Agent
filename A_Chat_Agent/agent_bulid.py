# server.py
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# ===== 0. 路径处理：用当前文件所在目录作为根目录 =====
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
STATIC_DIR = BASE_DIR / "static"

# 如果 static 目录不存在，就自动创建一个空的
STATIC_DIR.mkdir(exist_ok=True)


# ===== 1. 工具定义 =====
@tool
def calculator(expression: str) -> str:
    """计算一个简单的数学表达式，例如: (1+2)*3"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果是 {result}"
    except Exception as e:
        return f"表达式有问题：{e}"


@tool
def todo_summarizer(items: List[str]) -> str:
    """把待办事项列表压缩成一句总结。"""
    joined = "；".join(items)
    return f"你今天的待办事项有：{joined}。建议先完成最重要和最紧急的两项。"


tools = [calculator, todo_summarizer]


# ===== 2. LLM + Agent（带短期记忆） =====
def build_llm():
    return ChatOllama(
        model="qwen3:8b",  # ← 换成你 Ollama 里实际的模型名
        temperature=0.2,
    )


checkpointer = InMemorySaver()
llm = build_llm()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一个像微信好友一样的中文智能体，会记住用户之前说过的话。\n"
        "当用户说“算一下/计算”时，可以调用 calculator 工具；\n"
        "当用户说“待办/今天要做什么”时，可以调用 todo_summarizer 工具。\n"
        "回答要自然、简洁。"
    ),
    checkpointer=checkpointer,
)


# ===== 3. FastAPI 应用 =====
app = FastAPI()

# 静态目录挂载（虽然现在暂时没用到，可以先留着）
#app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """返回聊天界面的 HTML 文件"""
    if not INDEX_FILE.exists():
        # 如果还是找不到，就直接报 500，提示完整路径，方便你对照
        raise HTTPException(
            status_code=500,
            detail=f"index.html not found at: {INDEX_FILE}",
        )
    return FileResponse(str(INDEX_FILE))


# ===== 4. /chat 接口 =====
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    state = agent.invoke(
        {"messages": [{"role": "user", "content": req.message}]},
        config=config,
    )

    reply = state["messages"][-1].content
    return ChatResponse(reply=reply)
