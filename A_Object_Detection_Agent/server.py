# server.py
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_core import run_vision_agent


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
INDEX_FILE = BASE_DIR / "index.html"

STATIC_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail=f"index.html not found at {INDEX_FILE}")
    return FileResponse(str(INDEX_FILE))


class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None  # 例如："/static/uploads/xxx.jpg"


class ChatResponse(BaseModel):
    reply: str
    image_url: Optional[str] = None  # 若有画框结果，就返回新的图片 URL


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片接口：保存到 static/uploads 下，返回可访问的 URL。
    """
    suffix = Path(file.filename).suffix or ".jpg"
    # 简单处理：用原文件名（你可以改成 uuid 或时间戳）
    out_path = UPLOAD_DIR / file.filename
    content = await file.read()
    out_path.write_bytes(content)

    url_path = f"/static/uploads/{file.filename}"
    return {"image_url": url_path}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.image_url:
        return ChatResponse(reply="请先在左侧上传一张图片，再输入任务指令。", image_url=None)

    # 从 image_url 推出本地路径
    if not req.image_url.startswith("/static/"):
        raise HTTPException(status_code=400, detail="非法的 image_url")

    local_path = BASE_DIR / req.image_url.lstrip("/")
    if not local_path.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {local_path}")

    agent_result = run_vision_agent(str(local_path), req.message)

    reply = agent_result["reply"]
    result_image_path = agent_result.get("result_image_path")

    # 如果有新的结果图片，就把本地路径转成 URL
    result_image_url = None
    if result_image_path:
        result_image_path = Path(result_image_path)
        try:
            rel = result_image_path.relative_to(BASE_DIR)
        except ValueError:
            # 不在项目目录内，就不返回 URL
            result_image_url = None
        else:
            result_image_url = "/" + str(rel).replace("\\", "/")

    return ChatResponse(reply=reply, image_url=result_image_url)
