# agent_core.py
"""
视觉 Agent 调度层：
- 根据用户指令决定调用“描述工具”还是“画框工具”
- 用文本 LLM（ChatOllama）把结果组织成自然语言回复
"""

from typing import Literal

from langchain_ollama import ChatOllama

from vision_tools import vision_describe, vision_bbox_and_draw


def build_text_llm():
    """构建处理文本回复的 LLM（非 VL）"""
    return ChatOllama(
        model="qwen3:8b",   # ← 用你实际的文本模型名
        temperature=0.2,
    )


def decide_task_type(message: str) -> Literal["describe", "bbox"]:
    """
    非常简单的规则：
    - 如果包含“框/圈/标出/画出/bbox”，就当作框选任务
    - 否则默认当描述任务
    """
    keywords = ["框", "圈", "标出", "画出", "bbox"]
    if any(k in message for k in keywords):
        return "bbox"
    return "describe"


def run_vision_agent(
    image_path: str,
    user_message: str,
    llm: ChatOllama | None = None,
) -> dict:
    """
    统一对外接口：
    输入：图像路径 + 用户指令
    输出：{
        "reply": 文本回复,
        "result_image_path": 可选，画框后图片路径（若有）
    }
    """
    if llm is None:
        llm = build_text_llm()

    task_type = decide_task_type(user_message)

    # 1. 调视觉工具
    if task_type == "describe":
        raw_result = vision_describe(image_path, user_message)
        result_image_path = None
    else:
        raw_result, out_path = vision_bbox_and_draw(image_path, user_message)
        result_image_path = out_path or None

    # 2. 用文本 LLM 组织回复
    prompt = (
        "你是一个多模态助手，已经通过视觉工具分析了用户提供的图像。\n"
        f"用户的原始指令是：{user_message}\n"
        f"视觉工具返回的信息是：{raw_result}\n\n"
        "请用简洁自然的中文回复用户：\n"
        "- 如果是描述任务，请帮用户用一两句话总结图像内容。\n"
        "- 如果是框选任务，请告诉用户已经在图像上画好了框，并简单说明框住了什么。\n"
        "不需要提到“视觉工具”这几个字。"
    )

    llm_res = llm.invoke(prompt)
    reply_text = getattr(llm_res, "content", str(llm_res))

    return {
        "reply": reply_text,
        "result_image_path": result_image_path,
    }