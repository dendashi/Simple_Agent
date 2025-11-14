# vl_core.py
"""
统一封装 Qwen3-VL 的调用入口，方便后续扩展更多视觉任务。

当前支持的 task_type:
- "describe": 输出自然语言描述
- "bbox": 输出 JSON 格式的框 [{label, x1, y1, x2, y2}, ...]
"""

import base64
from enum import Enum
from pathlib import Path
from typing import Literal

import requests


# ===== 配置区域 =====

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
QWEN3_VL_MODEL = "qwen3-vl:4b"   # ← 和 `ollama list` 一致即可


class VisionTaskType(str, Enum):
    DESCRIBE = "describe"
    BBOX = "bbox"


def _build_prompt(instruction: str, task_type: VisionTaskType) -> str:
    """根据任务类型，构造对 Qwen3-VL 的提示词。"""
    if task_type == VisionTaskType.DESCRIBE:
        return (
            "你是一个图像理解助手。请根据给定图像和用户指令完成任务。\n"
            f"用户指令：{instruction}\n\n"
            "要求：\n"
            "1. 只用自然语言简要描述图像中与指令相关的内容。\n"
            "2. 不要输出 JSON 或代码块，直接用中文说明即可。"
        )

    if task_type == VisionTaskType.BBOX:
        return (
            "你是一个目标检测助手。现在给你一张图像，请根据用户指令框出目标。\n"
            f"用户指令：{instruction}\n\n"
            "请严格按照以下要求输出：\n"
            "1. 只输出一个 JSON 数组，不要有任何额外文字、前后缀说明或代码块标记。\n"
            "2. JSON 数组的每个元素为一个框，格式如下：\n"
            "   {\"label\": \"对象名称\", \"x1\": 左上角x, \"y1\": 左上角y, "
            "\"x2\": 右下角x, \"y2\": 右下角y}\n"
            "3. 所有坐标必须是相对坐标，已经归一化到 [0,1] 区间，"
            "   其中 0 表示最左/最上，1 表示最右/最下。\n"
            "4. 坐标请保留 4 位小数，例如 0.1234。\n"
            "5. 如果无法确定要框的目标，输出一个空数组 []。\n\n"
            "现在请只输出 JSON："
        )

    return instruction


def call_qwen3_vl(
    image_path: str | Path,
    instruction: str,
    task_type: Literal["describe", "bbox"] = "describe",
    *,
    timeout: int = 120,
) -> str:
    """
    调用本地 Ollama 中的 Qwen3-VL 模型，对图像执行指定任务。

    返回:
        describe: 模型的自然语言描述（str）
        bbox:     模型输出的 JSON 字符串（上层可再 json.loads）
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    img_bytes = image_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    vt = VisionTaskType(task_type)
    prompt = _build_prompt(instruction, vt)

    payload = {
        "model": QWEN3_VL_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,  # 关闭流式，返回单个 JSON
    }

    url = f"{OLLAMA_BASE_URL}/api/generate"
    resp = requests.post(url, json=payload, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama 请求失败: HTTP {resp.status_code}, body={resp.text[:300]!r}"
        )

    data = resp.json()
    text = data.get("response", "")
    return text
