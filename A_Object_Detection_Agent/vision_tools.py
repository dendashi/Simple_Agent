# vision_tools.py
"""
基于 Qwen3-VL 的视觉工具：
1）描述图像内容
2）根据模型输出的 JSON 在图像上画框
"""

from pathlib import Path

from vl_core import call_qwen3_vl
from bbox_drawer import draw_bboxes


def vision_describe(image_path: str, instruction: str) -> str:
    """
    对给定路径的图像按照指令进行自然语言描述。
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return f"找不到指定的图片路径: {img_path}"

    text = call_qwen3_vl(
        image_path=img_path,
        instruction=instruction,
        task_type="describe",
    )
    return text


def vision_bbox_and_draw(image_path: str, instruction: str) -> tuple[str, str]:
    """
    对给定路径的图像按照指令框出目标并画框。
    返回:
        (描述字符串, 输出图片的本地路径字符串)
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return f"找不到指定的图片路径: {img_path}", ""

    bbox_json_str = call_qwen3_vl(
        image_path=img_path,
        instruction=instruction,
        task_type="bbox",
    )

    stem = img_path.stem
    suffix = img_path.suffix or ".jpg"
    out_path = img_path.with_name(f"{stem}_bbox{suffix}")

    draw_bboxes(img_path, bbox_json_str, out_path)

    desc = f"已根据指令在图像上画框，输出图片路径为: {out_path}"
    return desc, str(out_path)