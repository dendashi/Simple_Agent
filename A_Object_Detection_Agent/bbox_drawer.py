# bbox_drawer.py
"""
根据 Qwen3-VL 输出的 JSON，在图片上画框。

约定：
- 首选使用相对坐标（0~1），再根据图像宽高转换为像素。
- 为了兼容旧输出，如果发现坐标中有 > 1.5 的值，则视为绝对像素。
"""

import json
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont


def _is_relative_coords(box: Dict[str, Any]) -> bool:
    """判断一组坐标更像相对坐标还是绝对像素。"""
    try:
        xs = [float(box[k]) for k in ("x1", "x2")]
        ys = [float(box[k]) for k in ("y1", "y2")]
    except Exception:
        return False

    max_val = max(xs + ys)
    return max_val <= 1.5


def draw_bboxes(
    image_path: str | Path,
    bbox_json_str: str,
    output_path: str | Path,
) -> Path:
    """
    根据模型输出的 JSON 在图片上画框，并保存新图片。

    返回: 输出图片的 Path
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    try:
        data = json.loads(bbox_json_str)
    except json.JSONDecodeError as e:
        print(f"[draw_bboxes] JSON 解析失败：{e}，直接复制原图。")
        img = Image.open(image_path).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path

    if not data:
        print("[draw_bboxes] 空框列表，直接复制原图。")
        img = Image.open(image_path).convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path

    if isinstance(data, dict):
        boxes = [data]
    else:
        boxes = list(data)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    font = None  # 用默认字体

    for box in boxes:
        try:
            label = str(box.get("label", ""))
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        except Exception as e:
            print(f"[draw_bboxes] 跳过异常 box {box}: {e}")
            continue

        if _is_relative_coords(box):
            X1 = int(x1 * W)
            Y1 = int(y1 * H)
            X2 = int(x2 * W)
            Y2 = int(y2 * H)
        else:
            X1 = int(x1)
            Y1 = int(y1)
            X2 = int(x2)
            Y2 = int(y2)

        X1 = max(0, min(X1, W - 1))
        X2 = max(0, min(X2, W - 1))
        Y1 = max(0, min(Y1, H - 1))
        Y2 = max(0, min(Y2, H - 1))

        if X2 <= X1 or Y2 <= Y1:
            print(f"[draw_bboxes] 无效框，被丢弃: {(X1, Y1, X2, Y2)}")
            continue

        draw.rectangle([X1, Y1, X2, Y2], outline="red", width=3)

        if label:
            text_pos = (X1 + 2, max(Y1 - 15, 0))
            draw.text(text_pos, label, fill="red", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"[draw_bboxes] 已保存带框图片到: {output_path}")
    return output_path
