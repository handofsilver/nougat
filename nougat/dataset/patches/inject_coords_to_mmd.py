import re
import json
from pathlib import Path
from typing import List, Dict


def is_caption_below(caption_boundary: Dict[str, float], region_boundary: Dict[str, float]) -> bool:
    """
    判断标题是否在图片下方
    caption_boundary: 标题边界
    region_boundary: 图片边界
    """
    # 计算标题的中点y坐标
    caption_mid_y = (caption_boundary["y1"] + caption_boundary["y2"]) / 2
    # 计算图片的中点y坐标
    image_mid_y = (region_boundary["y1"] + region_boundary["y2"]) / 2
    # 判断标题是否在图片下方
    return caption_mid_y > image_mid_y


def normalize_coords(coords: List[float], page_width: float, page_height: float) -> List[float]:
    """
    将坐标归一化到 [0, 1] 范围
    coords: 坐标列表 [x1, y1, x2, y2]
    page_width: 页面宽度
    page_height: 页面高度
    """
    x1, y1, x2, y2 = coords
    return [
        round(x1 / page_width, 4),
        round(1 - y2 / page_height, 4),
        round(x2 / page_width, 4),
        round(1 - y1 / page_height, 4),
    ]


def inject_coordinates(
    mmd_text: str, fig_info: List[dict], page_width: float = 595.0, page_height: float = 842.0
) -> str:
    """
    Inject coordinates into mmd_text according to fig_info.
    mmd_text: 页面文本(含 [FIGURE:xxx]... )
    fig_info: [{page, name, coords}, ...]
    """

    # 提取所有FIGURE标签
    pattern = re.compile(
        r"\[FIGURE:(.*?)\.F(.*?)\](.*?)\[END_FIGURE\]", re.S
    )  # [FIGURE:S3.F1]... [END_FIGURE]
    matches = pattern.findall(mmd_text)
    figure_map = {m[1]: m[2] for m in matches}
    # print(f"figure_map: {figure_map}")

    # 解析坐标
    for fig in fig_info:
        if fig["figType"] != "Figure":
            continue

        # 提取坐标，页码和标题边界
        page = fig["page"] + 1
        caption_boundary = fig["captionBoundary"]
        fig_name = fig["name"]
        coords = fig["regionBoundary"]

        # 根据fig_name获取[FIGURE:xxx]的内容
        title_str = figure_map.get(fig_name, "").strip()
        if not title_str:
            print(f"⚠️ 未找到匹配项: {fig_name}")
            raise ValueError(f"未找到匹配项: {fig_name}")

        print(f"✅ 匹配成功: {fig_name}", title_str)

        # 归一化坐标：将坐标归一化到 [0, 1] 范围
        normalized_coords = normalize_coords(
            [coords["x1"], coords["y1"], coords["x2"], coords["y2"]], page_width, page_height
        )

        coords_str = f"[FIGURE_COORDS](x1={normalized_coords[0]}, y1={normalized_coords[1]}, x2={normalized_coords[2]}, y2={normalized_coords[3]})[END_FIGURE_COORDS]"

        # 判断标题在图片上方还是下方
        if is_caption_below(caption_boundary, coords):
            # 标题在图片下方，则将图片坐标添加到标题之前
            figure_map[fig_name] = figure_map[fig_name].replace(
                title_str, f"{coords_str}\n{title_str}\n"
            )
        else:
            # 标题在图片上方，则将图片坐标添加到标题之后
            figure_map[fig_name] = figure_map[fig_name].replace(
                title_str, f"{title_str}\n{coords_str}\n"
            )

    # 替换原始文本中的[FIGURE:xxx]标签
    result = pattern.sub(
        lambda m: f"[FIGURE:{m.group(1)}.F{m.group(2)}]{figure_map[m.group(2)]}[END_FIGURE]",
        mmd_text,
    )

    return result
