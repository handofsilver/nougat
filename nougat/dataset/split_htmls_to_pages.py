"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import re
import json
import htmlmin
import argparse
import multiprocessing
import logging
import pypdf
from typing import Tuple, List
from pathlib import Path
from bs4 import BeautifulSoup
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm
from nougat.dataset.rasterize import rasterize_paper
from nougat.dataset.split_md_to_pages import split_markdown
from nougat.dataset.parser.markdown import format_document
from nougat.dataset.parser.html2md import parse_latexml
from nougat.dataset.pdffigures import call_pdffigures
from nougat.dataset.patches.inject_coords_to_mmd import inject_coordinates


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def figure_label_to_name(label_str: str) -> str:
    """
    如果 [FIGURE:S5.F7] -> '7' 的简单解析，比如只取结尾数字。
    如果你的 figure_info 里 'name':'7'，就这样匹配。
    若不需要解析，直接 return label_str
    """
    m = re.search(r"(\d+)$", label_str)
    return m.group(1) if m else label_str


def remove_invalid_figures(mmd_text: str, page_idx: int, figure_map: dict) -> str:
    """
    mmd_text: 页面文本(含 [FIGURE:xxx]... )
    page_idx: 0-based 页索引
    figure_map: {page_num(1-based): set-of-names}
                e.g. {5:{'1','2'},6:{'16'},...}
    """
    pattern = re.compile(r"\[FIGURE:(.*?)\](.*?)\[ENDFIGURE\]", re.S)

    valid_names = figure_map.get(page_idx + 1, set())

    def _replacer(match):
        label_str = match.group(1).strip()  # e.g. "S5.F7"
        label_name = figure_label_to_name(label_str)
        if label_name in valid_names:
            return match.group(0)  # 保留整块
        else:
            return ""

    # return mmd_text
    return pattern.sub(_replacer, mmd_text)


def build_page_to_figs(fig_info):
    page_map = {}
    for fig in fig_info:
        page = fig["page"] + 1  # PDF页 (1-based)
        name = fig.get("name", "").strip()
        if page not in page_map:
            page_map[page] = set()
        page_map[page].add(name)
    return page_map


def process_paper(
    fname: str, pdf_file: Path, html_file: Path, json_file: Path, args: argparse.Namespace
) -> Tuple[int, int]:
    """
    修改后的处理流程，适配新的 split_md_to_pages API
    """
    # 跳过已处理的文件
    outpath: Path = args.out / fname
    if outpath.exists() and not args.recompute:
        logger.info(f"{fname} already processed.")
        return 0, 0

    try:
        # 读取PDF
        pdf = pypdf.PdfReader(pdf_file)
        total_pages = len(pdf.pages)
        outpath: Path = args.out / fname

        # 解析HTML为markdown
        html = BeautifulSoup(
            htmlmin.minify(
                open(html_file, "r", encoding="utf-8").read().replace("\xa0", " "),
                remove_all_empty_space=True,
            ),
            features="html.parser",
        )
        doc = parse_latexml(html)
        if doc is None:
            return total_pages, 0

        # 获取图表信息（新API需要figure_info字典）
        figure_info = {}
        # 创建图表信息的JSON文件
        if json_file is None or not json_file.exists():
            # 调用 pdffigures 获取图表信息
            json_file = Path(call_pdffigures(pdf_file, args.figure))

        if json_file and json_file.exists():
            # 读取图表信息
            figure_info = json.load(open(json_file, "r", encoding="utf-8"))
        else:
            logger.error(f"No figure info found for {fname}")
            return total_pages, 0

        # 转换为markdown
        mmd_text, _ = format_document(doc, keep_refs=True)
        # 注入坐标
        mmd_text = inject_coordinates(
            mmd_text=mmd_text,
            fig_info=figure_info["figures"],
            page_width=pdf.pages[0].mediabox.width,
            page_height=pdf.pages[0].mediabox.height,
        )

        # 保存原始的markdown文本（可选）
        if args.markdown:
            mmd_file = args.markdown / f"{fname}.mmd"
            with open(mmd_file, "w", encoding="utf-8") as f:
                f.write(mmd_text)
            logger.info(f"Markdown saved to {mmd_file}")

        # 调用新的分页函数（返回格式：pages, coinside_pages, bad_pages）
        doc_text_by_pages, page_spans, coinside_pages, bad_pages = split_markdown(
            mmd_text, pdf=pdf, figure_info=figure_info  # 传入完整的figure_info字典
        )

        # 保存分页结果
        os.makedirs(outpath, exist_ok=True)
        recognized_indices = []
        for i, content in enumerate(doc_text_by_pages):
            if content.strip():
                # 保存为页码命名的文件 (01.mmd, 02.mmd...)
                # content = remove_invalid_figures(content, i, figure_info)
                with open(outpath / f"{i+1:02d}.mmd", "w", encoding="utf-8") as f:
                    f.write(content)
                recognized_indices.append(i)

        # 生成页面图像（仅保存有内容的页）
        rasterize_paper(pdf_file, outpath, dpi=args.dpi, pages=recognized_indices)

        return total_pages, len(recognized_indices)

    except Exception as e:
        logger.error(f"Error processing {fname}: {str(e)}")
        return total_pages, 0


def process_htmls(args):
    """
    主要功能：将HTML文件转换为按页分割的markdown文件和对应的页面图像

    输入：
    - args.html: HTML文件目录，包含*.html文件
    - args.pdfs: 对应的PDF文件目录
    - args.figure: 包含每篇论文图表信息的JSON文件目录

    输出目录结构：
    args.out/
    ├── paper1/              # 每篇论文一个目录
    │   ├── meta.json       # 论文元数据
    │   ├── 01.mmd         # 第1页的markdown内容
    │   ├── 01.png         # 第1页的图像
    │   ├── 01_OCR.txt     # (可选)第1页的OCR文本
    │   ├── 02.mmd         # 第2页的markdown内容
    │   ├── 02.png         # 第2页的图像
    │   └── ...
    └── paper2/
        └── ...

    args.figure/
    └── paper1.json         # 论文中图表的位置和内容信息
        {
          "figures": [
            {
              "page": 1,
              "regionBoundary": [x1, y1, x2, y2],
              "caption": "Figure 1: ...",
              "type": "Figure"
            },
            ...
          ]
        }
    """
    # 验证输入目录是否存在
    for input_dir in (args.pdfs, args.html):
        if not input_dir.exists() and not input_dir.is_dir():
            logger.error("%s does not exist or is no dir.", input_dir)
            return

    # 获取所有HTML文件路径
    htmls: List[Path] = args.html.rglob("*.html")

    # 创建输出目录
    args.out.mkdir(exist_ok=True)
    if args.markdown:
        args.markdown.mkdir(exist_ok=True)

    # 使用进程池并行处理文件
    with ProcessPool(max_workers=args.workers) as pool:
        total_pages, total_pages_extracted = 0, 0
        tasks = {}

        # 为每个HTML文件创建处理任务
        for j, html_file in enumerate(htmls):
            fname = html_file.stem
            # 查找对应的PDF文件
            pdf_file = args.pdfs / (fname + ".pdf")
            if not pdf_file.exists():
                logger.info("%s pdf could not be found.", fname)
                continue

            # 设置对应的图表信息JSON文件（尚未创建）
            json_file = args.figure / (fname + ".json")

            # 调度任务到进程池，设置超时时间
            tasks[fname] = pool.schedule(
                process_paper,
                args=[fname, pdf_file, html_file, json_file, args],
                timeout=args.timeout,
            )

        # 收集处理结果
        for fname in tqdm(tasks):
            try:
                res = tasks[fname].result()
                if res is None:
                    logger.info("%s is faulty", fname)
                    continue

                # 统计处理结果
                num_pages, num_recognized_pages = res
                total_pages += num_pages
                total_pages_extracted += num_recognized_pages

                # 记录每个文件的处理情况
                logger.info(
                    "%s: %i/%i pages recognized. Percentage: %.2f%%",
                    fname,
                    num_recognized_pages,
                    num_pages,
                    (100 * num_recognized_pages / max(1, num_pages)),
                )
            except TimeoutError:
                logger.info("%s timed out", fname)

    # 输出总体处理统计
    if total_pages > 0:
        logger.info(
            "In total: %i/%i pages recognized. Percentage: %.2f%%",
            total_pages_extracted,
            total_pages,
            (100 * total_pages_extracted / max(1, total_pages)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, help="HTML files", required=True)
    parser.add_argument("--pdfs", type=Path, help="PDF files", required=True)
    parser.add_argument("--out", type=Path, help="Output dir", required=True)
    parser.add_argument("--recompute", action="store_true", help="recompute all splits")
    parser.add_argument("--markdown", type=Path, help="Markdown output dir", default=None)
    parser.add_argument("--figure", type=Path, help="Figure info JSON dir")
    parser.add_argument(
        "--workers", type=int, default=multiprocessing.cpu_count(), help="How many processes to use"
    )
    parser.add_argument(
        "--dpi", type=int, default=96, help="What resolution the pages will be saved at"
    )
    parser.add_argument("--timeout", type=float, default=120, help="max time per paper in seconds")
    parser.add_argument(
        "--tesseract", action="store_true", help="Tesseract OCR prediction for each page"
    )
    args = parser.parse_args()
    print(args)
    process_htmls(args)
