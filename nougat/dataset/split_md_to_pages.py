"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

"""

import os
import re
import json
import pypdf
import argparse
import jieba
import numpy as np

from time import time
from typing import Dict, List, Tuple
from nougat.dataset.split_utils.markdown_parser import parse_markdown_lines
from nougat.dataset.split_utils.line_matcher import filter_and_match_lines
from nougat.dataset.split_utils.page_boundary import locate_page_boundaries, get_span_of_pages
from nougat.dataset.split_utils.rasterize import rasterize_paper


def printerr(*args, **kwargs):
    # uncomment for debugging:
    # print(*args, **kwargs)
    return

def split_markdown(
    doc: str, pdf: pypdf.PdfReader, figure_info: List[Dict], debug=False
) -> Tuple[List[str], Dict]:
    """
    Split a PDF document into Markdown paragraphs.

    Args:
        doc (str): latex 转 html 再转 markdown 后 md 文本内容.
        pdf (pypdf.PdfReader): 用 pypdf 读取 pdf 文件的结果.
        figure_info (Optional[List[Dict]]): 图表信息，每个字典指定一个图表的信息，包括标题、页码和边界框.

    Returns:
        doc_lines_by_pages: 每一页的行数组
        coinside_pages: 两页重合的页码对
        bad_pages: 两页冲突的页码对
    """
    # 解析markdown文本
    doc_lines = parse_markdown_lines(doc)

    # 过滤和匹配行
    time_start = time()
    valid_lines_of_pages = filter_and_match_lines(pdf, doc_lines, debug=debug)
    time_end = time()
    printerr(376, time_end - time_start)

    # 定位页边界
    page_start_positions, page_end_positions = locate_page_boundaries(
        valid_lines_of_pages, doc_lines, debug=debug
    )

    # 获取分割位置
    page_spans, coinside_pages, bad_pages = get_span_of_pages(
        doc_lines, page_start_positions, page_end_positions
    )

    # 根据分割位置拆分文档
    doc_text_by_pages = []
    for start, end in page_spans:
        if start > 0:
            doc_text_by_pages.append("\n".join(doc_lines[start : end + 1]))
        else:
            doc_text_by_pages.append("")

    printerr(516, len(doc_text_by_pages))
    return doc_text_by_pages, page_spans, coinside_pages, bad_pages


def use_split_markdown():
    """
    主函数，处理命令行参数并执行拆分过程。

    流程：
    1. 解析命令行参数（Markdown文件、PDF文件、输出目录、图表信息等）
    2. 读取Markdown和PDF文件
    3. 调用split_markdown函数进行拆分
    4. 将拆分结果保存到指定目录
    5. 调用rasterize_paper函数将PDF页面转换为图像
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=str, help="Markdown file", required=False)
    parser.add_argument("--pdf", type=str, help="PDF File", required=False)
    parser.add_argument("--out", type=str, help="Out dir", required=False)
    parser.add_argument(
        "--figure",
        type=str,
        help="Figure info JSON",
    )
    parser.add_argument("--dpi", type=int, default=96)
    args = parser.parse_args()

    doc_name = "2303.00058"
    # doc_name = "2402.00041"
    base_path = "/home/ninziwei/lyj/nougat/__test_0"
    args.md = f"{base_path}/markdown/{doc_name}.mmd"
    args.pdf = f"{base_path}/src/{doc_name}.pdf"
    args.figure = f"{base_path}/fig/{doc_name}.json"
    base_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test"
    args.out = f"{base_path}/out"

    md = open(args.md, "r", encoding="utf-8").read().replace("\xa0", " ")
    pdf = pypdf.PdfReader(args.pdf)
    fig_info = json.load(open(args.figure, "r", encoding="utf-8"))

    pages = [pdf.pages[i].extract_text() for i in range(len(pdf.pages))]
    with open(f"{args.out}/{doc_name}-pypdf.txt", "w", encoding="utf-8") as f:
        for page in pages:
            f.write(page)
            f.write("\n##### @@ #####\n")

    pages, page_spans, coinside_pages, bad_pages = split_markdown(md, pdf, fig_info)

    printerr(563, len(pages))
    with open(f"{args.out}/{doc_name}.txt", "w", encoding="utf-8") as f:
        for page in pages:
            f.write(page)
            f.write("\n##### @@ #####\n")

    if args.out:
        outpath = os.path.join(args.out, os.path.basename(args.pdf).partition(".")[0])
        os.makedirs(outpath, exist_ok=True)
        found_pages = []
        for i, content in enumerate(pages):
            if content:
                with open(
                    os.path.join(outpath, "%02d.mmd" % (i + 1)),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(content)
                found_pages.append(i)
        rasterize_paper(args.pdf, outpath, dpi=args.dpi, pages=found_pages)


if __name__ == "__main__":
    use_split_markdown()
