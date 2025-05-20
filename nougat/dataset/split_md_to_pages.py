"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

"""

import pypdf
from typing import Dict, List, Tuple
from nougat.dataset.split_utils.markdown_parser import parse_markdown_lines
from nougat.dataset.split_utils.line_matcher import filter_and_match_lines
from nougat.dataset.split_utils.page_boundary import locate_page_boundaries, get_span_of_pages


def split_markdown(
    doc: str, pdf: pypdf.PdfReader, figure_info: List[Dict], debug=False
) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
    """
    Split a PDF document into Markdown paragraphs.

    Args:
        doc (str): latex 转 html 再转 markdown 后 md 文本内容.
        pdf (pypdf.PdfReader): 用 pypdf 读取 pdf 文件的结果.
        figure_info (Optional[List[Dict]]): 图表信息，每个字典指定一个图表的信息，包括标题、页码和边界框.

    Returns:
        Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
            - doc_lines_by_pages: 每一页的行数组
            - page_spans: 每一页的文本范围
            - coincident_pages: 两页重合的页码对
            - bad_pages: 两页冲突的页码对
    """
    # 解析markdown文本
    doc_lines, text_obj_map, line_tag_map = parse_markdown_lines(doc)

    # 过滤和匹配行
    valid_lines_of_pages = filter_and_match_lines(pdf, doc_lines, debug=debug)

    # 定位页边界
    page_start_positions, page_end_positions = locate_page_boundaries(
        valid_lines_of_pages, doc_lines
    )

    # 获取分割位置
    result = get_span_of_pages(doc_lines, page_start_positions, page_end_positions)

    # 根据分割位置拆分文档
    doc_text_by_pages = []
    for start, end in result.page_spans:
        if start > 0:
            doc_text_by_pages.append("\n".join(doc_lines[start : end + 1]))
        else:
            doc_text_by_pages.append("")

    return doc_text_by_pages, result.page_spans, result.coincident_pages, result.bad_pages
