"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

"""

import pypdf
from typing import Dict, List, Tuple
from nougat.dataset.split_utils.markdown_parser import parse_markdown_lines
from nougat.dataset.split_utils.line_matcher import filter_and_match_lines
from nougat.dataset.split_utils.page_splitter import PageSplitter
from nougat.dataset.split_utils.character_splitter import split_characters_in_pages
from nougat.dataset.split_utils.content_separator import separate_content_by_type


def split_markdown(
    doc: str, pdf: pypdf.PdfReader, figure_info: Dict
) -> Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
    """
    Split a PDF document into Markdown paragraphs.

    Args:
        doc (str): latex 转 html 再转 markdown 后 md 文本内容.
        pdf (pypdf.PdfReader): 用 pypdf 读取 pdf 文件的结果.
        figure_info (Dict): 图表信息字典，包含figures列表等信息.

    Returns:
        Tuple[List[str], List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
            - doc_lines_by_pages: 每一页的行数组
            - page_spans: 每一页的文本范围
            - coincident_pages: 两页重合的页码对
            - bad_pages: 两页冲突的页码对
    """
    # 解析markdown文本
    doc_lines, text_obj_map, line_tag_map = parse_markdown_lines(doc)

    # 分离内容
    ordered_lines, unordered_lines = separate_content_by_type(doc_lines, line_tag_map)

    # 使用line_matcher进行PDF到MMD的行映射
    valid_lines_of_pages, index_mappings = filter_and_match_lines(
        pdf, ordered_lines, unordered_lines
    )

    # 使用page_splitter进行行级分页
    page_splitter = PageSplitter(doc_lines, line_tag_map)
    doc_pages, valid_index_mappings = page_splitter.split_markdown_pages(index_mappings)

    # 使用character_splitter进行字符级分割（处理重复行）
    refined_pages, split_lines = split_characters_in_pages(
        doc_lines, doc_pages, valid_index_mappings
    )

    # 使用细化后的页面结果
    doc_pages = refined_pages

    # 生成最终的页面内容
    doc_text_by_pages = []

    for page_result in doc_pages:
        page_content = []

        # 为每个mmd_index维护出现次数计数器
        mmd_index_counters = {}

        # 处理该页面的每个doc_line_index
        for doc_line_index in page_result.doc_lines_by_page:
            if doc_line_index in split_lines:
                # 使用字符级分割结果
                split_line = split_lines[doc_line_index]

                # 获取当前mmd_index在页面中的出现次数
                if doc_line_index not in mmd_index_counters:
                    mmd_index_counters[doc_line_index] = 0

                occurrence = mmd_index_counters[doc_line_index]

                # 找到属于当前页面的segments
                page_segments = [
                    s for s in split_line.segments if s.page_index == page_result.page_index
                ]

                # 根据出现次数选择对应的segment
                if occurrence < len(page_segments):
                    segment = page_segments[occurrence]
                    if segment.content.strip():
                        page_content.append(segment.content)

                # 增加出现次数
                mmd_index_counters[doc_line_index] += 1

            else:
                # 直接使用整行内容
                doc_line_content = _get_doc_line_content(doc_lines, doc_line_index).strip()
                if doc_line_content:
                    # 检查是否在text_obj_map中有映射
                    if doc_line_content in text_obj_map:
                        content = text_obj_map[doc_line_content]
                    else:
                        content = doc_line_content

                    if content.strip():
                        page_content.append(content)

        # 组合页面内容
        page_text = "\n".join(page_content)
        doc_text_by_pages.append(page_text)

    # 简化处理：返回空的coincident_pages和bad_pages
    coincident_pages = []
    bad_pages = []

    return doc_text_by_pages, coincident_pages, bad_pages


def _get_doc_line_content(doc_lines: List[str], mmd_index: int) -> str:
    """获取指定行的内容"""
    if 0 <= mmd_index < len(doc_lines):
        return doc_lines[mmd_index]
    return ""
