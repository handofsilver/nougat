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
from nougat.dataset.split_utils.character_splitter import CharacterSplitter
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
    valid_lines_of_pages, detailed_mappings = filter_and_match_lines(
        pdf, ordered_lines, unordered_lines
    )

    # 使用page_splitter进行行级分页
    page_splitter = PageSplitter(doc_lines, line_tag_map)
    doc_pages, valid_index_mappings = page_splitter.split_markdown_pages(detailed_mappings)

    # TODO: 使用character_splitter进行字符级分割（处理重复行）
    # character_splitter = CharacterSplitter(doc_lines, doc_pages)
    #
    # # 获取重复索引
    # duplicate_indices = character_splitter.get_duplicate_indices()
    #
    # # 对重复索引进行字符级分割
    # split_results = character_splitter.split_duplicate_indices()
    # split_results_dict = {sr.mmd_index: sr for sr in split_results}

    # 暂时跳过字符级分割，直接使用行级分页结果
    duplicate_indices = set()
    split_results_dict = {}

    # 生成最终的页面内容
    doc_text_by_pages = []
    page_spans = []

    for page_result in doc_pages:
        page_content = []

        # 处理该页面的每个mmd_line_index
        for mmd_index in page_result.doc_lines_by_page:
            if mmd_index not in duplicate_indices:
                # 情况1：只出现一次，直接整行加入
                line_content = _get_line_content(doc_lines, mmd_index)
                if line_content:
                    page_content.append(line_content)
            else:
                # 情况2：出现多次，使用字符级分割
                if mmd_index in split_results_dict:
                    split_result = split_results_dict[mmd_index]
                    # 找到属于当前页面的分割片段
                    for split in split_result.splits:
                        if split.page_index == page_result.page_index:
                            if split.content.strip():  # 只添加非空内容
                                page_content.append(split.content)

        # 组合页面内容
        page_text = "\n".join(page_content)
        doc_text_by_pages.append(page_text)

    # 简化处理：返回空的coincident_pages和bad_pages
    coincident_pages = []
    bad_pages = []

    return doc_text_by_pages, coincident_pages, bad_pages


def _get_line_content(doc_lines: List[str], mmd_index: int) -> str:
    """获取指定行的内容"""
    if 0 <= mmd_index < len(doc_lines):
        return doc_lines[mmd_index]
    return ""
