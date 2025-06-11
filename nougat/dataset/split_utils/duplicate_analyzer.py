"""
重复索引分析器

分析page_splitter输出结果中重复出现的mmd_index，
为后续的字符级分页精化提供输入数据。
"""

from typing import List
from collections import Counter
from nougat.dataset.split_utils.page_splitter import PageResult


def filter_duplicate_indices(doc_pages: List[PageResult]) -> List[int]:
    """
    筛选出Line-Level分页结果中重复出现过的MMD索引(只可能是ordered类型)

    Args:
        doc_pages: page_splitter的输出结果

    Returns:
        List[int]: 重复索引列表
    """
    index_counts = Counter()

    for page_result in doc_pages:
        for doc_line_index in page_result.doc_lines_by_page:
            index_counts[doc_line_index] += 1

    duplicate_indices = [idx for idx, count in index_counts.items() if count > 1]
    duplicate_indices.sort()

    return duplicate_indices
