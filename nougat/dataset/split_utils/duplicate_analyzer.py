"""
重复索引分析器

分析page_splitter输出结果中重复出现的mmd_index，
为后续的字符级分页精化提供输入数据。
"""

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from nougat.dataset.split_utils.page_splitter import PageResult, ContentType


@dataclass
class DuplicateIndexInfo:
    """重复索引信息"""

    mmd_index: int
    occurrences: List[Tuple[int, int]]  # [(page_index, position_in_page), ...]
    content_type: str  # 'ordered' or 'unordered'
    total_count: int


@dataclass
class DuplicateAnalysisResult:
    """重复索引分析结果"""

    duplicate_indices: List[int]  # 重复出现的索引列表
    duplicate_info: Dict[int, DuplicateIndexInfo]  # 详细信息
    total_duplicates: int
    summary: Dict[str, int]  # 统计摘要


class DuplicateAnalyzer:
    """重复索引分析器"""

    def __init__(self, line_tag_map: Dict[int, Dict]):
        """
        初始化分析器

        Args:
            line_tag_map: 行标签映射 {行号: {'type': 标签类型, 'content_type': 'ordered'|'unordered'}}
        """
        self.line_tag_map = line_tag_map

    def get_content_type(self, mmd_index: int) -> str:
        """获取MMD行的内容类型"""
        tag_info = self.line_tag_map.get(mmd_index, {})
        return tag_info.get('content_type', 'ordered')

    def analyze_duplicates(self, doc_pages: List[PageResult]) -> DuplicateAnalysisResult:
        """
        分析重复出现的mmd_index

        Args:
            doc_pages: page_splitter的输出结果

        Returns:
            DuplicateAnalysisResult: 重复索引分析结果
        """
        # 收集所有索引的出现情况
        index_occurrences = defaultdict(list)

        for page_result in doc_pages:
            page_index = page_result.page_index
            for position, mmd_index in enumerate(page_result.doc_lines_by_page):
                index_occurrences[mmd_index].append((page_index, position))

        # 统计出现次数
        index_counts = Counter()
        for mmd_index, occurrences in index_occurrences.items():
            index_counts[mmd_index] = len(occurrences)

        # 筛选重复出现的索引（出现次数 > 1）
        duplicate_indices = [idx for idx, count in index_counts.items() if count > 1]
        duplicate_indices.sort()

        # 生成详细信息
        duplicate_info = {}
        for mmd_index in duplicate_indices:
            occurrences = index_occurrences[mmd_index]
            content_type = self.get_content_type(mmd_index)

            duplicate_info[mmd_index] = DuplicateIndexInfo(
                mmd_index=mmd_index,
                occurrences=occurrences,
                content_type=content_type,
                total_count=len(occurrences),
            )

        # 生成统计摘要
        ordered_duplicates = sum(
            1 for info in duplicate_info.values() if info.content_type == 'ordered'
        )
        unordered_duplicates = sum(
            1 for info in duplicate_info.values() if info.content_type == 'unordered'
        )

        summary = {
            'total_duplicate_indices': len(duplicate_indices),
            'ordered_duplicates': ordered_duplicates,
            'unordered_duplicates': unordered_duplicates,
            'total_duplicate_occurrences': sum(
                info.total_count for info in duplicate_info.values()
            ),
        }

        return DuplicateAnalysisResult(
            duplicate_indices=duplicate_indices,
            duplicate_info=duplicate_info,
            total_duplicates=len(duplicate_indices),
            summary=summary,
        )

    def filter_ordered_duplicates(self, analysis_result: DuplicateAnalysisResult) -> List[int]:
        """
        筛选出ordered类型的重复索引

        Args:
            analysis_result: 重复索引分析结果

        Returns:
            List[int]: 只包含ordered类型的重复索引
        """
        ordered_duplicates = []

        for mmd_index, info in analysis_result.duplicate_info.items():
            if info.content_type == 'ordered':
                ordered_duplicates.append(mmd_index)

        ordered_duplicates.sort()
        return ordered_duplicates
