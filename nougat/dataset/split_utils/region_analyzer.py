"""
区域划分分析器

分析重复mmd_index在index_mappings中的连续出现区域，
区分真实的内容断点（分页或unordered插入）和公式导致的映射间隔。
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class RegionMapping:
    """区域映射信息"""

    pdf_line_index: int
    pdf_line_content: str
    match_score: float
    match_type: str


@dataclass
class ContinuousRegion:
    """连续区域信息"""

    region_id: int
    page_index: int
    start_pdf_line: int
    end_pdf_line: int
    mapping_count: int
    first_mapping: RegionMapping
    last_mapping: RegionMapping


class RegionAnalyzer:
    """区域划分分析器"""

    def __init__(self, index_mappings: List[Dict]):
        """
        初始化分析器

        Args:
            index_mappings: line_matcher的index_mappings输出结果
        """
        self.index_mappings = index_mappings

    def analyze_regions(self, duplicate_indices: List[int]) -> Dict[int, List[ContinuousRegion]]:
        """
        分析重复索引的连续区域

        Args:
            duplicate_indices: 重复出现的mmd索引列表

        Returns:
            Dict[int, List[ContinuousRegion]]: 每个索引的区域分析结果
        """
        results = {}

        for doc_line_index in duplicate_indices:
            regions = self._analyze_single_index_regions(doc_line_index)
            results[doc_line_index] = regions

        return results

    def _analyze_single_index_regions(self, doc_line_index: int) -> List[ContinuousRegion]:
        """分析单个索引的连续区域"""
        # 收集该索引的所有映射
        mappings_by_doc_index = self._collect_index_mappings(doc_line_index)

        if not mappings_by_doc_index:
            return []

        # 按页面和PDF行号排序
        mappings_by_doc_index = sorted(
            mappings_by_doc_index, key=lambda x: (x['page_index'], x['pdf_line_index'])
        )

        # 划分连续区域
        regions = []
        current_region_mappings = []
        current_page = None
        last_pdf_line = None

        for mapping in mappings_by_doc_index:
            page_index, pdf_line_index = mapping['page_index'], mapping['pdf_line_index']

            # 判断是否需要开始新区域
            should_start_new_region = self._should_start_new_region(
                current_page, last_pdf_line, page_index, pdf_line_index, current_region_mappings
            )

            if should_start_new_region:
                # 保存当前区域
                if current_region_mappings:
                    region = self._create_region(len(regions) + 1, current_region_mappings)
                    regions.append(region)

                # 开始新区域
                current_region_mappings = [mapping]
            else:
                # 继续当前区域
                current_region_mappings.append(mapping)

            current_page = page_index
            last_pdf_line = pdf_line_index

        # 保存最后一个区域
        if current_region_mappings:
            region = self._create_region(len(regions) + 1, current_region_mappings)
            regions.append(region)

        return regions

    def _collect_index_mappings(self, doc_line_index: int) -> List[Dict]:
        """收集指定索引的所有映射"""
        collected_mappings = []

        for page_idx, page_mappings in enumerate(self.index_mappings):
            for mapping in page_mappings:
                if mapping.matched_doc_line_index == doc_line_index:
                    mapping_with_page = {
                        'page_index': page_idx,
                        'pdf_line_index': mapping.pdf_line_index,
                        'pdf_line_content': mapping.pdf_line_content,
                        'match_score': mapping.match_score,
                        'match_type': mapping.match_type,
                    }

                    collected_mappings.append(mapping_with_page)

        return collected_mappings

    def _should_start_new_region(
        self,
        current_page: Optional[int],
        last_pdf_line: Optional[int],
        page_idx: int,
        pdf_line: int,
        current_mappings: List[Dict],
    ) -> bool:
        """判断是否应该开始新区域"""
        # 第一个映射，开始第一个区域
        if current_page is None or not current_mappings:
            return True

        # 不同页面，肯定是新区域
        if page_idx != current_page:
            return True

        # 同一页面内，检查是否有unordered内容插入
        if pdf_line > last_pdf_line + 1:
            # 检查间隔中是否有unordered内容
            gap_start = last_pdf_line + 1
            gap_end = pdf_line - 1

            has_unordered_content = self._has_unordered_content_in_gap(page_idx, gap_start, gap_end)

            if has_unordered_content:
                return True

        return False

    def _has_unordered_content_in_gap(self, page_idx: int, gap_start: int, gap_end: int) -> bool:
        """检查PDF行间隔中是否有unordered内容"""
        for mapping in self.index_mappings[page_idx]:
            pdf_line = mapping.pdf_line_index
            if gap_start <= pdf_line <= gap_end:
                if mapping.content_type == 'unordered':
                    return True

        return False

    def _create_region(self, region_id: int, mappings: List[Dict]) -> ContinuousRegion:
        """创建连续区域对象"""
        first_mapping = mappings[0]
        last_mapping = mappings[-1]

        return ContinuousRegion(
            region_id=region_id,
            page_index=first_mapping['page_index'],
            start_pdf_line=first_mapping['pdf_line_index'],
            end_pdf_line=last_mapping['pdf_line_index'],
            mapping_count=len(mappings),
            first_mapping=RegionMapping(
                pdf_line_index=first_mapping['pdf_line_index'],
                pdf_line_content=first_mapping['pdf_line_content'],
                match_score=first_mapping['match_score'],
                match_type=first_mapping['match_type'],
            ),
            last_mapping=RegionMapping(
                pdf_line_index=last_mapping['pdf_line_index'],
                pdf_line_content=last_mapping['pdf_line_content'],
                match_score=last_mapping['match_score'],
                match_type=last_mapping['match_type'],
            ),
        )
