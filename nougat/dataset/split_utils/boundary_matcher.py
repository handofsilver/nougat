"""
双向边界匹配器

对重复索引的连续区域边界进行精确的字符级匹配，
直接使用经过验证的get_char_match_score算法。
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from nougat.dataset.split_utils.string_matcher import get_char_match_score


@dataclass
class BoundaryTest:
    """边界测试结果"""

    boundary_type: str
    region_id: int
    mapping_type: str
    page_info: str
    pdf_line: int
    query_length: int
    match_score: float
    start_pos: int
    end_pos: int
    match_type: str
    query_content: str
    matched_content: str
    match_length: Optional[int] = None


@dataclass
class BoundaryMatchResult:
    """边界匹配结果"""

    index: int
    mmd_content_length: int
    total_regions: int
    boundary_tests: List[BoundaryTest]


class BoundaryMatcher:
    """双向边界匹配器"""

    def __init__(self, doc_lines_by_page: List[str]):
        """
        初始化匹配器

        Args:
            doc_lines_by_page: MMD文档按行分割的内容
        """
        self.doc_lines_by_page = doc_lines_by_page
        self.mmd_content = '\n'.join(doc_lines_by_page)

    def match_boundaries(self, region_results: Dict) -> List[BoundaryMatchResult]:
        """
        对所有重复索引进行边界匹配

        Args:
            region_results: 区域分析结果

        Returns:
            List[BoundaryMatchResult]: 边界匹配结果列表
        """
        results = []

        for mmd_index, region_result in region_results.items():
            boundary_tests = self._match_index_boundaries(mmd_index, region_result)

            match_result = BoundaryMatchResult(
                index=mmd_index,
                mmd_content_length=len(self._get_index_content(mmd_index)),
                total_regions=len(region_result),
                boundary_tests=boundary_tests,
            )
            results.append(match_result)

        # 按索引排序
        results.sort(key=lambda x: x.index)
        return results

    def _match_index_boundaries(self, mmd_index: int, region_result) -> List[BoundaryTest]:
        """匹配单个索引的所有边界"""
        boundary_tests = []

        for i in range(len(region_result)):
            region = region_result[i]

            # 只测试有相邻下一个区域的边界
            if i + 1 < len(region_result):
                next_region = region_result[i + 1]

                # 测试当前区域的最后一个映射
                last_test = self._test_region_boundary(mmd_index, region, "last", i + 1)
                if last_test:
                    boundary_tests.append(last_test)

                # 测试下一个区域的第一个映射
                first_test = self._test_region_boundary(mmd_index, next_region, "first", i + 2)
                if first_test:
                    boundary_tests.append(first_test)

        return boundary_tests

    def _test_region_boundary(
        self, mmd_index: int, region, boundary_type: str, region_id: int
    ) -> Optional[BoundaryTest]:
        """测试区域边界"""
        if boundary_type == "last":
            mapping = region.last_mapping
            mapping_type = "last_mapping"
            boundary_name = f"region{region_id}_last"
            bidirectional_match = True
            prefer_end = True
        else:  # first
            mapping = region.first_mapping
            mapping_type = "first_mapping"
            boundary_name = f"region{region_id}_first"
            bidirectional_match = True
            prefer_end = False

            # 获取查询内容
        query_content = self._clean_text(mapping.pdf_line_content)
        if not query_content.strip():
            return None

        # 获取MMD内容
        mmd_content = self._get_index_content(mmd_index)

        # 使用原有的get_char_match_score算法
        if boundary_type == "last":
            # 对于last边界，使用双向匹配，优选end位置靠后的结果
            match_result = get_char_match_score(
                content=mmd_content.lower(),
                query=query_content.lower(),
                return_position=True,
                bidirectional_match=True,
            )

            if isinstance(match_result, dict):
                match_score = match_result['score']
                start_pos = match_result['start']
                end_pos = match_result['end']
                match_type = match_result['match_type']
                match_length = match_result.get('match_length')
                matched_content = (
                    mmd_content[start_pos : end_pos + 1] if start_pos >= 0 and end_pos >= 0 else ""
                )
            else:
                return None
        else:
            # 对于first边界，使用正向匹配
            match_result = get_char_match_score(
                content=mmd_content.lower(),
                query=query_content.lower(),
                return_position=True,
                bidirectional_match=False,
            )

            if isinstance(match_result, tuple) and len(match_result) >= 3:
                match_score, start_pos, end_pos = match_result
                match_type = "forward"
                match_length = None
                matched_content = (
                    mmd_content[start_pos : end_pos + 1] if start_pos >= 0 and end_pos >= 0 else ""
                )
            else:
                return None

        if match_score < 0.05:  # 最低匹配阈值
            return None

        return BoundaryTest(
            boundary_type=boundary_name,
            region_id=region_id,
            mapping_type=mapping_type,
            page_info=f"page_{region.page_index}",
            pdf_line=mapping.pdf_line_index,
            query_length=len(query_content),
            match_score=match_score,
            start_pos=start_pos,
            end_pos=end_pos,
            match_type=match_type,
            query_content=query_content,
            matched_content=matched_content,
            match_length=match_length,
        )

    def _get_index_content(self, mmd_index: int) -> str:
        """获取指定索引的MMD内容"""
        if 0 <= mmd_index < len(self.doc_lines_by_page):
            return self.doc_lines_by_page[mmd_index]
        return ""

    def _clean_text(self, text: str) -> str:
        """清理文本，移除多余空格"""
        return re.sub(r'\s+', ' ', text.strip())
