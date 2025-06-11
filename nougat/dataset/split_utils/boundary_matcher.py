"""
字符边界匹配器

对重复索引的连续区域边界进行精确的字符级匹配
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from nougat.dataset.split_utils.string_matcher import get_char_match_score


@dataclass
class RegionBoundary:
    """区域边界信息"""

    region_id: int
    pdf_line_index: int
    match_score: float
    start_pos: int
    end_pos: int
    match_type: str
    boundary_type: str  # "first" 或 "last"


@dataclass
class LineBoundaries:
    """行的所有边界匹配结果"""

    doc_line_index: int
    boundaries: List[RegionBoundary]


class RegionBoundaryMatcher:
    """区域边界匹配器"""

    def __init__(self, doc_lines: List[str]):
        self.doc_lines = doc_lines

    def find_boundaries(self, doc_indices_to_regions: Dict) -> List[LineBoundaries]:
        """
        找到所有重复索引的区域边界

        Args:
            doc_indices_to_regions: 区域分析结果

        Returns:
            List[LineBoundaries]: 边界匹配结果列表
        """
        results = []

        for doc_line_index, regions in doc_indices_to_regions.items():
            boundaries = []

            for i in range(len(regions) - 1):
                current_region = regions[i]
                next_region = regions[i + 1]

                # 测试当前区域的结束边界
                end_boundary = self._match_boundary(
                    doc_line_index, current_region.last_mapping, "last", i + 1
                )
                if end_boundary:
                    boundaries.append(end_boundary)

                # 测试下一个区域的开始边界
                start_boundary = self._match_boundary(
                    doc_line_index, next_region.first_mapping, "first", i + 2
                )
                if start_boundary:
                    boundaries.append(start_boundary)

            if boundaries:
                results.append(LineBoundaries(doc_line_index, boundaries))

        results.sort(key=lambda x: x.doc_line_index)
        return results

    def _match_boundary(
        self, doc_line_index: int, mapping, boundary_type: str, region_id: int
    ) -> Optional[RegionBoundary]:
        """匹配单个边界"""
        query = self._clean_pdf_text(mapping.pdf_line_content)
        if not query.strip():
            return None

        content = self._get_doc_line_content(doc_line_index)
        if not content:
            return None

        # 选择匹配策略
        if boundary_type == "last":
            # 对于region末尾，使用双向匹配，优选end位置靠后的结果
            match_result = get_char_match_score(
                content=content.lower(),
                query=query.lower(),
                return_position=True,
                bidirectional_match=True,
            )

            if isinstance(match_result, dict):
                score = match_result['score']
                start_pos = match_result['start']
                end_pos = match_result['end']
                match_type = match_result['match_type']
            else:
                return None
        else:
            # 对于region开头，使用正向匹配
            match_result = get_char_match_score(
                content=content.lower(),
                query=query.lower(),
                return_position=True,
                bidirectional_match=False,
            )

            if isinstance(match_result, tuple) and len(match_result) >= 3:
                score, start_pos, end_pos = match_result
                match_type = "forward"
            else:
                return None

        if score < 0.05:
            return None

        return RegionBoundary(
            region_id=region_id,
            pdf_line_index=mapping.pdf_line_index,
            match_score=score,
            start_pos=start_pos,
            end_pos=end_pos,
            match_type=match_type,
            boundary_type=boundary_type,
        )

    def _get_doc_line_content(self, doc_line_index: int) -> str:
        """获取指定索引的MMD内容"""
        if 0 <= doc_line_index < len(self.doc_lines):
            return self.doc_lines[doc_line_index]
        return ""

    def _clean_pdf_text(self, text: str) -> str:
        """清理文本，移除多余空格"""
        return re.sub(r'\s+', ' ', text.strip())
