"""
字符级分割器

分析重复行并提供字符级分割结果，供页面分割器使用。
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from nougat.dataset.split_utils.duplicate_analyzer import filter_duplicate_indices
from nougat.dataset.split_utils.region_analyzer import RegionAnalyzer
from nougat.dataset.split_utils.boundary_matcher import BoundaryMatcher


@dataclass
class ContentSplit:
    """内容分割信息"""

    start_char: int  # 分割起始字符位置
    end_char: int  # 分割结束字符位置
    page_index: int  # 分配到的页面索引
    content: str  # 分割的内容
    region_id: int  # 对应的区域ID


@dataclass
class IndexSplitResult:
    """索引分割结果"""

    mmd_index: int  # MMD行索引
    original_content: str  # 原始内容
    total_splits: int  # 总分割数
    splits: List[ContentSplit]  # 分割列表


class CharacterSplitter:
    """字符级分割器 - 专门负责分析重复行并提供片段内容"""

    def __init__(self, doc_lines_by_page: List[str], page_results: List):
        """
        初始化分割器

        Args:
            doc_lines_by_page: MMD文档按行分割的内容
            page_results: 页面结果列表（包含每页的行索引映射）
        """
        self.doc_lines_by_page = doc_lines_by_page
        self.page_results = page_results

        # 初始化两个分析器
        self.region_analyzer = RegionAnalyzer(page_results)
        self.boundary_matcher = BoundaryMatcher(doc_lines_by_page)

    def split_duplicate_indices(self) -> List[IndexSplitResult]:
        """
        对重复索引进行字符级分割

        Returns:
            List[IndexSplitResult]: 每个重复索引的分割结果
        """
        # 步骤1：找到重复索引
        duplicate_indices = filter_duplicate_indices(self.page_results)

        if not duplicate_indices:
            return []

        # 步骤2：分析连续区域
        region_results = self.region_analyzer.analyze_regions(duplicate_indices)

        # 步骤3：进行边界匹配
        boundary_results = self.boundary_matcher.match_boundaries(region_results)

        # 步骤4：生成分割结果
        split_results = []

        for boundary_result in boundary_results:
            split_result = self._generate_split_result(boundary_result, region_results)
            if split_result:
                split_results.append(split_result)

        return split_results

    def _generate_split_result(self, boundary_result, region_results) -> Optional[IndexSplitResult]:
        """根据边界匹配结果生成分割结果"""
        mmd_index = boundary_result.index
        original_content = self._get_line_content(mmd_index)

        if not original_content:
            return None

        # 获取该索引的区域信息
        region_result = region_results.get(mmd_index)
        if not region_result:
            return None

        # 根据边界测试结果确定分割点
        split_points = self._determine_split_points(boundary_result, original_content)

        # 生成分割片段
        splits = []
        for i, (start_char, end_char, page_index, region_id) in enumerate(split_points):
            content = original_content[start_char : end_char + 1] if end_char >= start_char else ""

            splits.append(
                ContentSplit(
                    start_char=start_char,
                    end_char=end_char,
                    page_index=page_index,
                    content=content,
                    region_id=region_id,
                )
            )

        return IndexSplitResult(
            mmd_index=mmd_index,
            original_content=original_content,
            total_splits=len(splits),
            splits=splits,
        )

    def _determine_split_points(
        self, boundary_result, original_content: str
    ) -> List[Tuple[int, int, int, int]]:
        """
        根据边界测试结果确定分割点

        Returns:
            List[Tuple[start_char, end_char, page_index, region_id]]: 分割点列表
        """
        content_length = len(original_content)
        split_points = []

        # 按边界位置排序
        boundaries = []
        for test in boundary_result.test_results:
            if test.found:
                boundaries.append((test.char_pos, test))

        boundaries.sort()

        # 生成分割点
        current_start = 0
        current_page = self._get_first_page_index(boundary_result)
        current_region = 0

        for char_pos, test in boundaries:
            # 添加当前分割
            if char_pos > current_start:
                split_points.append((current_start, char_pos - 1, current_page, current_region))

            # 更新状态
            current_start = char_pos
            current_page = self._get_page_index_from_test(test)
            current_region += 1

        # 添加最后一个分割
        if current_start < content_length:
            end_char = content_length - 1
            end_page = self._get_last_page_index(boundary_result)
            split_points.append((current_start, end_char, end_page, current_region))

        return split_points

    def _get_first_page_index(self, boundary_result) -> int:
        """获取第一个页面索引"""
        if boundary_result.test_results:
            return boundary_result.test_results[0].region.start_page
        return 0

    def _get_page_index_from_test(self, test) -> int:
        """从测试结果获取页面索引"""
        if hasattr(test, 'region') and test.region:
            return test.region.start_page
        return 0

    def _get_last_page_index(self, boundary_result) -> int:
        """获取最后一个页面索引"""
        if boundary_result.test_results:
            return boundary_result.test_results[-1].region.end_page
        return 0

    def _get_line_content(self, mmd_index: int) -> str:
        """获取指定行的内容"""
        if 0 <= mmd_index < len(self.doc_lines_by_page):
            return self.doc_lines_by_page[mmd_index]
        return ""
