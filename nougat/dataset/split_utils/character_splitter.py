"""
字符级分页分割器

处理跨页重复行的字符级精确分割，基于行级分页结果进行细化
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

from nougat.dataset.split_utils.region_analyzer import get_doc_indices_to_regions
from nougat.dataset.split_utils.boundary_matcher import RegionBoundaryMatcher, RegionBoundary
from nougat.dataset.split_utils.page_splitter import PageResult


@dataclass
class CharacterSegment:
    """字符级分割段"""

    start_pos: int
    end_pos: int
    page_index: int
    content: str


@dataclass
class SplitLine:
    """分割后的行信息"""

    original_line_index: int
    original_content: str
    segments: List[CharacterSegment]


class CharacterPageSplitter:
    """字符级分页分割器"""

    def __init__(
        self, doc_lines: List[str], doc_pages: List[PageResult], valid_index_mappings: List[List]
    ):
        """
        初始化分割器

        Args:
            doc_lines: MMD文档所有行内容
        """
        self.doc_lines = doc_lines
        self.doc_pages = doc_pages
        self.valid_index_mappings = valid_index_mappings

    def filter_duplicate_indices(self) -> List[int]:
        """
        筛选出Line-Level分页结果中重复出现过的MMD索引(只可能是ordered类型)

        Args:
            doc_pages: page_splitter的输出结果

        Returns:
            List[int]: 重复索引列表
        """
        index_counts = Counter()

        for page_result in self.doc_pages:
            for doc_line_index in page_result.doc_lines_by_page:
                index_counts[doc_line_index] += 1

        duplicate_indices = [idx for idx, count in index_counts.items() if count > 1]
        duplicate_indices.sort()

        return duplicate_indices

    def refine_page_splits(self) -> Tuple[List, Dict[int, SplitLine]]:
        """
        对页面分割结果进行字符级细化

        Returns:
            Tuple[List, Dict[int, SplitLine]]:
                - 细化后的页面结果
                - 字符级分割的行信息字典
        """
        # 1. 找到重复索引
        duplicate_indices = self.filter_duplicate_indices()

        if not duplicate_indices:
            return self.doc_pages, {}

        # 2. 分析连续区域
        regions_dict = get_doc_indices_to_regions(self.valid_index_mappings, duplicate_indices)

        # 3. 进行边界匹配
        boundary_matcher = RegionBoundaryMatcher(self.doc_lines)
        boundary_results = boundary_matcher.find_boundaries(regions_dict)

        # 4. 生成字符级分割结果
        split_lines = {}
        for boundary_result in boundary_results:
            doc_line_index = boundary_result.doc_line_index
            doc_line_content = self._get_doc_line_content(doc_line_index)

            if not doc_line_content:
                continue

            regions = regions_dict.get(doc_line_index, [])
            if not regions:
                continue

            # 根据边界信息直接生成分割段
            segments = self._split_line_by_boundaries(
                doc_line_content, boundary_result.boundaries, regions
            )

            if segments:
                split_lines[doc_line_index] = SplitLine(
                    original_line_index=doc_line_index,
                    original_content=doc_line_content,
                    segments=segments,
                )

        return self.doc_pages, split_lines

    def _split_line_by_boundaries(
        self, doc_line_content: str, boundaries: List, regions: List
    ) -> List[CharacterSegment]:
        """
        根据边界信息分割行内容

        注意：regions是连续区域，同一页面内可能有多个regions
        确保segments数量s和boundaries数量b满足关系：b = 2s - 2
        """
        if not boundaries or not regions:
            # 没有边界，整行属于第一个区域
            return [
                CharacterSegment(
                    start_pos=0,
                    end_pos=len(doc_line_content) - 1,
                    page_index=regions[0].page_index,
                    content=doc_line_content,
                )
            ]

        # 按boundary_type和位置排序边界
        sorted_boundaries = sorted(boundaries, key=lambda b: (b.start_pos, b.boundary_type))

        # 构建分割点对：每对代表一个segment的起始和结束
        segment_ranges = []
        current_start = 0

        i = 0
        while i < len(sorted_boundaries):
            if i + 1 < len(sorted_boundaries):
                # 成对处理边界
                boundary1 = sorted_boundaries[i]
                boundary2 = sorted_boundaries[i + 1]

                if boundary1.boundary_type == 'last' and boundary2.boundary_type == 'first':
                    # region末尾 + region开头的配对
                    segment_end = boundary1.end_pos
                    next_segment_start = boundary2.start_pos

                    # 可以在这里做一个判断，如果中间有个小片段，我们根据前几个字符匹配，如果匹配到，则将小片段加入到某一个segment中
                    # refined_segment_end, refined_next_segment_start = self._merge_missing_segment(
                    #     doc_line_content, segment_end, next_segment_start, boundary1, boundary2
                    # )
                    
                    if next_segment_start > segment_end + 1:
                        # 有间隙，直接跳过（扔掉小片段）
                        pass
                    
                    refined_segment_end, refined_next_segment_start = segment_end, next_segment_start
                        
                    # 添加当前segment
                    if current_start <= refined_segment_end:
                        segment_ranges.append((current_start, refined_segment_end))

                    current_start = refined_next_segment_start
                    i += 2
                else:
                    # 处理单个边界的情况
                    i += 1
            else:
                # 最后一个边界
                i += 1

        # 添加最后一个segment
        if current_start < len(doc_line_content):
            segment_ranges.append((current_start, len(doc_line_content) - 1))

        # 根据分割范围创建segments
        segments = []
        for i, (start, end) in enumerate(segment_ranges):
            content = doc_line_content[start : end + 1]
            if content.strip():  # 只保留非空segments
                # 确定对应的页面索引
                page_idx = regions[i].page_index if i < len(regions) else regions[-1].page_index

                segments.append(
                    CharacterSegment(
                        start_pos=start, end_pos=end, page_index=page_idx, content=content
                    )
                )

        return segments

    def _merge_missing_segment(
        self,
        doc_line_content: str,
        segment_end: int,
        next_segment_start: int,
        boundary1: RegionBoundary,
        boundary2: RegionBoundary,
    ) -> Tuple[int, int]:
        """
        合并缺失的segment
        """
        if next_segment_start == segment_end + 1:  # 完美匹配，直接返回
            return segment_end, next_segment_start

        last_pdf_line_reversed = boundary1.pdf_line_content[::-1].strip().lower()
        next_pdf_line = boundary2.pdf_line_content.strip().lower()

        last_segment_content_reversed = doc_line_content[:segment_end + 1][::-1].strip().lower()
        next_segment_content = doc_line_content[next_segment_start:].strip().lower()

        last_match_length = 0
        next_match_length = 0

        i = 0
        while i < min(len(last_segment_content_reversed), len(last_pdf_line_reversed)):
            if last_segment_content_reversed[i] == last_pdf_line_reversed[i]:
                last_match_length += 1
            else:
                break
            i += 1

        j = 0
        while j < min(len(next_segment_content), len(next_pdf_line)):
            if next_segment_content[j] == next_pdf_line[j]:
                next_match_length += 1
            else:
                break
            j += 1

        # with open('character_splitter.txt', 'a') as f:
        #     f.write(
        #         f'last_match_length: {last_match_length}, next_match_length: {next_match_length}\n'
        #     )
        #     f.write(f'last_segment_content: {last_segment_content_reversed[::-1]}\n')
        #     f.write(f'next_segment_content: {next_segment_content}\n')
        #     f.write(f'last_pdf_line: {last_pdf_line_reversed[::-1]}\n')
        #     f.write(f'next_pdf_line: {next_pdf_line}\n')
        #     f.write(f'segment_end: {segment_end}, next_segment_start: {next_segment_start}\n')
        #     f.write(f'doc_line_content: {doc_line_content}\n')
        #     f.write(f"--------------------------------\n\n")

        if last_match_length > next_match_length:
            return segment_end, segment_end + 1
        else:
            return next_segment_start - 1, next_segment_start

    def _get_doc_line_content(self, doc_line_index: int) -> str:
        """获取指定行的内容"""
        if 0 <= doc_line_index < len(self.doc_lines):
            return self.doc_lines[doc_line_index]
        return ""


def split_characters_in_pages(
    doc_lines: List[str], doc_pages: List, valid_index_mappings: List[List]
) -> Tuple[List, Dict[int, SplitLine]]:
    """
    对页面分割结果进行字符级分割的便捷函数

    Args:
        doc_lines: MMD文档所有行内容
        doc_pages: 页面分割结果
        valid_index_mappings: 有效索引映射

    Returns:
        Tuple[List, Dict[int, SplitLine]]: 细化后的页面结果和分割行信息
    """
    splitter = CharacterPageSplitter(doc_lines, doc_pages, valid_index_mappings)
    return splitter.refine_page_splits()
