"""
行级分页模块

基于PDF到MMD行映射关系，使用双指针算法实现Markdown文档的分页。
核心原则：
1. [INVALID, None]映射结果被忽略，不参与决策
2. ordered内容必须严格遵循递增原则
3. unordered内容独立处理，避免重复
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# 类型别名
PdfLineIndex = int
DocLineIndex = int
MappingResult = Tuple[str, Optional[DocLineIndex]]  # (status, doc_index)


class ContentType(Enum):
    """内容类型枚举"""

    ORDERED = "ordered"
    UNORDERED = "unordered"


@dataclass
class PageResult:
    """单页分页结果"""

    page_index: int
    doc_lines_by_page: List[DocLineIndex]
    is_valid: bool = True


class PageSplitter:
    """行级分页器"""

    def __init__(self, doc_lines: List[str], line_tag_map: Dict[int, Dict]):
        """
        初始化分页器

        Args:
            doc_lines: Markdown文档行列表
            line_tag_map: 行标签映射 {行号: {'type': 标签类型, 'content_type': 'ordered'|'unordered'}}
        """
        self.doc_lines = doc_lines
        self.line_tag_map = line_tag_map

        # 预处理：分析所有Markdown行的类型
        self.content_types = {}  # {doc_line_idx: ContentType}
        for doc_line_idx in range(len(doc_lines)):
            tag_info = line_tag_map.get(doc_line_idx, {})
            content_type = tag_info.get('content_type', 'ordered')  # 默认为ordered
            self.content_types[doc_line_idx] = (
                ContentType.ORDERED if content_type == 'ordered' else ContentType.UNORDERED
            )

    def get_content_type(self, doc_line_idx: DocLineIndex) -> ContentType:
        """获取Markdown行的内容类型"""
        return self.content_types.get(doc_line_idx, ContentType.ORDERED)

    def split_markdown_pages(self, index_mappings: List[List]) -> List[PageResult]:
        """
        执行分页

        Args:
            index_mappings: 来自line_matcher的详细映射结果 [[DualMatchResult对象列表]]

        Returns:
            List[PageResult]: 分页结果
        """
        # 初始化数据结构
        matched_doc_lines_set = self._build_matched_set(index_mappings)
        processed_ordered_set = set()
        processed_unordered_set = set()
        current_ordered_pointer = -1

        doc_pages = []
        valid_index_mappings = []

        # 遍历每个PDF页
        for pdf_page_idx, page_mappings in enumerate(index_mappings):
            last_page_result = doc_pages[-1] if doc_pages else None

            page_result, page_unordered_lines, ordered_pointer, valid_page_mappings = (
                self._process_single_page(
                    pdf_page_idx=pdf_page_idx,
                    page_mappings=page_mappings,
                    matched_doc_lines_set=matched_doc_lines_set,
                    processed_ordered_set=processed_ordered_set,
                    processed_unordered_set=processed_unordered_set,
                    current_ordered_pointer=current_ordered_pointer,
                    last_page_result=last_page_result,
                )
            )

            # 更新全局状态
            current_ordered_pointer = ordered_pointer
            # 只更新ordered部分
            page_ordered_lines = [
                idx
                for idx in page_result.doc_lines_by_page
                if self.get_content_type(idx) == ContentType.ORDERED
            ]
            processed_ordered_set.update(page_ordered_lines)
            processed_unordered_set.update(page_unordered_lines)

            doc_pages.append(page_result)
            valid_index_mappings.append(valid_page_mappings)

        return doc_pages, valid_index_mappings

    def _build_matched_set(self, index_mappings: List[List]) -> Set[DocLineIndex]:
        """构建匹配成功的Markdown行索引集合"""
        matched_doc_lines_set = set()
        for page_mappings in index_mappings:
            for mapping in page_mappings:
                if mapping.is_valid and mapping.matched_doc_line_index is not None:
                    matched_doc_lines_set.add(mapping.matched_doc_line_index)
        return matched_doc_lines_set

    def _process_single_page(
        self,
        pdf_page_idx: int,
        page_mappings: List,
        matched_doc_lines_set: Set[DocLineIndex],
        processed_ordered_set: Set[DocLineIndex],
        processed_unordered_set: Set[DocLineIndex],
        current_ordered_pointer: int,
        last_page_result: PageResult,
    ) -> PageResult:
        """处理单个PDF页 - 按映射顺序去重连续重复"""

        current_page_lines = []  # 按顺序收集行
        page_unordered_lines = set()  # 当前页新处理的unordered

        # 按顺序处理去重后的映射
        last_ordered_in_page = current_ordered_pointer  # 跟踪页面内最后一个ordered

        # 收集有效映射并去重连续重复
        last_doc_line_idx = None
        valid_page_mappings = []

        start_of_page = True
        current_page_is_valid = True

        for mapping_idx, mapping in enumerate(page_mappings):
            if not mapping.is_valid or mapping.matched_doc_line_index is None:
                continue

            doc_line_idx = mapping.matched_doc_line_index

            # 避免连续重复添加相同行
            if last_doc_line_idx == doc_line_idx:
                # 1. 同一行在当前页面重新出现，说明是页内unordered行隔开
                # 2. 跨页面情况：同一行在新页面重新出现
                if (
                    valid_page_mappings
                    and valid_page_mappings[-1].matched_doc_line_index == doc_line_idx
                ):
                    valid_page_mappings.append(mapping)
                    start_of_page = False
                continue

            last_doc_line_idx = doc_line_idx
            content_type = self.get_content_type(doc_line_idx)

            if content_type == ContentType.UNORDERED:
                # 处理unordered行
                if doc_line_idx not in processed_unordered_set:
                    valid_page_mappings.append(mapping)
                    current_page_lines.append(doc_line_idx)
                    page_unordered_lines.add(doc_line_idx)
            else:
                # 处理ordered行
                if doc_line_idx == last_ordered_in_page:
                    valid_page_mappings.append(mapping)
                    current_page_lines.append(doc_line_idx)
                    start_of_page = False

                elif doc_line_idx > last_ordered_in_page:
                    # 检查是否需要填充gap
                    if doc_line_idx > last_ordered_in_page + 1:
                        gap_start = last_ordered_in_page + 1
                        gap_end = doc_line_idx - 1

                        # 检查gap中是否有未来会匹配的ordered行
                        has_future_ordered_match = False
                        for gap_idx in range(gap_start, gap_end + 1):
                            if (
                                self.get_content_type(gap_idx) == ContentType.ORDERED
                                and gap_idx in matched_doc_lines_set
                                and gap_idx not in processed_ordered_set
                            ):
                                has_future_ordered_match = True
                                break

                        if has_future_ordered_match:
                            # 如果gap中有未来会匹配的行，跳过当前行
                            continue
                        else:
                            # 否则填充gap中的ordered行
                            gaps_to_fill = []
                            for gap_idx in range(gap_start, gap_end + 1):
                                if self.get_content_type(gap_idx) == ContentType.ORDERED:
                                    gaps_to_fill.append(gap_idx)

                            # 如果是当前页第一个mapping（有效），则填充gap到last_page_result.doc_lines_by_page
                            if mapping_idx == 0:
                                for gap_idx in gaps_to_fill:
                                    last_page_result.doc_lines_by_page.append(gap_idx)
                            # 否则填充gap到current_page_lines
                            else:
                                if start_of_page and gaps_to_fill:
                                    last_page_result.is_valid = False
                                    current_page_is_valid = False
                                    # with open('page_splitter.txt', 'a') as f:
                                    #     f.write(f"pdf_page_idx: {pdf_page_idx}\n")
                                    #     f.write(f"last_ordered_in_page: {last_ordered_in_page}\n")
                                    #     f.write(f"gap_start: {gap_start}\n")
                                    #     f.write(f"gap_end: {gap_end}\n")
                                    #     f.write(f"gap_to_fill: {gaps_to_fill}\n")
                                    #     f.write("--------------------------------\n")
                                for gap_idx in gaps_to_fill:
                                    current_page_lines.append(gap_idx)

                    # 添加当前ordered行
                    start_of_page = False
                    valid_page_mappings.append(mapping)
                    current_page_lines.append(doc_line_idx)
                    last_ordered_in_page = doc_line_idx

        page_result = PageResult(
            page_index=pdf_page_idx,
            doc_lines_by_page=current_page_lines,
            is_valid=current_page_is_valid,
        )

        return page_result, page_unordered_lines, last_ordered_in_page, valid_page_mappings
