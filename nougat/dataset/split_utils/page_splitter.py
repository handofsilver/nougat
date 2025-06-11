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

    def is_endtext_line(self, doc_line_idx: DocLineIndex) -> bool:
        """检查是否为[ENDTEXT]行"""
        if doc_line_idx >= len(self.doc_lines):
            return False
        return self.doc_lines[doc_line_idx].strip() == '[ENDTEXT]'

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
            page_result, page_unordered_lines, ordered_pointer, valid_page_mappings = (
                self._process_single_page(
                    pdf_page_idx=pdf_page_idx,
                    page_mappings=page_mappings,
                    matched_doc_lines_set=matched_doc_lines_set,
                    processed_ordered_set=processed_ordered_set,
                    processed_unordered_set=processed_unordered_set,
                    current_ordered_pointer=current_ordered_pointer,
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

        # 后处理：基于修正后的分页结果统一处理ENDTEXT
        doc_pages = self._post_process_all_endtext(doc_pages)

        return doc_pages, valid_index_mappings

    def _build_matched_set(self, index_mappings: List[List]) -> Set[DocLineIndex]:
        """构建匹配成功的Markdown行索引集合"""
        matched_doc_lines_set = set()
        for page_mappings in index_mappings:
            for mapping in page_mappings:
                if mapping.is_valid and mapping.matched_doc_line_index is not None:
                    matched_doc_lines_set.add(mapping.matched_doc_line_index)
        return matched_doc_lines_set

    def _post_process_all_endtext(self, doc_pages: List[PageResult]) -> List[PageResult]:
        """基于修正后的分页结果统一处理ENDTEXT"""
        # 1. 先移除所有现有的ENDTEXT行，避免重复
        cleaned_pages = []
        for page_result in doc_pages:
            cleaned_lines = [
                line_idx
                for line_idx in page_result.doc_lines_by_page
                if not self.is_endtext_line(line_idx)
            ]
            cleaned_page = PageResult(
                page_index=page_result.page_index, doc_lines_by_page=cleaned_lines
            )
            cleaned_pages.append(cleaned_page)

        # 2. 基于清理后的结果计算每个ordered行的最后出现页面（排除ENDTEXT）
        last_occurrence_pages = {}
        for page_result in cleaned_pages:
            for line_idx in page_result.doc_lines_by_page:
                if self.get_content_type(line_idx) == ContentType.ORDERED:
                    last_occurrence_pages[line_idx] = page_result.page_index

        # 3. 为每个页面添加合适的ENDTEXT（保持原有的偷看逻辑）
        updated_pages = []
        for page_result in cleaned_pages:
            updated_lines = self._add_endtext_for_page(
                page_result.doc_lines_by_page, page_result.page_index, last_occurrence_pages
            )
            updated_page = PageResult(
                page_index=page_result.page_index, doc_lines_by_page=updated_lines
            )
            updated_pages.append(updated_page)

        return updated_pages

    def _process_single_page(
        self,
        pdf_page_idx: int,
        page_mappings: List,
        matched_doc_lines_set: Set[DocLineIndex],
        processed_ordered_set: Set[DocLineIndex],
        processed_unordered_set: Set[DocLineIndex],
        current_ordered_pointer: int,
    ) -> PageResult:
        """处理单个PDF页 - 按映射顺序去重连续重复"""

        current_page_lines = []  # 按顺序收集行
        page_unordered_lines = set()  # 当前页新处理的unordered

        # 按顺序处理去重后的映射
        last_ordered_in_page = current_ordered_pointer  # 跟踪页面内最后一个ordered

        # 收集有效映射并去重连续重复
        last_doc_line_idx = None
        valid_page_mappings = []

        for mapping in page_mappings:
            if mapping.is_valid and mapping.matched_doc_line_index is not None:
                doc_line_idx = mapping.matched_doc_line_index
                # 避免连续重复添加相同行
                if last_doc_line_idx == doc_line_idx:
                    if (
                        valid_page_mappings
                        and valid_page_mappings[-1].matched_doc_line_index == doc_line_idx
                    ):
                        valid_page_mappings.append(mapping)
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
                        # 1. 同一行在当前页面重新出现，说明是页内unordered行隔开
                        # 2. 跨页面情况：同一行在新页面重新出现
                        valid_page_mappings.append(mapping)
                        current_page_lines.append(doc_line_idx)

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
                                for gap_idx in range(gap_start, gap_end + 1):
                                    if self.get_content_type(gap_idx) == ContentType.ORDERED:
                                        current_page_lines.append(gap_idx)
                                        last_ordered_in_page = gap_idx  # 更新指针到填充的行

                        # 添加当前ordered行
                        valid_page_mappings.append(mapping)
                        current_page_lines.append(doc_line_idx)
                        last_ordered_in_page = doc_line_idx

        page_result = PageResult(page_index=pdf_page_idx, doc_lines_by_page=current_page_lines)

        return page_result, page_unordered_lines, last_ordered_in_page, valid_page_mappings

    def _add_endtext_for_page(
        self,
        page_lines: List[DocLineIndex],
        page_idx: int,
        last_occurrence_pages: Dict[DocLineIndex, int],
    ) -> List[DocLineIndex]:
        """为单个页面添加合适的ENDTEXT，保持原有偷看逻辑但避免重复"""
        updated_lines = []

        for line_idx in page_lines:
            updated_lines.append(line_idx)

            # 如果是ordered行，检查是否需要"偷看"添加ENDTEXT
            if self.get_content_type(line_idx) == ContentType.ORDERED:
                # 检查这个ordered行是否在当前页面是最后一次出现
                is_last_occurrence = (
                    line_idx in last_occurrence_pages
                    and last_occurrence_pages[line_idx] == page_idx
                )

                if is_last_occurrence:
                    next_line_idx = line_idx + 1
                    # 原有的"偷看"逻辑：如果下一行是ENDTEXT就添加
                    if self.is_endtext_line(next_line_idx):
                        updated_lines.append(next_line_idx)

        return updated_lines
