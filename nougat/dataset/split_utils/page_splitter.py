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

    def split_markdown_pages(self, detailed_mappings: List[List]) -> List[PageResult]:
        """
        执行分页

        Args:
            detailed_mappings: 来自line_matcher的详细映射结果 [[DualMatchResult对象列表]]

        Returns:
            List[PageResult]: 分页结果
        """
        # 初始化数据结构
        matched_doc_lines_set = self._build_matched_set(detailed_mappings)
        processed_ordered_set = set()
        processed_unordered_set = set()
        current_ordered_pointer = -1

        doc_pages = []

        # 遍历每个PDF页
        for pdf_page_idx, page_mappings in enumerate(detailed_mappings):
            page_result, page_unordered_lines, ordered_pointer = self._process_single_page(
                pdf_page_idx=pdf_page_idx,
                page_mappings=page_mappings,
                matched_doc_lines_set=matched_doc_lines_set,
                processed_ordered_set=processed_ordered_set,
                processed_unordered_set=processed_unordered_set,
                current_ordered_pointer=current_ordered_pointer,
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

        return doc_pages

    def _build_matched_set(self, detailed_mappings: List[List]) -> Set[DocLineIndex]:
        """构建匹配成功的Markdown行索引集合"""
        matched_doc_lines_set = set()
        for page_mappings in detailed_mappings:
            for mapping in page_mappings:
                if mapping.is_valid and mapping.matched_doc_line_idx is not None:
                    matched_doc_lines_set.add(mapping.matched_doc_line_idx)
        return matched_doc_lines_set

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

        # 收集有效映射并去重连续重复
        valid_mappings = []
        for mapping in page_mappings:
            if mapping.is_valid and mapping.matched_doc_line_idx is not None:
                doc_line_idx = mapping.matched_doc_line_idx
                # 避免连续重复添加相同行
                if not valid_mappings or valid_mappings[-1] != doc_line_idx:
                    valid_mappings.append(doc_line_idx)

        # 按顺序处理去重后的映射
        last_ordered_in_page = current_ordered_pointer  # 跟踪页面内最后一个ordered
        page_ordered_pointer = current_ordered_pointer  # 页面开始时的指针状态

        for doc_line_idx in valid_mappings:
            content_type = self.get_content_type(doc_line_idx)

            if content_type == ContentType.UNORDERED:
                # 处理unordered行
                if doc_line_idx not in processed_unordered_set:
                    current_page_lines.append(doc_line_idx)
                    page_unordered_lines.add(doc_line_idx)
            else:
                # 处理ordered行
                if doc_line_idx == last_ordered_in_page:
                    # 跨页面情况：同一行在新页面重新出现
                    # 只有当这个行等于页面开始时的指针状态时，才是真正的跨页重复
                    if pdf_page_idx > 0 and doc_line_idx == page_ordered_pointer:
                        current_page_lines.append(doc_line_idx)
                        # 注意：不更新last_ordered_in_page，保持当前指针位置
                    # 否则是同一页内的重复，忽略

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
                    current_page_lines.append(doc_line_idx)
                    last_ordered_in_page = doc_line_idx

        # 后处理：检查ENDTEXT
        current_page_lines = self._post_process_endtext(current_page_lines)

        page_result = PageResult(page_index=pdf_page_idx, doc_lines_by_page=current_page_lines)

        return page_result, page_unordered_lines, last_ordered_in_page

    def _post_process_endtext(self, current_page_lines: List[DocLineIndex]) -> List[DocLineIndex]:
        """后处理：检查并添加ENDTEXT行"""
        if not current_page_lines:
            return current_page_lines

        last_line_idx = max(current_page_lines)
        next_line_idx = last_line_idx + 1

        if self.is_endtext_line(next_line_idx):
            current_page_lines.append(next_line_idx)

        return current_page_lines
