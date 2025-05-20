import numpy as np
from typing import List, Tuple, TypeVar, Optional
from nougat.dataset.split_utils.text_cleaner import squeeze_text
from nougat.dataset.split_utils.string_matcher import get_char_match_score
from dataclasses import dataclass
from itertools import chain
from enum import Enum, auto

# 定义类型别名，使代码更易读
PageIndex = int  # 页面索引类型
PageSpan = Tuple[PageIndex, PageIndex]  # 页面范围类型
PagePair = Tuple[PageIndex, PageIndex]  # 页面对类型


class PageIndex:
    START_OF_DOC = 0  # 文档开始位置
    NO_LINES = -1  # 页面没有有效行
    BOTH_MISSING = -2  # 页面头尾都失配
    INVALID_POS = -3  # 无效位置


class PageMatchStatus(Enum):
    """页面匹配状态"""

    BOTH_MATCHED = 1  # 页面头尾都匹配成功
    NO_LINES = -1  # 页面没有有效行
    BOTH_MISSING = -2  # 页面头尾都失配
    HEAD_MISSING = -3  # 页面头失配
    TAIL_MISSING = -4  # 页面尾失配
    ORDER_WRONG = -5  # 页面头尾匹配但顺序错误


@dataclass
class PageBoundaryResult:
    """页面边界处理结果"""

    page_spans: List[PageSpan]  # 每页的文本范围
    coincident_pages: List[PagePair]  # 重合的页面对
    bad_pages: List[PageIndex]  # 需要丢弃的页码


@dataclass
class BoundaryMatchResult:
    """边界匹配结果"""

    status: PageMatchStatus  # 匹配状态
    start_pos: PageIndex  # 处理后的开始位置
    end_pos: PageIndex  # 处理后的结束位置
    last_end: PageIndex  # 更新后的last_end
    should_skip: bool  # 是否应该跳过当前页面


def _check_page_coincidence(
    current_end: PageIndex, next_start: PageIndex, current_page: PageIndex
) -> Optional[PagePair]:
    """
    检查两个页面是否重合

    Args:
        current_end: 当前页的结束位置
        next_start: 下一页的开始位置
        current_page: 当前页码

    Returns:
        Optional[PagePair]: 如果页面重合，返回(当前页, 下一页)的页码对；否则返回None
    """
    if current_end == next_start:
        return (current_page, current_page + 1)
    return None


def _get_match_status(start_idx: PageIndex, end_idx: PageIndex) -> PageMatchStatus:
    """
    根据开始和结束索引确定匹配状态

    Args:
        start_idx: 开始索引
        end_idx: 结束索引

    Returns:
        PageMatchStatus: 匹配状态
    """
    if start_idx == PageIndex.NO_LINES:
        return PageMatchStatus.NO_LINES
    if start_idx == PageIndex.BOTH_MISSING and end_idx == PageIndex.BOTH_MISSING:
        return PageMatchStatus.BOTH_MISSING
    if start_idx < PageIndex.START_OF_DOC and end_idx >= PageIndex.START_OF_DOC:
        return PageMatchStatus.HEAD_MISSING
    if start_idx >= PageIndex.START_OF_DOC and end_idx < PageIndex.START_OF_DOC:
        return PageMatchStatus.TAIL_MISSING
    if end_idx >= start_idx:
        return PageMatchStatus.BOTH_MATCHED
    return PageMatchStatus.ORDER_WRONG


def _handle_page_boundaries(
    start_idx: PageIndex,
    end_idx: PageIndex,
    last_end_idx: PageIndex,
    next_start_idx: PageIndex,
    last_end: PageIndex,
    is_first_page: bool,
) -> BoundaryMatchResult:
    """
    处理单个页面的边界情况

    Args:
        doc_lines: 文档行列表
        start_idx: 当前页的开始索引
        end_idx: 当前页的结束索引
        last_end_idx: 上一页的结束索引
        next_start_idx: 下一页的开始索引
        last_end: 上一页的实际结束位置
        is_first_page: 是否是第一页

    Returns:
        BoundaryMatchResult: 边界匹配结果
    """
    status = _get_match_status(start_idx, end_idx)

    # 根据不同的匹配状态处理边界
    if status == PageMatchStatus.NO_LINES:
        return BoundaryMatchResult(
            status=status,
            start_pos=PageIndex.NO_LINES,
            end_pos=PageIndex.NO_LINES,
            last_end=last_end,
            should_skip=True,
        )

    if status == PageMatchStatus.BOTH_MISSING:
        return BoundaryMatchResult(
            status=status,
            start_pos=PageIndex.BOTH_MISSING,
            end_pos=PageIndex.BOTH_MISSING,
            last_end=last_end,
            should_skip=False,
        )

    if status == PageMatchStatus.HEAD_MISSING:
        if is_first_page:
            # 第一页特殊处理：如果头失配，使用0作为开始位置
            return BoundaryMatchResult(
                status=status,
                start_pos=PageIndex.START_OF_DOC,
                end_pos=end_idx,
                last_end=end_idx,
                should_skip=False,
            )
        if last_end_idx >= PageIndex.START_OF_DOC:
            # 如果上一页正常，从上一页结束位置后开始
            return BoundaryMatchResult(
                status=status,
                start_pos=last_end + 1,
                end_pos=end_idx,
                last_end=end_idx,
                should_skip=False,
            )
        # 如果上一页无效，使用上一页的结束位置
        return BoundaryMatchResult(
            status=status, start_pos=last_end, end_pos=end_idx, last_end=end_idx, should_skip=False
        )

    if status == PageMatchStatus.TAIL_MISSING:
        if next_start_idx > PageIndex.START_OF_DOC:
            # 如果下一页正常，使用下一页开始位置前作为结束
            return BoundaryMatchResult(
                status=status,
                start_pos=start_idx,
                end_pos=next_start_idx - 1,
                last_end=next_start_idx - 1,
                should_skip=False,
            )
        # 如果下一页无效，使用当前开始位置作为结束
        return BoundaryMatchResult(
            status=status,
            start_pos=start_idx,
            end_pos=start_idx,
            last_end=start_idx,
            should_skip=False,
        )

    if status == PageMatchStatus.BOTH_MATCHED:
        return BoundaryMatchResult(
            status=status, start_pos=start_idx, end_pos=end_idx, last_end=end_idx, should_skip=False
        )

    # PageMatchStatus.ORDER_WRONG
    return BoundaryMatchResult(
        status=status, start_pos=last_end, end_pos=last_end, last_end=last_end, should_skip=False
    )


def get_span_of_pages(
    doc_lines: List[str], page_start_positions: List[PageIndex], page_end_positions: List[PageIndex]
) -> PageBoundaryResult:
    """
    获取每一页的文本范围，并处理页面边界情况

    Args:
        doc_lines: 文档行列表
        page_start_positions: 每一页的开始行索引
        page_end_positions: 每一页的结束行索引

    Returns:
        PageBoundaryResult: 包含页面范围、重合页面和无效页面的结果对象

    Example:
        >>> doc_lines = ["第1页", "第2页", "第3页"]
        >>> starts = [0, 1, 2]
        >>> ends = [0, 1, 2]
        >>> result = get_span_of_pages(doc_lines, starts, ends)
        >>> print(result.page_spans)  # [(0,0), (1,1), (2,2)]
        >>> print(result.coincident_pages)  # []
        >>> print(result.bad_pages)  # []
    """
    # 初始化结果
    result = PageBoundaryResult(
        page_spans=[],  # 存储每页的文本范围
        coincident_pages=[],  # 存储重合的页码对
        bad_pages=[],  # 存储需要丢弃的页码
    )

    # 准备辅助参数
    last_end: PageIndex = PageIndex.START_OF_DOC
    last_end_positions = [PageIndex.INVALID_POS] + page_end_positions[:-1]  # 上一页的结束位置列表
    next_start_positions = page_start_positions[1:] + [len(doc_lines)]  # 下一页的开始位置列表

    # 处理每一页的边界
    for i, (start_idx, end_idx, last_end_idx, next_start_idx) in enumerate(
        zip(page_start_positions, page_end_positions, last_end_positions, next_start_positions)
    ):
        # 处理当前页的边界
        match_result = _handle_page_boundaries(
            start_idx=start_idx,
            end_idx=end_idx,
            last_end_idx=last_end_idx,
            next_start_idx=next_start_idx,
            last_end=last_end,
            is_first_page=(i == 0),
        )

        # 如果应该跳过当前页面，直接添加占位结果
        if match_result.should_skip:
            result.page_spans.append((PageIndex.NO_LINES, PageIndex.NO_LINES))
            continue

        # 更新结果
        result.page_spans.append((match_result.start_pos, match_result.end_pos))
        if match_result.status in [PageMatchStatus.BOTH_MISSING, PageMatchStatus.ORDER_WRONG]:
            result.bad_pages.append(i)

        # 检查页面是否重合
        if i < len(page_start_positions) - 1:
            if coincident := _check_page_coincidence(
                current_end=match_result.end_pos,
                next_start=page_start_positions[i + 1],
                current_page=i,
            ):
                result.coincident_pages.append(coincident)

        # 更新last_end
        last_end = match_result.last_end

    return result


def _match_page_boundary(
    doc_lines: List[str],
    query_line: str,
    pointer: PageIndex,
    window_size: int,
    score_thresh: float,
    min_window_size: int,
) -> Tuple[PageIndex, PageIndex, int]:
    """
    匹配单个页面边界

    Args:
        doc_lines: 清理后的文档行
        query_line: 要匹配的查询行
        pointer: 当前指针位置
        window_size: 当前窗口大小
        score_thresh: 匹配分数阈值
        min_window_size: 最小窗口大小

    Returns:
        Tuple[PageIndex, PageIndex, int]: (匹配位置, 新的指针位置, 新的窗口大小)
    """
    # 第一次尝试：使用当前窗口
    window_end = min(pointer + window_size, len(doc_lines))
    scores = [
        get_char_match_score(content=doc_line, query=query_line)
        for doc_line in doc_lines[pointer:window_end]
    ]

    if scores and max(scores) > score_thresh:
        match_idx = pointer + np.argmax(scores)
        return match_idx, match_idx + 1, min_window_size

    # 第二次尝试：扩大窗口
    window_end = min(pointer + window_size + min_window_size, len(doc_lines))
    scores = [
        get_char_match_score(content=doc_line, query=query_line)
        for doc_line in doc_lines[pointer:window_end]
    ]

    if scores and max(scores) > score_thresh:
        match_idx = pointer + np.argmax(scores)
        return match_idx, match_idx + 1, min_window_size

    return PageIndex.BOTH_MISSING, pointer, window_size + min_window_size


def locate_page_boundaries(
    valid_lines_of_pages: List[List[str]],
    doc_lines: List[str],
    min_window_size: int = 30,
    score_thresh: float = 0.85,
) -> Tuple[List[PageIndex], List[PageIndex]]:
    """
    定位页面的边界

    Args:
        valid_lines_of_pages: 每一PDF页的有效行
        doc_lines: Markdown文档行
        min_window_size: 最小窗口大小
        score_thresh: 匹配分数阈值

    Returns:
        Tuple[List[PageIndex], List[PageIndex]]:
            - page_start_positions: 每一Markdown页的开始行索引
            - page_end_positions: 每一Markdown页的结束行索引
    """
    # 初始化
    start_pointer = end_pointer = PageIndex.START_OF_DOC
    start_window_size = end_window_size = min_window_size
    page_start_positions = []  # 该Markdown页第一行文本在 doc_lines 中的索引
    page_end_positions = []  # 该Markdown页最后一行文本在 doc_lines 中的索引

    # 预处理：清理空白字符
    strip_doc_lines = [squeeze_text(line).lower() for line in doc_lines]

    # 处理每一页
    for page_lines in valid_lines_of_pages:
        # 处理空页
        if not page_lines:
            page_start_positions.append(PageIndex.NO_LINES)
            page_end_positions.append(PageIndex.NO_LINES)
            continue

        # 获取并清理页面边界行
        start_line = squeeze_text(page_lines[0])
        end_line = squeeze_text(page_lines[-1])

        # 匹配开始边界
        start_idx, start_pointer, start_window_size = _match_page_boundary(
            doc_lines=strip_doc_lines,
            query_line=start_line,
            pointer=start_pointer,
            window_size=start_window_size,
            score_thresh=score_thresh,
            min_window_size=min_window_size,
        )
        page_start_positions.append(start_idx)

        # 匹配结束边界
        end_idx, end_pointer, end_window_size = _match_page_boundary(
            doc_lines=strip_doc_lines,
            query_line=end_line,
            pointer=end_pointer,
            window_size=end_window_size,
            score_thresh=score_thresh,
            min_window_size=min_window_size,
        )
        page_end_positions.append(end_idx)

        # 处理顺序错误的情况
        if end_idx > PageIndex.START_OF_DOC and start_idx > end_idx:
            page_start_positions[-1] = PageIndex.BOTH_MISSING
            start_pointer = end_idx - 1

    return page_start_positions, page_end_positions
