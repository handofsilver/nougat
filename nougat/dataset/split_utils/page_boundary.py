import numpy as np
from typing import List, Tuple
from nougat.dataset.split_utils.text_cleaner import squeeze_text
from nougat.dataset.split_utils.string_matcher import get_char_match_score, build_partial_match_table

def locate_page_boundaries(
    valid_lines_of_pages: List[List[str]],
    doc_lines: List[str],
    min_window_size: int = 30,
    score_thresh: float = 0.85,
    debug: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    定位页面的边界

    Args:
        valid_lines_of_pages: 每一页的有效行
        doc_lines: 文档行
        min_window_size: 最小窗口大小
        score_thresh: 匹配分数阈值
        debug: 是否开启调试模式
    
    Returns:
        page_start_positions: 每一页的开始行索引
        page_end_positions: 每一页的结束行索引
    """
    start_pointer, end_pointer = 0, 0
    start_window_size, end_window_size = min_window_size, min_window_size
    page_start_positions = []  # 该页第一行文本在 doc_lines 中的索引
    page_end_positions = []  # 该页最后一行文本在 doc_lines 中的索引

    strip_doc_lines = [squeeze_text(line) for line in doc_lines]
    
    for page_lines in valid_lines_of_pages:
        # 如果该页没有行，则认为没有有效位置
        if not page_lines:
            page_start_positions.append(-1)
            page_end_positions.append(-1)
            continue

        # 获得该页的开始和结束行
        start_line = squeeze_text(page_lines[0])
        end_line = squeeze_text(page_lines[-1])

        start_line_table = build_partial_match_table(start_line)
        end_line_table = build_partial_match_table(end_line)

        # 计算开始行的匹配分数
        start_window_end = min(start_pointer + start_window_size, len(strip_doc_lines))
        start_scores = [
            get_char_match_score(doc_line, start_line, start_line_table)
            for doc_line in strip_doc_lines[start_pointer:start_window_end]
        ]

        if start_scores and max(start_scores) > score_thresh:
            start_idx = start_pointer + np.argmax(start_scores)
            page_start_positions.append(start_idx)
            start_pointer = start_idx + 1
            # 如果匹配成功，则恢复原始窗口大小
            start_window_size = min_window_size
        else:
            # 第一次没有匹配成功，则给个机会扩大窗口再来一次
            start_window_end = min(
                start_pointer + start_window_size + min_window_size,
                len(strip_doc_lines),
            )
            start_scores = [
                get_char_match_score(doc_line, start_line, start_line_table)
                for doc_line in strip_doc_lines[start_pointer:start_window_end]
            ]

            if start_scores and max(start_scores) > score_thresh:
                start_idx = start_pointer + np.argmax(start_scores)
                page_start_positions.append(start_idx)
                start_pointer = start_idx + 1
                # 如果匹配成功，则恢复原始窗口大小
                start_window_size = min_window_size
            else:
                page_start_positions.append(-2)
                # 如果匹配失败，则扩大窗口
                start_window_size += min_window_size

        # 计算结束行的匹配分数
        end_window_end = min(end_pointer + end_window_size, len(strip_doc_lines))
        end_scores = [
            get_char_match_score(doc_line, end_line, end_line_table)
            for doc_line in strip_doc_lines[end_pointer:end_window_end]
        ]

        if debug and "six-dofhapti" in end_line:
            print(max(end_scores), end_pointer, end_window_end, end_line)
            idx = np.argmax(end_scores)
            print(idx)
            print(strip_doc_lines[end_pointer:end_window_end][idx])

        if end_scores and max(end_scores) > score_thresh:
            end_idx = end_pointer + np.argmax(end_scores)
            page_end_positions.append(end_idx)
            end_pointer = end_idx + 1
            # 如果匹配成功，则恢复原始窗口大小
            end_window_size = min_window_size
        else:
            # 第一次没有匹配成功，则给个机会扩大窗口再来一次
            end_window_end = min(
                end_pointer + end_window_size + min_window_size, len(strip_doc_lines)
            )
            end_scores = [
                get_char_match_score(doc_line, end_line, end_line_table)
                for doc_line in strip_doc_lines[end_pointer:end_window_end]
            ]

            if end_scores and max(end_scores) > score_thresh:
                end_idx = end_pointer + np.argmax(end_scores)
                page_end_positions.append(end_idx)
                end_pointer = end_idx + 1
                # 如果匹配成功，则恢复原始窗口大小
                end_window_size = min_window_size
            else:
                page_end_positions.append(-2)
                # 如果匹配失败，则扩大窗口
                end_window_size += min_window_size

        if (
            page_end_positions[-1] > 0
            and page_start_positions[-1] > page_end_positions[-1]
        ):
            page_start_positions[-1] = -2
            start_pointer = page_end_positions[-1] - 1

        if debug:
            print(page_start_positions[-1], max(start_scores), start_line)
            print(page_end_positions[-1], max(end_scores), end_line)

    if debug:
        print(page_start_positions)
        print(page_end_positions)

    return page_start_positions, page_end_positions

def get_span_of_pages(doc_lines, page_start_positions, page_end_positions):
    """
    获取每一页的文本范围
    
    Args:
        doc_lines: 文档行列表
        page_start_positions: 每一页的开始行索引
        page_end_positions: 每一页的结束行索引
    
    Returns:
        page_span: 每一页的文本范围列表
        coinside_pages: 两页重合的页码对
        bad_pages: 两页冲突的页码对
    """
    # 确定分割位置
    start_list, end_list = [], []
    whole_start_list, whole_end_list = [], []
    bad_pages = []  # 如果开头结尾冲突，则认为这两页需要丢弃
    coinside_pages = []  # 两页重合的页码对

    # 辅助参数
    last_end = 0
    last_end_positions = [-1] + page_end_positions[:-1]
    next_start_positions = page_start_positions[1:] + [len(doc_lines)]
    # 获取每一页的开始和结束位置
    for i, (start_idx, end_idx, last_end_idx, next_start_idx) in enumerate(
        zip(
            page_start_positions,
            page_end_positions,
            last_end_positions,
            next_start_positions,
        )
    ):
        # 该页没有行，直接添加占位结果
        if start_idx == -1:
            whole_start_list.append(-1)
            whole_end_list.append(-1)
            continue
        # 该页头尾都失配，直接添加占位结果
        elif start_idx == -2 and end_idx == -2:
            start_list.append(-2)
            end_list.append(-2)
            bad_pages.append(i)
        # 该页头失配，尾匹配成功
        elif start_idx < 0 and end_idx >= 0:
            # 如果该页是第一页，则直接添加0
            if i == 0:
                start_list.append(0)
            # 如果上一页正常有效，则按照上一页来
            elif last_end_idx >= 0:
                start_list.append(last_end + 1)
            # 如果上一页是无效匹配，则直接添加占位结果
            else:
                start_list.append(last_end)
                bad_pages.append(i)
            end_list.append(end_idx)
        # 该页头匹配成功，尾失配
        elif start_idx >= 0 and end_idx < 0:
            start_list.append(start_idx)
            if next_start_idx > 0:
                end_list.append(next_start_idx - 1)
            else:
                end_list.append(start_idx)
                bad_pages.append(i)
        # 该页头尾都匹配成功
        else:
            if end_idx >= start_idx:
                start_list.append(start_idx)
                end_list.append(end_idx)
            else:
                start_list.append(last_end)
                end_list.append(last_end)
                bad_pages.append(i)

        # 更新完整参数
        whole_start_list.append(start_list[-1])
        whole_end_list.append(end_list[-1])

        last_end = end_list[-1]

        # 判断分页是否在同一段
        if end_idx == next_start_idx:
            coinside_pages.append((i, i + 1))

    page_span = [(start, end) for start, end in zip(whole_start_list, whole_end_list)]

    return page_span, coinside_pages, bad_pages 