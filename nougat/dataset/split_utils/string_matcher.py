def build_partial_match_table(pattern):
    """构建部分匹配表，用于快速找到可能的匹配起始位置"""
    table = {}
    for i in range(len(pattern) - 2):
        tri_char = pattern[i : i + 3]
        if tri_char not in table:
            table[tri_char] = []
        table[tri_char].append(i)
    return table


def calculate_similarity_score(content: str, query: str, max_gap=3, max_discontinuities=3) -> float:
    """
    计算两个字符串的相似度分数

    Args:
        content: 目标文本
        query: 查询文本
        max_gap: 不连续位置的最大间隔
        max_discontinuities: 允许的最大不连续次数

    Returns:
        float: 相似度分数 (0.0-1.0)
    """
    if not query or not content:
        return 0.0

    return _single_direction_match(content, query, max_gap, max_discontinuities)[0]


def find_match_positions(
    content: str, query: str, max_gap=3, max_discontinuities=3, bidirectional_match=False
):
    """
    查找匹配位置信息

    Args:
        content: 目标文本
        query: 查询文本
        max_gap: 不连续位置的最大间隔
        max_discontinuities: 允许的最大不连续次数
        bidirectional_match: 是否进行双向匹配

    Returns:
        tuple: (score, start_pos, end_pos) 或
        dict: {'score': float, 'start': int, 'end': int, 'match_type': str}
    """
    if not query or not content:
        if bidirectional_match:
            return {'score': 0.0, 'start': -1, 'end': -1, 'match_type': 'no_match'}
        else:
            return 0.0, -1, -1

    # 进行正向匹配
    forward_score, forward_start, forward_end, forward_length = _single_direction_match(
        content, query, max_gap, max_discontinuities, False
    )

    if not bidirectional_match:
        return forward_score, forward_start, forward_end

    # 进行反向匹配
    reversed_content = content[::-1]
    reversed_query = query[::-1]
    reverse_score, reverse_start, reverse_end, reverse_length = _single_direction_match(
        reversed_content, reversed_query, max_gap, max_discontinuities, True, len(content)
    )

    # 选择end位置更靠后的结果
    if forward_end >= reverse_end:
        return {
            'score': forward_score,
            'start': forward_start,
            'end': forward_end,
            'match_type': 'forward',
            'match_length': forward_length,
        }
    else:
        return {
            'score': reverse_score,
            'start': reverse_start,
            'end': reverse_end,
            'match_type': 'reverse',
            'match_length': reverse_length,
        }


def _single_direction_match(
    content_str, query_str, max_gap, max_discontinuities, is_reversed=False, original_length=None
):
    """单方向匹配的核心逻辑"""
    # 如果是完全匹配，直接返回1.0和匹配位置
    if query_str in content_str:
        start_pos = content_str.find(query_str)
        end_pos = start_pos + len(query_str) - 1
        if is_reversed and original_length is not None:
            # 坐标转换回原始字符串
            original_end = original_length - start_pos - 1
            original_start = original_length - end_pos - 1
            return 1.0, original_start, original_end, len(query_str)
        else:
            return 1.0, start_pos, end_pos, len(query_str)

    # 构建query的部分匹配表
    match_table = build_partial_match_table(query_str)

    # 找到所有可能的起始匹配位置
    potential_starts = []
    potential_query_starts = []
    for i in range(len(content_str) - 2):
        tri_char = content_str[i : i + 3]
        if tri_char in match_table:
            for j in match_table[tri_char]:
                potential_starts.append(i)
                potential_query_starts.append(j)
                break

    # 从可能的起始位置开始匹配
    best_match_length = 0
    best_continuous_length = 0
    best_discontinuities = 0
    best_start_pos = -1
    best_end_pos = -1

    for start, query_start in zip(potential_starts, potential_query_starts):
        i = start
        j = query_start
        match_length = 0
        discontinuities = 0
        current_continuous_length = 0
        max_continuous_length = 0
        current_end_pos = -1

        while i < len(content_str) and j < len(query_str):
            if content_str[i] == query_str[j]:
                match_length += 1
                current_continuous_length += 1
                current_end_pos = i  # 记录当前匹配的结束位置
                i += 1
                j += 1
            else:
                # 更新最大连续匹配长度
                max_continuous_length = max(max_continuous_length, current_continuous_length)

                # 保存当前位置，以便策略失败时回退
                original_i = i
                original_j = j

                # 策略1: 在content中向前查找匹配当前query字符
                found_match = False
                next_query_char = query_str[original_j]
                for gap in range(1, max_gap + 1):
                    if (
                        original_i + gap < len(content_str)
                        and content_str[original_i + gap] == next_query_char
                    ):
                        match_length += 1
                        i = original_i + gap + 1
                        j = original_j + 1
                        discontinuities += 1
                        found_match = True
                        current_continuous_length = 1
                        current_end_pos = original_i + gap  # 更新结束位置
                        break

                if found_match:
                    continue

                # 策略2: 在query中向前查找匹配当前content字符
                next_content_char = content_str[original_i]
                for gap in range(1, max_gap + 1):
                    if (
                        original_j + gap < len(query_str)
                        and query_str[original_j + gap] == next_content_char
                    ):
                        match_length += 1
                        i = original_i + 1
                        j = original_j + gap + 1
                        discontinuities += 1
                        found_match = True
                        current_continuous_length = 1
                        current_end_pos = original_i  # 更新结束位置
                        break

                if found_match:
                    continue

                # 策略3: 如果前两种策略都失败，同时跳过当前字符
                if discontinuities < max_discontinuities:
                    i = original_i + 1
                    j = original_j + 1
                    discontinuities += 1
                    current_continuous_length = 0
                    continue

                # 如果所有策略都失败或不连续次数超过阈值，结束当前位置的匹配尝试
                break

        # 更新最后一次连续匹配长度
        max_continuous_length = max(max_continuous_length, current_continuous_length)

        # 如果当前匹配结果更好，更新最佳匹配
        if match_length > best_match_length:
            best_match_length = match_length
            best_continuous_length = max_continuous_length
            best_discontinuities = discontinuities
            best_start_pos = start
            best_end_pos = current_end_pos

    # 计算连续匹配比例和匹配比例
    match_ratio = (
        best_match_length / (len(query_str) + best_discontinuities)
        if best_match_length > 0
        else 0.0
    )

    if is_reversed and best_start_pos >= 0 and best_end_pos >= 0 and original_length is not None:
        # 坐标转换回原始字符串
        original_end = original_length - best_start_pos - 1
        original_start = original_length - best_end_pos - 1
        return match_ratio, original_start, original_end, best_match_length
    else:
        return match_ratio, best_start_pos, best_end_pos, best_match_length
