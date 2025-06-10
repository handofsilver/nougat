def build_partial_match_table(pattern):
    """构建部分匹配表，用于快速找到可能的匹配起始位置"""
    table = {}
    for i in range(len(pattern) - 2):
        tri_char = pattern[i : i + 3]
        if tri_char not in table:
            table[tri_char] = []
        table[tri_char].append(i)
    return table


def get_char_match_score(
    content: str,
    query: str,
    max_gap=3,  # 不连续位置的最大间隔
    max_discontinuities=3,  # 允许的最大不连续次数
    return_position=False,
    bidirectional_match=False,  # 是否进行双向匹配，选择更长的结果
):
    """
    计算 query 在 content 中的最长匹配子序列的长度，要求匹配结果中不连续位置的间隔不能超过max_gap;
    如果匹配结果中匹配位置不连续的次数不超过max_discontinuities，则认为匹配结果有效;
    用匹配到的最长子序列长度除以 query 的长度，得到匹配度;
    使用改进的算法，先找到可能的起始匹配位置，再进行详细匹配

    Args:
        bidirectional_match: 如果为True，会同时进行正向和反向匹配，选择匹配长度更长的结果

    返回值: (匹配分数, 开始位置, 结束位置) 或者包含匹配类型的dict
    """
    if not query or not content:
        if return_position:
            if bidirectional_match:
                return {'score': 0.0, 'start': -1, 'end': -1, 'match_type': 'no_match'}
            else:
                return 0.0, -1, -1
        else:
            return 0.0

    def _single_direction_match(content_str, query_str, is_reversed=False):
        """单方向匹配的核心逻辑"""
        # 如果是完全匹配，直接返回1.0和匹配位置
        if query_str in content_str:
            start_pos = content_str.find(query_str)
            end_pos = start_pos + len(query_str) - 1
            if is_reversed:
                # 坐标转换回原始字符串
                original_end = len(content) - start_pos - 1
                original_start = len(content) - end_pos - 1
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

        if is_reversed and best_start_pos >= 0 and best_end_pos >= 0:
            # 坐标转换回原始字符串
            original_end = len(content) - best_start_pos - 1
            original_start = len(content) - best_end_pos - 1
            return match_ratio, original_start, original_end, best_match_length
        else:
            return match_ratio, best_start_pos, best_end_pos, best_match_length

    # 进行正向匹配
    forward_score, forward_start, forward_end, forward_length = _single_direction_match(
        content, query, False
    )

    if not bidirectional_match:
        # 不进行双向匹配，返回正向结果
        if return_position:
            return forward_score, forward_start, forward_end
        else:
            return forward_score

    # 进行反向匹配
    reversed_content = content[::-1]
    reversed_query = query[::-1]
    reverse_score, reverse_start, reverse_end, reverse_length = _single_direction_match(
        reversed_content, reversed_query, True
    )

    # 选择end位置更靠后的结果
    if forward_end >= reverse_end:
        best_score = forward_score
        best_start = forward_start
        best_end = forward_end
        match_type = 'forward'
        match_length = forward_length
    else:
        best_score = reverse_score
        best_start = reverse_start
        best_end = reverse_end
        match_type = 'reverse'
        match_length = reverse_length

    if return_position:
        return {
            'score': best_score,
            'start': best_start,
            'end': best_end,
            'match_type': match_type,
            'match_length': match_length,
        }
    else:
        return best_score
