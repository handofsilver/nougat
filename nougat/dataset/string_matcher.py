def build_partial_match_table(pattern):
    """构建部分匹配表，用于快速找到可能的匹配起始位置"""
    table = {}
    for i, char in enumerate(pattern):
        if char not in table:
            table[char] = set()
        table[char].add(i)
    return table


def get_char_match_score(content, query):
    """
    计算 query 在 content 中的匹配度
    计算 query 在 content 中的最长匹配子序列的长度，要求匹配结果中不连续位置的间隔不能超过max_gap
    如果匹配结果中匹配位置不连续的次数不超过max_discontinuities，则认为匹配结果有效
    用匹配到的最长子序列长度除以 query 的长度，得到匹配度
    使用改进的算法，先找到可能的起始匹配位置，再进行详细匹配
    """
    if not query or not content:
        return 0.0

    # 如果是完全匹配，直接返回1.0
    if query in content:
        return 1.0

    max_gap = 3  # 不连续位置的最大间隔
    max_discontinuities = 3  # 允许的最大不连续次数

    # 构建query的部分匹配表
    match_table = build_partial_match_table(query)

    # 找到所有可能的起始匹配位置
    potential_starts = []
    for i, char in enumerate(content):
        if char in match_table and 0 in match_table[char]:
            potential_starts.append(i)

    # 从可能的起始位置开始匹配
    best_match_length = 0
    best_continuous_length = 0
    best_discontinuities = 0

    for start in potential_starts:
        i = start
        j = 0
        match_length = 0
        discontinuities = 0
        current_continuous_length = 0
        max_continuous_length = 0

        while i < len(content) and j < len(query):
            if content[i] == query[j]:
                match_length += 1
                current_continuous_length += 1
                i += 1
                j += 1
            else:
                # 更新最大连续匹配长度
                max_continuous_length = max(
                    max_continuous_length, current_continuous_length
                )

                # 尝试两种策略：
                # 1. 在content中向前查找匹配当前query字符
                found_in_content = False
                next_char = query[j]
                if next_char in match_table:
                    for gap in range(1, max_gap + 1):
                        if i + gap < len(content) and content[i + gap] == next_char:
                            match_length += 1
                            i = i + gap + 1
                            j += 1
                            discontinuities += 1
                            found_in_content = True
                            current_continuous_length = 1
                            break

                # 2. 如果策略1失败，尝试跳过query中的当前字符
                if not found_in_content and discontinuities < max_discontinuities:
                    j += 1
                    discontinuities += 1
                    current_continuous_length = 0
                    continue

                # 如果两种策略都失败或不连续次数超过阈值，结束当前位置的匹配尝试
                if not found_in_content:
                    break

                # 如果不连续次数超过阈值，结束当前位置的匹配尝试
                if discontinuities > max_discontinuities:
                    break

        # 更新最后一次连续匹配长度
        max_continuous_length = max(max_continuous_length, current_continuous_length)

        # 如果当前匹配结果更好，更新最佳匹配
        if match_length > best_match_length:
            best_match_length = match_length
            best_continuous_length = max_continuous_length
            best_discontinuities = discontinuities

    # 计算连续匹配比例
    continuous_ratio = best_continuous_length / len(query)
    match_ratio = best_match_length / (len(query) + best_discontinuities)
    # match_ratio = best_match_length / len(query)

    # 根据查询长度和连续匹配比例判断是否有效
    if len(query) > 12 and continuous_ratio < 0.35:
        return 0.0

    # print(best_match_length, len(query), best_discontinuities, continuous_ratio)
    return match_ratio


if __name__ == "__main__":
    content = "ωp=ω·vpn(31)"
    query = "ωp=⌊ω·vp"
    print(get_char_match_score(content, query))
