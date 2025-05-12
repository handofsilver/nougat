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
    table={},
    max_gap=3,  # 不连续位置的最大间隔
    max_discontinuities=3,  # 允许的最大不连续次数
    return_position=False,
    debug=False,
):
    """
    计算 query 在 content 中的最长匹配子序列的长度，要求匹配结果中不连续位置的间隔不能超过max_gap;
    如果匹配结果中匹配位置不连续的次数不超过max_discontinuities，则认为匹配结果有效;
    用匹配到的最长子序列长度除以 query 的长度，得到匹配度;
    使用改进的算法，先找到可能的起始匹配位置，再进行详细匹配
    返回值: (匹配分数, 开始位置, 结束位置)
    """
    if not query or not content:
        if return_position:
            return 0.0, -1, -1
        else:
            return 0.0

    # 如果是完全匹配，直接返回1.0和匹配位置
    if query in content:
        start_pos = content.find(query)
        end_pos = start_pos + len(query) - 1
        if return_position:
            return 1.0, start_pos, end_pos
        else:
            return 1.0

    # 构建query的部分匹配表
    if table:
        match_table = table
    else:
        match_table = build_partial_match_table(query)

    # 找到所有可能的起始匹配位置
    potential_starts = []
    potential_query_starts = []
    for i in range(len(content) - 2):
        tri_char = content[i : i + 3]
        if tri_char in match_table:
            for j in range(8):
                if j in match_table[tri_char]:
                    potential_starts.append(i)
                    potential_query_starts.append(j)
                    break
    if debug:
        print(37, len(potential_starts))
        print(potential_starts)
        print(potential_query_starts)

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

        while i < len(content) and j < len(query):
            if content[i] == query[j]:
                match_length += 1
                current_continuous_length += 1
                current_end_pos = i  # 记录当前匹配的结束位置
                i += 1
                j += 1
            else:
                # 更新最大连续匹配长度
                max_continuous_length = max(
                    max_continuous_length, current_continuous_length
                )

                # 保存当前位置，以便策略失败时回退
                original_i = i
                original_j = j

                # 策略1: 在content中向前查找匹配当前query字符
                found_match = False
                next_query_char = query[original_j]
                for gap in range(1, max_gap + 1):
                    if (
                        original_i + gap < len(content)
                        and content[original_i + gap] == next_query_char
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
                next_content_char = content[original_i]
                for gap in range(1, max_gap + 1):
                    if (
                        original_j + gap < len(query)
                        and query[original_j + gap] == next_content_char
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

    # 计算连续匹配比例
    continuous_ratio = best_continuous_length / len(query)
    match_ratio = best_match_length / (len(query) + best_discontinuities)

    if debug:
        print("最佳匹配位置:", best_start_pos, best_end_pos)
        if best_start_pos >= 0 and best_end_pos >= 0:
            print("匹配内容:", content[best_start_pos : best_end_pos + 1])

    if return_position:
        return match_ratio, best_start_pos, best_end_pos
    else:
        return match_ratio


if __name__ == "__main__":
    content = "ωp=ω·vpn(31)"
    query = "ωp=⌊ω·vp"

    content = "weintroduceanewmethodbasedonnonnegativematrixfactorization,neuralnmf,fordetectinglatenthierarchicalstructur"
    query = ".weintroduceaneswmethodbasedonnonnegativematrixfactorization,neuralnmf,for"
    content = "{(l)},suchas|z⊙(y-bs(l))|,whichwillinfluencethelearnedaandsmatricesviabackpropagation."
    query = "@(y−bs(l))∥,whichwillinfluencethelearnedaandsmatricesviabackpropagation."
    print(get_char_match_score(content, query, debug=True))
