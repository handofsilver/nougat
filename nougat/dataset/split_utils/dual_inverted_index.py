from typing import List, Dict, Tuple
from nougat.dataset.split_utils.content_separator import ContentSeparationResult, ContentLine
from nougat.dataset.split_utils.line_matcher import build_inverted_index


class DualInvertedIndex:
    """双重倒排索引类"""

    def __init__(
        self,
        ordered_index: Dict[str, List[int]],
        unordered_index: Dict[str, List[int]],
        ordered_lines: List[ContentLine],
        unordered_lines: List[ContentLine],
    ):
        self.ordered_index = ordered_index  # 有序内容的倒排索引
        self.unordered_index = unordered_index  # 无序内容的倒排索引
        self.ordered_lines = ordered_lines  # 有序内容行列表
        self.unordered_lines = unordered_lines  # 无序内容行列表

    def get_ordered_candidates(self, words: List[str], max_words: int = 5) -> List[int]:
        """
        获取有序内容的候选行索引

        Args:
            words: 查询词列表
            max_words: 最大使用词数

        Returns:
            List[int]: 候选行在ordered_text_lines中的索引列表
        """
        candidates = []
        word_count = 0

        for word in words:
            if word in self.ordered_index:
                candidates.extend(self.ordered_index[word])
                word_count += 1
            if word_count >= max_words:
                break

        return list(set(candidates))

    def get_unordered_candidates(self, words: List[str], max_words: int = 5) -> List[int]:
        """
        获取无序内容的候选行索引

        Args:
            words: 查询词列表
            max_words: 最大使用词数

        Returns:
            List[int]: 候选行在unordered_text_lines中的索引列表
        """
        candidates = []
        word_count = 0

        for word in words:
            if word in self.unordered_index:
                candidates.extend(self.unordered_index[word])
                word_count += 1
            if word_count >= max_words:
                break

        return list(set(candidates))

    def get_statistics(self) -> Dict[str, int]:
        """获取索引统计信息"""
        return {
            'ordered_vocab_size': len(self.ordered_index),
            'unordered_vocab_size': len(self.unordered_index),
            'ordered_lines_count': len(self.ordered_lines),
            'unordered_lines_count': len(self.unordered_lines),
            'total_ordered_entries': sum(len(indices) for indices in self.ordered_index.values()),
            'total_unordered_entries': sum(
                len(indices) for indices in self.unordered_index.values()
            ),
        }


def build_dual_inverted_index(separation_result: ContentSeparationResult) -> DualInvertedIndex:
    """
    为有序和无序内容分别构建倒排索引

    Args:
        separation_result: 内容分离结果

    Returns:
        DualInvertedIndex: 双重倒排索引对象
    """
    # 构建有序内容的倒排索引
    ordered_index = build_inverted_index(separation_result.ordered_text_lines)

    # 构建无序内容的倒排索引
    unordered_index = build_inverted_index(separation_result.unordered_text_lines)

    with open("ordered_index.json", "w", encoding="utf-8") as f:
        json.dump(ordered_index, f, ensure_ascii=False, indent=2)
    with open("unordered_index.json", "w", encoding="utf-8") as f:
        json.dump(unordered_index, f, ensure_ascii=False, indent=2)

    return DualInvertedIndex(
        ordered_index=ordered_index,
        unordered_index=unordered_index,
        ordered_lines=separation_result.ordered_lines,
        unordered_lines=separation_result.unordered_lines,
    )


def filter_ordered_candidates_by_pointer(
    candidates: List[int], current_pointer: int, window_before: int = 3, window_after: int = 5
) -> List[int]:
    """
    根据当前指针位置过滤有序内容的候选项

    Args:
        candidates: 候选行索引列表
        current_pointer: 当前指针位置
        window_before: 向前搜索窗口大小
        window_after: 向后搜索窗口大小

    Returns:
        List[int]: 过滤后的候选行索引列表
    """
    start = max(0, current_pointer - window_before)
    end = current_pointer + window_after

    return [idx for idx in candidates if start <= idx <= end]


def create_ordered_line_mapping(ordered_lines: List[ContentLine]) -> Dict[int, int]:
    """
    创建有序内容行的映射：从text_lines索引到ordered_lines索引

    Args:
        ordered_lines: 有序内容行列表

    Returns:
        Dict[int, int]: {text_lines中的索引: ordered_lines中的索引}
    """
    mapping = {}
    text_line_index = 0

    for ordered_index, line in enumerate(ordered_lines):
        if not line.is_tag_line:  # 只有非标签行才在text_lines中
            mapping[text_line_index] = ordered_index
            text_line_index += 1

    return mapping


if __name__ == "__main__":
    # 测试代码
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from markdown_parser import parse_markdown_lines
    from content_separator import separate_content_by_type
    import json

    # 测试文件路径
    test_mmd = "/home/ninziwei/lyj/nougat/__test_0/markdown/2303.00058.mmd"

    if os.path.exists(test_mmd):
        print("🔍 开始测试双重倒排索引功能...")

        # 读取测试文件
        with open(test_mmd, "r", encoding="utf-8") as f:
            mmd_text = f.read()

        # 解析markdown
        doc_lines, text_obj_map, line_tag_map = parse_markdown_lines(mmd_text)

        # 执行内容分离
        separation_result = separate_content_by_type(doc_lines, line_tag_map)

        # 构建双重倒排索引
        dual_index = build_dual_inverted_index(separation_result)

        # 获取统计信息
        stats = dual_index.get_statistics()

        print(f"📊 倒排索引统计:")
        print(f"  有序词汇表大小: {stats['ordered_vocab_size']}")
        print(f"  无序词汇表大小: {stats['unordered_vocab_size']}")
        print(f"  有序行数: {stats['ordered_lines_count']}")
        print(f"  无序行数: {stats['unordered_lines_count']}")
        print(f"  有序索引条目总数: {stats['total_ordered_entries']}")
        print(f"  无序索引条目总数: {stats['total_unordered_entries']}")

        # 测试查询功能
        print(f"\n🔍 测试查询功能:")
        test_words = ["neural", "matrix", "factorization"]

        ordered_candidates = dual_index.get_ordered_candidates(test_words)
        unordered_candidates = dual_index.get_unordered_candidates(test_words)

        print(f"  查询词: {test_words}")
        print(f"  有序内容候选数: {len(ordered_candidates)}")
        print(f"  无序内容候选数: {len(unordered_candidates)}")

        # 显示前几个候选结果
        if ordered_candidates:
            print(f"  有序候选示例 (前3个): {ordered_candidates[:3]}")
        if unordered_candidates:
            print(f"  无序候选示例 (前3个): {unordered_candidates[:3]}")

        # 测试指针过滤
        if ordered_candidates:
            filtered = filter_ordered_candidates_by_pointer(ordered_candidates, 10)
            print(f"  指针过滤后 (pointer=10): {len(filtered)} 个候选")

        # 保存测试结果
        test_result = {
            "statistics": stats,
            "test_query": {
                "words": test_words,
                "ordered_candidates_count": len(ordered_candidates),
                "unordered_candidates_count": len(unordered_candidates),
                "ordered_sample": ordered_candidates[:5],
                "unordered_sample": unordered_candidates[:5],
            },
        }

        with open("dual_index_test.json", "w", encoding="utf-8") as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)

        print("✅ 测试完成！结果已保存到 dual_index_test.json")

    else:
        print(f"❌ 测试文件不存在: {test_mmd}")
