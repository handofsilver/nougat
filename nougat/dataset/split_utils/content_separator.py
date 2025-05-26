from typing import List, Dict, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class ContentLine:
    """表示一行内容的数据结构"""

    line_index: int  # 在原始doc_lines中的行号
    tag_type: str  # 标签类型，如'TEXT', 'FIGURE_TITLE'等
    content: str  # 行的文本内容
    is_tag_line: bool = False  # 是否是独立的标签行（如[TEXT]、[ENDTEXT]）


class ContentSeparationResult(NamedTuple):
    """内容分离结果"""

    ordered_lines: List[ContentLine]  # 有序内容行列表
    unordered_lines: List[ContentLine]  # 无序内容行列表
    ordered_text_lines: List[str]  # 有序内容的纯文本列表（用于构建倒排索引）
    unordered_text_lines: List[str]  # 无序内容的纯文本列表（用于构建倒排索引）


def separate_content_by_type(
    doc_lines: List[str], line_tag_map: Dict[int, Dict]
) -> ContentSeparationResult:
    """
    将文档行分离为有序和无序两类

    Args:
        doc_lines: 文档行列表
        line_tag_map: 行标签映射，格式为 {行号: {'type': 标签类型, 'content_type': 'ordered'|'unordered', 'is_tag_line': bool}}

    Returns:
        ContentSeparationResult: 包含分离后的有序和无序内容
    """
    ordered_lines = []
    unordered_lines = []
    ordered_text_lines = []
    unordered_text_lines = []

    for line_index, line_content in enumerate(doc_lines):
        # 获取该行的标签信息
        tag_info = line_tag_map.get(line_index)

        if tag_info is None:
            # 如果没有标签信息，跳过该行（理论上不应该发生）
            continue

        # 创建ContentLine对象
        content_line = ContentLine(
            line_index=line_index,
            tag_type=tag_info['type'],
            content=line_content,
            is_tag_line=tag_info.get('is_tag_line', False),
        )

        # 根据content_type分类
        if tag_info['content_type'] == 'ordered':
            ordered_lines.append(content_line)
            # 只有非标签行才加入文本列表（用于倒排索引）
            if not content_line.is_tag_line:
                # 清理标签后加入文本列表
                clean_text = extract_clean_text_from_content_line(content_line)
                if clean_text:  # 只有非空文本才加入
                    ordered_text_lines.append(clean_text)
        else:  # 'unordered'
            unordered_lines.append(content_line)
            # 清理标签后加入文本列表
            clean_text = extract_clean_text_from_content_line(content_line)
            if clean_text:  # 只有非空文本才加入
                unordered_text_lines.append(clean_text)

    return ContentSeparationResult(
        ordered_lines=ordered_lines,
        unordered_lines=unordered_lines,
        ordered_text_lines=ordered_text_lines,
        unordered_text_lines=unordered_text_lines,
    )


def extract_clean_text_from_content_line(content_line: ContentLine) -> str:
    """
    从ContentLine中提取用于匹配的干净文本

    Args:
        content_line: 内容行对象

    Returns:
        str: 清理后的文本内容
    """
    # 如果是标签行，返回空字符串（不参与匹配）
    if content_line.is_tag_line:
        return ""

    text = content_line.content
    tag_type = content_line.tag_type

    # 对于TEXT类型，直接返回内容（已经是纯文本）
    if tag_type == 'TEXT':
        return text.strip()

    import re

    # 定义标签清理规则
    tag_patterns = {
        # 标准标签格式：[TAG]content[ENDTAG]
        'TITLE': rf'\[{tag_type}\](.*?)\[END{tag_type}\]',
        'AUTHOR': rf'\[{tag_type}\](.*?)\[END{tag_type}\]',
        'SUBTITLE': rf'\[{tag_type}\](.*?)\[END{tag_type}\]',
        'FORMULA': r'\[FORMULA\](.*?)\[ENDFORMULA\]',
        'FIGURE_TITLE': rf'\[{tag_type}\](.*?)\[END{tag_type}\]',
        'TABLE_TITLE': rf'\[{tag_type}\](.*?)\[END{tag_type}\]',
        'ALGORITHM_TITLE': rf'\[{tag_type}\](.*?)\[END{tag_type}\]',
        'THANK_NOTE': r'\[THANK_NOTE\](.*?)\[ENDTHANK_NOTE\]',
        # 特殊格式：脚注
        'FOOTNOTE': r'\[FOOTNOTE:.*?\](.*?)\[ENDFOOTNOTE\]',
    }

    # 获取对应的正则模式
    pattern = tag_patterns.get(tag_type)
    if not pattern:
        return text.strip()

    # 提取标签内容
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

        # 特殊后处理
        if tag_type in ['TITLE', 'SUBTITLE']:
            # 移除markdown标记
            text = re.sub(r'^#+\s*', '', text)
        elif tag_type == 'FOOTNOTE':
            # 移除"Footnote X:"前缀
            text = re.sub(r'^Footnote\s+\d+:\s*', '', text)

    return text.strip()


def build_ordered_lines_with_pointer_info(
    ordered_lines: List[ContentLine],
) -> List[Tuple[int, ContentLine]]:
    """
    为有序内容行构建带指针信息的列表，用于后续的顺序匹配

    Args:
        ordered_lines: 有序内容行列表

    Returns:
        List[Tuple[int, ContentLine]]: [(在有序列表中的索引, ContentLine对象), ...]
    """
    result = []
    for i, content_line in enumerate(ordered_lines):
        # 只有非标签行才参与匹配
        if not content_line.is_tag_line:
            result.append((i, content_line))
    return result


def get_content_statistics(separation_result: ContentSeparationResult) -> Dict[str, int]:
    """
    获取内容分离的统计信息

    Args:
        separation_result: 内容分离结果

    Returns:
        Dict[str, int]: 统计信息
    """
    stats = {
        'total_ordered_lines': len(separation_result.ordered_lines),
        'total_unordered_lines': len(separation_result.unordered_lines),
        'ordered_text_lines': len(separation_result.ordered_text_lines),
        'unordered_text_lines': len(separation_result.unordered_text_lines),
    }

    # 统计各种标签类型的数量
    ordered_tag_counts = {}
    unordered_tag_counts = {}

    for line in separation_result.ordered_lines:
        tag_type = line.tag_type
        ordered_tag_counts[tag_type] = ordered_tag_counts.get(tag_type, 0) + 1

    for line in separation_result.unordered_lines:
        tag_type = line.tag_type
        unordered_tag_counts[tag_type] = unordered_tag_counts.get(tag_type, 0) + 1

    stats['ordered_tag_counts'] = ordered_tag_counts
    stats['unordered_tag_counts'] = unordered_tag_counts

    return stats


if __name__ == "__main__":
    # 测试代码
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from markdown_parser import parse_markdown_lines
    import json

    # 测试文件路径
    test_mmd = "/home/ninziwei/lyj/nougat/__test_0/markdown/2303.00058.mmd"

    if os.path.exists(test_mmd):
        print("🔍 开始测试内容分离功能...")

        # 读取测试文件
        with open(test_mmd, "r", encoding="utf-8") as f:
            mmd_text = f.read()

        # 解析markdown
        doc_lines, text_obj_map, line_tag_map = parse_markdown_lines(mmd_text)

        # 执行内容分离
        separation_result = separate_content_by_type(doc_lines, line_tag_map)

        # 获取统计信息
        stats = get_content_statistics(separation_result)

        print(f"📊 分离统计:")
        print(f"  总有序行数: {stats['total_ordered_lines']}")
        print(f"  总无序行数: {stats['total_unordered_lines']}")
        print(f"  有序文本行数: {stats['ordered_text_lines']}")
        print(f"  无序文本行数: {stats['unordered_text_lines']}")

        print(f"\n📋 有序标签统计: {stats['ordered_tag_counts']}")
        print(f"📋 无序标签统计: {stats['unordered_tag_counts']}")

        # 保存分离结果示例
        print(f"\n💾 保存分离结果示例...")

        # 保存前5个有序行和前5个无序行作为示例
        sample_data = {
            "ordered_sample": [
                {
                    "line_index": line.line_index,
                    "tag_type": line.tag_type,
                    "content": (
                        line.content[:100] + "..." if len(line.content) > 100 else line.content
                    ),
                    "is_tag_line": line.is_tag_line,
                }
                for line in separation_result.ordered_lines[:5]
            ],
            "unordered_sample": [
                {
                    "line_index": line.line_index,
                    "tag_type": line.tag_type,
                    "content": (
                        line.content[:100] + "..." if len(line.content) > 100 else line.content
                    ),
                    "is_tag_line": line.is_tag_line,
                }
                for line in separation_result.unordered_lines[:5]
            ],
            "statistics": stats,
        }

        with open("content_separation_test.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print("✅ 测试完成！结果已保存到 content_separation_test.json")

    else:
        print(f"❌ 测试文件不存在: {test_mmd}")
