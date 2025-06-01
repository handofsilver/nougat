import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from nougat.dataset.split_utils.markdown_encoder import encode_formula_in_markdown


@dataclass
class ContentLine:
    """表示一行内容的数据结构"""

    line_index: int  # 在原始doc_lines中的行号
    local_index: int  # 在ordered或unordered中的局部索引
    tag_type: str  # 标签类型，如'TEXT', 'FIGURE_TITLE'等
    content_type: str  # 内容类型，如'ordered', 'unordered'
    content: str  # 行的文本内容
    is_tag_line: bool = False  # 是否是独立的标签行（如[TEXT]、[ENDTEXT]）

    def replace_content(self, content: str):
        return ContentLine(
            line_index=self.line_index,
            local_index=self.local_index,
            tag_type=self.tag_type,
            content_type=self.content_type,
            content=content,
            is_tag_line=self.is_tag_line,
        )


def separate_content_by_type(
    doc_lines: List[str], line_tag_map: Dict[int, Dict]
) -> Tuple[List[ContentLine], List[ContentLine]]:
    """
    将文档行分离为有序和无序两类

    Args:
        doc_lines: 文档行列表
        line_tag_map: 行标签映射，格式为 {行号: {'type': 标签类型, 'content_type': 'ordered'|'unordered', 'is_tag_line': bool}}

    Returns:
        Tuple[List[ContentLine], List[ContentLine]]: 分别返回有序和无序内容的纯文本列表
    """
    doc_lines = encode_formula_in_markdown(doc_lines)

    ordered_length, unordered_length = 0, 0
    ordered_text_lines, unordered_text_lines = [], []

    for line_index, line_content in enumerate(doc_lines):
        # 获取该行的标签信息
        tag_info = line_tag_map.get(line_index)

        if tag_info is None:
            # 如果没有标签信息，跳过该行（理论上不应该发生）
            continue

        local_index = ordered_length if tag_info['content_type'] == 'ordered' else unordered_length

        # 创建ContentLine对象
        content_line = ContentLine(
            line_index=line_index,
            local_index=local_index,
            tag_type=tag_info['type'],
            content_type=tag_info['content_type'],
            content=line_content.strip(),
            is_tag_line=tag_info.get('is_tag_line', False),
        )

        clean_lowercase_text = get_lowercase_text_for_matching(content_line)
        if not clean_lowercase_text:
            continue

        content_line = content_line.replace_content(clean_lowercase_text)

        # 根据content_type分类
        if tag_info['content_type'] == 'ordered':
            ordered_text_lines.append(content_line)
            ordered_length += 1
        else:
            unordered_text_lines.append(content_line)
            unordered_length += 1

    return ordered_text_lines, unordered_text_lines


def get_lowercase_text_for_matching(content_line: ContentLine) -> str:
    """
    从ContentLine中提取用于匹配的干净文本，转为小写

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
        return text.strip().lower()

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
        'FOOTNOTE': r'\[FOOTNOTE:.*?\](.*?)\[ENDFOOTNOTE\]',
    }

    # 获取对应的正则模式
    pattern = tag_patterns.get(tag_type)
    if not pattern:
        return text.strip().lower()

    # 提取标签内容
    match = re.search(pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

        # 特殊后处理
        if tag_type in ['TITLE', 'SUBTITLE']:
            text = re.sub(r'^#+\s*', '', text)  # 移除markdown标记

    return text.strip().lower()
