import re
from typing import List, Dict, Tuple
import json


def reorder_ieee_copyright(doc: str) -> str:
    """
    重新排序IEEE版权声明
    查找包含'IEEE Copyright Notice'和'Personal use of this material is permitted.'的[TEXT]标签对，
    将其移动到文档最前面
    """
    ieee_copyright_pattern = r'\[TEXT\](.*?)\[END_TEXT\]'
    ieee_copyright_block = None

    # 查找包含IEEE版权信息的TEXT标签对
    for match in re.finditer(ieee_copyright_pattern, doc, re.DOTALL):
        content = match.group(1)
        if (
            'IEEE Copyright Notice' in content
            and 'Personal use of this material is permitted' in content
        ):
            ieee_copyright_block = match.group(0)  # 完整的[TEXT]...[END_TEXT]块
            break

    if ieee_copyright_block:
        # 从原位置移除IEEE版权块
        doc_without_copyright = re.sub(re.escape(ieee_copyright_block), '', doc, count=1)
        # 将IEEE版权块移到文档最前面
        doc = ieee_copyright_block + '\n' + doc_without_copyright

    return doc


def preprocess_author_tag(doc: str) -> str:
    """
    预处理作者信息和致谢注释：
    将[AUTHOR]标签中的嵌套标签（目前仅发现[FOOTNOTE]标签）移出，并将[AUTHOR]标签内容压缩为一行
    支持处理多个FOOTNOTE标签
    """
    author_pattern = r'\[AUTHOR\](.*?)\[END_AUTHOR\]'
    author_match = re.search(author_pattern, doc, re.DOTALL)

    if not author_match:
        return doc

    author_full_content = author_match.group(1)

    # 检查AUTHOR中是否包含FOOTNOTE标签
    footnote_pattern = r'\[FOOTNOTE:.*?\].*?\[END_FOOTNOTE\]'
    footnote_matches = list(re.finditer(footnote_pattern, author_full_content, re.DOTALL))

    if footnote_matches:
        # AUTHOR中有FOOTNOTE，需要提取出来
        # 收集所有footnote内容
        footnote_contents = []
        for match in footnote_matches:
            footnote_contents.append(match.group(0))
        
        # 从AUTHOR内容中移除所有FOOTNOTE
        author_clean_content = re.sub(footnote_pattern, '', author_full_content, flags=re.DOTALL)

        # 压缩AUTHOR内容为一行
        author_clean_content = ' '.join(
            line.strip() for line in author_clean_content.split('\n') if line.strip()
        )

        # 替换整个AUTHOR部分，将所有FOOTNOTE放到外面
        all_footnotes = '\n'.join(footnote_contents)
        replacement = f'[AUTHOR]{author_clean_content}[END_AUTHOR]\n{all_footnotes}\n'
        doc = re.sub(author_pattern, replacement, doc, flags=re.DOTALL)
    else:
        # AUTHOR中没有其他标签，只压缩换行
        author_clean_content = ' '.join(
            line.strip() for line in author_full_content.split('\n') if line.strip()
        )

        # 替换AUTHOR内容
        replacement = f'[AUTHOR]{author_clean_content}[END_AUTHOR]'
        doc = re.sub(author_pattern, replacement, doc, flags=re.DOTALL)

    return doc

def remove_formatting_commands(doc: str) -> str:
    """
    删除排版格式命令
    """
    doc = re.sub(r'\\leavevmode', '', doc)
    doc = re.sub(r'\\nobreak', '', doc)
    doc = re.sub(r'\\pagebreak', '', doc)
    doc = re.sub(r'\\nopagebreak', '', doc)
    doc = re.sub(r'\\enlargethispage\{.*?\}', '', doc)
    doc = re.sub(r'\\leavevmode\\nobreak\\', '', doc)
    return doc

def parse_markdown_lines(doc: str) -> Tuple[List[str], Dict[str, str], Dict[int, str]]:
    """
    解析Markdown文档并返回行列表和映射信息

    Returns:
        Tuple[List[str], Dict[str, str], Dict[int, str]]:
            - 文档的行列表
            - 标题到对象的映射
            - 行标签映射
    """
    # 1. 预处理：处理作者信息标签
    doc = preprocess_author_tag(doc)

    # 2. 预处理：展平嵌套标签
    doc = flatten_nested_text_tag(doc)

    # 3. 删除排版格式命令
    doc = remove_formatting_commands(doc)

    # 4. 重新排序IEEE版权声明
    doc = reorder_ieee_copyright(doc)

    # 5. 构建text_to_object映射
    text_obj_map = build_tfa_text_to_object(doc)

    # 6. 将TFA标签替换为其标题
    doc = replace_tfa_with_titles(doc)

    # 7. 获取文档行
    doc = re.sub(r'\[TEXT\]|\[END_TEXT\]', '', doc)
    doc_lines = doc.split("\n")
    doc_lines = [line.strip() for line in doc_lines if line.strip()]

    # 8. 构建行标签映射
    line_tag_map = build_line_tag_mapping(doc_lines)

    return doc_lines, text_obj_map, line_tag_map


def flatten_nested_text_tag(doc: str) -> str:
    """
    只保留最外层的[TEXT]...[END_TEXT]，去除内部所有嵌套的[TEXT]和[END_TEXT]标签。
    """
    result = []
    stack = []
    i = 0
    n = len(doc)
    last_pos = 0

    text_tag_length = len('[TEXT]')
    end_text_tag_length = len('[END_TEXT]')

    while i < n:
        if doc.startswith('[TEXT]', i):
            if not stack:
                # 记录最外层[TEXT]前的内容
                result.append(doc[last_pos:i])
                start_outer = i
            stack.append(i)
            i += text_tag_length
        elif doc.startswith('[END_TEXT]', i):
            if stack:
                start = stack.pop()
                if not stack:
                    # 这是最外层的[TEXT]...[END_TEXT]
                    content = doc[start_outer + text_tag_length : i]
                    # 去掉内部所有[TEXT]和[END_TEXT]
                    content = re.sub(r'\[TEXT\]|\[END_TEXT\]', '', content)
                    result.append('[TEXT]' + content + '[END_TEXT]')
                    last_pos = i + end_text_tag_length
            i += end_text_tag_length
        else:
            i += 1
    # 添加最后一段内容
    result.append(doc[last_pos:])
    return ''.join(result)


def build_tfa_text_to_object(doc: str) -> Dict[str, str]:
    """
    构建TFA标题内容到完整对象内容的映射
    例如：
    {
        '图1的标题': '[FIGURE]...[END_FIGURE]完整内容',
        '表1的标题': '[TABLE]...[END_TABLE]完整内容',
        '算法1的标题': '[ALGORITHM]...[END_ALGORITHM]完整内容'
    }
    """
    doc = re.sub(r'\n\n+', '\n', doc)
    doc = re.sub(r'\[TEXT\]|\[END_TEXT\]', '', doc) # 删除[TEXT]和[END_TEXT]标签

    text_to_obj = {}
    
    def filter_figure_content(content: str) -> str:
        """
        过滤图片内容，只保留图片标题和图片内容
        """
        lines = content.split('\n')
        filtered_lines = []
        
        for line in lines:
            if line.startswith('[FIGURE'):
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    # 处理图片
    figure_matches = re.finditer(r"\[FIGURE:.*?\](.*?)\[END_FIGURE\]", doc, re.DOTALL)
    for match in figure_matches:
        full_content = re.sub(
            r'\[([A-Z]+):[^\]]*\]', r'[\1]', match.group(0)
        )  # 新增：将[TAG:*]替换为[TAG]
        full_content = filter_figure_content(full_content)
        title_match = re.search(
            r"\[FIGURE_TITLE\](.*?)\[END_FIGURE_TITLE\]", match.group(1), re.DOTALL
        )
        if title_match:
            title = title_match.group(0).strip()
            text_to_obj[title] = full_content

    # 处理表格
    table_matches = re.finditer(r"\[TABLE:.*?\](.*?)\[END_TABLE\]", doc, re.DOTALL)
    for match in table_matches:
        full_content = re.sub(
            r'\[([A-Z]+):[^\]]*\]', r'[\1]', match.group(0)
        )  # 新增：将[TAG:*]替换为[TAG]
        title_match = re.search(
            r"\[TABLE_TITLE\](.*?)\[END_TABLE_TITLE\]", match.group(1), re.DOTALL
        )
        if title_match:
            title = title_match.group(0).strip()
            text_to_obj[title] = full_content

    # 处理算法
    algo_matches = re.finditer(r"\[ALGORITHM\](.*?)\[END_ALGORITHM\]", doc, re.DOTALL)
    for match in algo_matches:
        full_content = match.group(0)
        title_match = re.search(
            r"\[ALGORITHM_TITLE\](.*?)\[END_ALGORITHM_TITLE\]", match.group(1), re.DOTALL
        )
        if title_match:
            title = title_match.group(0).strip()
            text_to_obj[title] = full_content

    return text_to_obj


def build_line_tag_mapping(doc_lines: List[str]) -> Dict[int, Dict]:
    """
    构建行号到标签类型的映射。每行只可能是以下三种情况之一：
    1. 带标签的内容行：[TAG]内容[END_TAG]
    2. [TEXT]和[END_TEXT]之间的正文行

    Args:
        doc_lines: 已经预处理过的文档行列表（已去除空行和[TEXT]、[END_TEXT]）

    Returns:
        Dict[int, Dict]: {
            行号: {
                'type': 标签类型,
                'content_type': 'ordered'|'unordered'
            }
        }
    """
    line_tag_map = {}

    # 定义标签类型和它们的属性
    tag_types = {
        'TEXT': {'content_type': 'ordered'},
        'TITLE': {'content_type': 'ordered'},
        'AUTHOR': {'content_type': 'ordered'},
        'SUBTITLE': {'content_type': 'ordered'},
        'FORMULA': {'content_type': 'ordered'},
        'FIGURE_TITLE': {'content_type': 'unordered'},
        'TABLE_TITLE': {'content_type': 'unordered'},
        'ALGORITHM_TITLE': {'content_type': 'unordered'},
        'FOOTNOTE': {'content_type': 'unordered'},
        # 'THANK_NOTE': {'content_type': 'unordered'},
    }

    for line_num, line in enumerate(doc_lines):
        line = line.strip()

        # 处理其他标签行
        for tag, properties in tag_types.items():
            if line.startswith(f'[{tag}') and line.endswith(f'[END_{tag}]'):
                line_tag_map[line_num] = {'type': tag, 'content_type': properties['content_type']}
                break

        # 如果该行不是其他标签行，则为正文行
        if line_num not in line_tag_map:
            line_tag_map[line_num] = {'type': 'TEXT', 'content_type': 'ordered'}

    return line_tag_map


def replace_tfa_with_titles(doc: str) -> str:
    """
    将TFA(Table/Figure/Algorithm)标签替换为其标题内容

    Args:
        doc (str): 原始文档内容

    Returns:
        str: 替换后的文档内容
    """
    doc = re.sub(
        r"\[(FIGURE:.*?|TABLE:.*?|ALGORITHM)\](.*?)\[END_(FIGURE|TABLE|ALGORITHM)\]",
        lambda m: (
            re.search(
                r"\[(FIGURE|TABLE|ALGORITHM)_TITLE\](.*?)\[END_(FIGURE|TABLE|ALGORITHM)_TITLE\]",
                m.group(2),
                re.DOTALL,
            )
            .group(0)
            .strip()
            if re.search(
                r"\[(FIGURE|TABLE|ALGORITHM)_TITLE\](.*?)\[END_(FIGURE|TABLE|ALGORITHM)_TITLE\]",
                m.group(2),
                re.DOTALL,
            )
            else ""
        ),
        doc,
        flags=re.DOTALL,
    )

    return doc
