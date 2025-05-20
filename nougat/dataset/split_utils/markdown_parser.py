import re
from typing import List, Dict, Tuple
from nougat.dataset.split_utils.markdown_encoder import encode_formula_in_markdown
import json


def preprocess_author_and_thank_note(doc: str) -> str:
    """
    预处理作者信息和致谢注释：
    1. 提取[THANK_NOTE]标签对
    2. 将其移动到[ENDAUTHOR_INFO]后
    3. 压缩[AUTHOR_INFO]中的换行
    """
    # 提取[THANK_NOTE]内容
    thank_note_pattern = r'\[THANK_NOTE\].*?\[ENDTHANK_NOTE\]'
    thank_note = re.search(thank_note_pattern, doc, re.DOTALL)
    thank_note_content = thank_note.group(0) if thank_note else ''

    # 从[AUTHOR_INFO]中移除[THANK_NOTE]内容并压缩换行
    author_pattern = r'\[AUTHOR\](.*?)\[ENDAUTHOR\]'
    if re.search(author_pattern, doc, re.DOTALL):
        author_content = re.search(author_pattern, doc, re.DOTALL).group(1)
        author_content = re.sub(thank_note_pattern, '', author_content, flags=re.DOTALL)
        author_content = ' '.join(
            line.strip() for line in author_content.split('\n') if line.strip()
        )
    else:
        author_content = ''

    # 替换原文中的[AUTHOR_INFO]部分
    doc = re.sub(
        author_pattern,
        f'[AUTHOR]{author_content}[ENDAUTHOR]\n{thank_note_content}',
        doc,
        flags=re.DOTALL,
    )

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
    # 1. 预处理：处理作者信息和致谢注释
    doc = preprocess_author_and_thank_note(doc)

    # 2. 预处理：展平嵌套标签，编码公式
    doc = flatten_nested_text_tag(doc)
    doc = encode_formula_in_markdown(doc)

    # 3. 构建text_to_object映射
    text_obj_map = build_text_to_object(doc)
    # with open("text_obj_map.json", "w", encoding="utf-8") as f:
    #     json.dump(text_obj_map, f, ensure_ascii=False, indent=4)

    # 4. 将TFA标签替换为其标题
    doc = replace_tfa_with_titles(doc)

    # 5. 获取文档行
    doc_lines = doc.split("\n")
    doc_lines = [line.strip() for line in doc_lines if line.strip()]

    # 6. 构建行标签映射
    line_tag_map = build_line_tag_mapping(doc_lines)

    return doc_lines, text_obj_map, line_tag_map


def flatten_nested_text_tag(doc: str) -> str:
    """
    只保留最外层的[TEXT]...[ENDTEXT]，去除内部所有嵌套的[TEXT]和[ENDTEXT]标签。
    """
    result = []
    stack = []
    i = 0
    n = len(doc)
    last_pos = 0
    while i < n:
        if doc.startswith('[TEXT]', i):
            if not stack:
                # 记录最外层[TEXT]前的内容
                result.append(doc[last_pos:i])
                start_outer = i
            stack.append(i)
            i += 6
        elif doc.startswith('[ENDTEXT]', i):
            if stack:
                start = stack.pop()
                if not stack:
                    # 这是最外层的[TEXT]...[ENDTEXT]
                    content = doc[start_outer + 6 : i]
                    # 去掉内部所有[TEXT]和[ENDTEXT]
                    content = re.sub(r'\[TEXT\]|\[ENDTEXT\]', '', content)
                    result.append('[TEXT]' + content + '[ENDTEXT]')
                    last_pos = i + 9
            i += 9
        else:
            i += 1
    # 添加最后一段内容
    result.append(doc[last_pos:])
    return ''.join(result)


def build_text_to_object(doc: str) -> Dict[str, str]:
    """
    构建标题内容到完整对象内容的映射
    例如：
    {
        '图1的标题': '[FIGURE]...[ENDFIGURE]完整内容',
        '表1的标题': '[TABLE]...[ENDTABLE]完整内容'
    }
    """
    text_to_obj = {}

    # 处理图片
    figure_matches = re.finditer(r"\[FIGURE:.*?\](.*?)\[ENDFIGURE\]", doc, re.DOTALL)
    for match in figure_matches:
        full_content = match.group(0)
        title_match = re.search(
            r"\[FIGURE_TITLE\](.*?)\[ENDFIGURE_TITLE\]", match.group(1), re.DOTALL
        )
        if title_match:
            title = title_match.group(1).strip()
            text_to_obj[title] = full_content

    # 处理表格
    table_matches = re.finditer(r"\[TABLE:.*?\](.*?)\[ENDTABLE\]", doc, re.DOTALL)
    for match in table_matches:
        full_content = match.group(0)
        title_match = re.search(
            r"\[TABLE_TITLE\](.*?)\[ENDTABLE_TITLE\]", match.group(1), re.DOTALL
        )
        if title_match:
            title = title_match.group(1).strip()
            text_to_obj[title] = full_content

    # 处理算法
    algo_matches = re.finditer(r"\[ALGORITHM\](.*?)\[ENDALGORITHM\]", doc, re.DOTALL)
    for match in algo_matches:
        full_content = match.group(0)
        title_match = re.search(
            r"\[ALGORITHM_TITLE\](.*?)\[ENDALGORITHM_TITLE\]", match.group(1), re.DOTALL
        )
        if title_match:
            title = title_match.group(1).strip()
            text_to_obj[title] = full_content

    return text_to_obj


def build_line_tag_mapping(doc_lines: List[str]) -> Dict[int, Dict]:
    """
    构建行号到标签类型的映射。每行只可能是以下三种情况之一：
    1. 独立标签行：[TEXT]或[ENDTEXT]
    2. 带标签的内容行：[TAG]内容[ENDTAG]
    3. [TEXT]和[ENDTEXT]之间的正文行

    Args:
        doc_lines: 已经预处理过的文档行列表（已去除空行和空格）

    Returns:
        Dict[int, Dict]: {
            行号: {
                'type': 标签类型,
                'content_type': 'ordered'|'unordered',
                'is_tag_line': bool  # 只有[TEXT]的开始结束标签会被标记为True
            }
        }
    """
    line_tag_map = {}
    in_text = False

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
        'THANK_NOTE': {'content_type': 'unordered'},
    }

    for line_num, line in enumerate(doc_lines):
        line = line.strip()

        # 处理[TEXT]标签
        if line == '[TEXT]':
            in_text = True
            line_tag_map[line_num] = {
                'type': 'TEXT',
                'content_type': 'ordered',
                'is_tag_line': True,
            }
            continue
        elif line == '[ENDTEXT]':
            in_text = False
            line_tag_map[line_num] = {
                'type': 'TEXT',
                'content_type': 'ordered',
                'is_tag_line': True,
            }
            continue

        # 处理其他标签行
        for tag, properties in tag_types.items():
            if f'[{tag}' in line and f'[END{tag}]' in line:
                line_tag_map[line_num] = {
                    'type': tag,
                    'content_type': properties['content_type'],
                    'is_tag_line': False,
                }
                break

        # 如果在[TEXT]标签内且不是其他标签行，则为正文行
        if in_text and line_num not in line_tag_map:
            line_tag_map[line_num] = {
                'type': 'TEXT',
                'content_type': 'ordered',
                'is_tag_line': False,
            }

        # if line_tag_map[line_num] is None:
        #     print(line)
        #     with open("error_line.txt", "a", encoding="utf-8") as f:
        #         f.write(line + "\n")

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
        r"\[(FIGURE:.*?|TABLE:.*?|ALGORITHM)\](.*?)\[END(FIGURE|TABLE|ALGORITHM)\]",
        lambda m: (
            re.search(
                r"\[(FIGURE|TABLE|ALGORITHM)_TITLE\](.*?)\[END(FIGURE|TABLE|ALGORITHM)_TITLE\]",
                m.group(2),
                re.DOTALL,
            )
            .group(0)
            .strip()
            if re.search(
                r"\[(FIGURE|TABLE|ALGORITHM)_TITLE\](.*?)\[END(FIGURE|TABLE|ALGORITHM)_TITLE\]",
                m.group(2),
                re.DOTALL,
            )
            else ""
        ),
        doc,
        flags=re.DOTALL,
    )

    return doc


if __name__ == "__main__":
    test_mmd = "/home/ninziwei/lyj/nougat/__test_0/markdown/2303.00058.mmd"
    output_mmd = "/home/ninziwei/lyj/nougat/__test_0/markdown/2303.00058_processed.mmd"

    with open(test_mmd, "r", encoding="utf-8") as f:
        mmd_text = f.read()

    doc_lines, text_obj_map, line_tag_map = parse_markdown_lines(mmd_text)

    with open("text_obj_map.json", "w", encoding="utf-8") as f:
        json.dump(text_obj_map, f, ensure_ascii=False, indent=4)

    with open("line_tag_map.json", "w", encoding="utf-8") as f:
        json.dump(line_tag_map, f, ensure_ascii=False, indent=4)

    with open(output_mmd, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
