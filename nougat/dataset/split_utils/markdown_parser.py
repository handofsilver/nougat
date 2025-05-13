import re
from typing import List
from nougat.dataset.split_utils.markdown_encoder import encode_formula_in_md


def parse_markdown(doc: str) -> str:
    """
    解析Markdown文档, 提取文本内容

    1. 清洗掉所有特殊标记标签的内容：
       - [TABLE:.*] 和 [ENDTABLE] 之间
       - [FIGURE:.*] 和 [ENDFIGURE] 之间
       - [LISTING:.*] 和 [ENDLISTING] 之间
       - [ALGORITHM] 和 [ENDALGORITHM] 之间
       - [FOOTNOTE] 和 [ENDFOOTNOTE] 之间
    2. 提取所有 [TEXT] 和 [ENDTEXT] 之间的文本
    3. 用换行符连接所有提取的文本
    """

    # 清洗特殊标签之间的内容
    doc = re.sub(r"\[TABLE:.*?\].*?\[ENDTABLE\]", "", doc, flags=re.DOTALL)
    doc = re.sub(r"\[FIGURE:.*?\].*?\[ENDFIGURE\]", "", doc, flags=re.DOTALL)
    doc = re.sub(r"\[LISTING:.*?\].*?\[ENDLISTING\]", "", doc, flags=re.DOTALL)
    doc = re.sub(r"\[ALGORITHM\].*?\[ENDALGORITHM\]", "", doc, flags=re.DOTALL)
    doc = re.sub(r"\[FOOTNOTE\].*?\[ENDFOOTNOTE\]", "", doc, flags=re.DOTALL)

    # 提取[TEXT]和[ENDTEXT]之间的文本，注意[TEXT]和[ENDTEXT]可能递归出现
    stack = []
    result = []
    pos = 0

    while pos < len(doc):
        start_match = re.search(r"\[TEXT\]", doc[pos:])
        end_match = re.search(r"\[ENDTEXT\]", doc[pos:])

        start_pos = pos + start_match.start() if start_match else float("inf")
        end_pos = pos + end_match.start() if end_match else float("inf")

        if start_pos < end_pos:
            # 记录前一段非标签内容到当前栈顶
            if stack:
                stack[-1] += "\n" + doc[pos:start_pos] + "\n"
            stack.append("")  # 新开一层
            pos = start_pos + len("[TEXT]")
        elif end_pos < float("inf"):
            if stack:
                stack[-1] += "\n" + doc[pos:end_pos] + "\n"
                content = stack.pop().split("\n")
                content = [line.strip() for line in content if line.strip()]
                if stack:
                    stack[-1] += "\n".join(content) + "\n"
                else:
                    result.append("\n".join(content))
            pos = end_pos + len("[ENDTEXT]")
        else:
            if stack:
                stack[-1] += "\n" + doc[pos:] + "\n"
            pos = len(doc)

    doc_text = "\n".join(result).strip()

    # 如果提取的文本为空，则返回原始文档
    if not doc_text:
        doc_text = doc

    # 清理[FORMULA]和[ENDFORMULA]
    doc_text = doc_text.replace("[FORMULA]", "").replace("[ENDFORMULA]", "")

    return doc_text


def parse_markdown_lines(doc: str) -> List[str]:
    """
    解析Markdown文档并返回行列表

    1. 调用 encode_formula_in_md 处理文档中的公式
    2. 调用 parse_markdown 提取文本内容
    3. 将文本分割成行并清理
    """
    doc = encode_formula_in_md(doc)
    doc_text = parse_markdown(doc).lower()

    # 获得文档中每一行文本内容
    doc_lines = doc_text.split("\n")
    doc_lines = [line.strip() for line in doc_lines if line.strip()]  # 清理空白字符
    doc_lines = [
        line for line in doc_lines if line != "[text]" and line != "[endtext]"
    ]  # 清理[text]和[endtext]

    return doc_lines


if __name__ == "__main__":
    test_mmd = "/home/ninziwei/lyj/nougat/__test_0/markdown/2303.00058.mmd"
    output_mmd = "/home/ninziwei/lyj/nougat/__test_0/markdown/2303.00058_processed.mmd"

    with open(test_mmd, "r", encoding="utf-8") as f:
        mmd_text = f.read()

    mmd_text = parse_markdown(mmd_text)

    with open(output_mmd, "w", encoding="utf-8") as f:
        f.write(mmd_text)
