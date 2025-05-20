import re


def remove_comments(tex_content):
    """删除 LaTeX 文本中的注释"""

    # 删除以 % 开头的注释行（允许空格/Tab），包括空行
    tex_content = re.sub(r"(?m)^\s*%.*(?:\n|$)", "", tex_content)

    # 删除行尾注释（非转义 %），包括可能没有 \n 的最后一行
    tex_content = re.sub(r"(?<!\\)%.*", "", tex_content)

    # 清理空行
    tex_content = re.sub(r"\n\s*\n", "\n", tex_content)

    return tex_content


def remove_useless_items(tex_content):
    # 如果文件中不包含 \maketitle，则删除 \author, \title, \date 等内容
    if re.search(r"\\maketitle", tex_content):
        return tex_content

    author_pattern = re.compile(r"\\author\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}")
    title_pattern = re.compile(r"\\title\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}")
    date_pattern = re.compile(r"\\date\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}")

    tex_content = re.sub(author_pattern, "", tex_content)
    tex_content = re.sub(title_pattern, "", tex_content)
    tex_content = re.sub(date_pattern, "", tex_content)

    return tex_content


def preprocess_tex(tex_file_path):
    """预处理 tex 文件"""
    with open(tex_file_path, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()

    # 删除注释
    content = remove_comments(content)

    # 删除 \author, \title, \date 相关的内容（如果没有\maketitle）
    content = remove_useless_items(content)

    with open(tex_file_path, "w", encoding="utf-8") as file:
        file.write(content.strip() + "\n")


if __name__ == "__main__":
    tex_file_path = (
        "/home/ninziwei/lyj/nougat/__test_1/src/2303.00065/root.tex"  # 替换为你的文件路径
    )
    preprocess_tex(tex_file_path)
