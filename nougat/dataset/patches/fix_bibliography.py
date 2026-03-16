import re
from bs4 import BeautifulSoup
from nougat.dataset.patches.latex_to_html import latex_to_html


def parse_bbl(bbl_file) -> BeautifulSoup:
    with open(bbl_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 更精确的正则表达式匹配模式
    items = re.findall(
        r"\\bibitem(?:\[\{(.*?)\}\])?\{(.*?)\}(.*?)(?=\\bibitem|\\end{thebibliography})",
        content,
        flags=re.DOTALL,
    )

    soup = BeautifulSoup(features="html.parser")

    # 创建section元素
    section = soup.new_tag("section", **{"class": "ltx_bibliography", "id": "bib"})

    # 标题部分
    h2 = soup.new_tag("h2", **{"class": "ltx_title ltx_title_bibliography"})
    h2.string = "References"
    section.append(h2)

    # 参考文献列表
    ul = soup.new_tag("ul", **{"class": "ltx_biblist"})

    for label, key, entry in items:
        li = soup.new_tag(
            "li",
            **{"class": "ltx_bibitem", "id": f"bib.{key}"},  # 直接使用bibitem的引用键
        )

        # 标签处理（含特殊字符转换）
        label = re.sub(r"\\protect\\BIBand{}", " and ", label)
        label = re.sub(r"\\\w+", "", label)  # 移除其他LaTeX命令

        tag = soup.new_tag(
            "span", **{"class": "ltx_tag ltx_role_refnum ltx_tag_bibitem"}
        )
        tag.string = label.strip()
        li.append(tag)

        # 内容处理（含LaTeX转换）
        entry = latex_to_html(entry)

        # 创建带HTML格式的内容块
        entry_span = soup.new_tag("span", **{"class": "ltx_bibblock"})
        entry_span.append(BeautifulSoup(entry, "html.parser"))
        li.append(entry_span)

        ul.append(li)

    section.append(ul)

    return section


def fix_bibliography(html_file, bbl_file):
    """
    Fix the bibliography in the HTML file by replacing it with the parsed content from the BBL file.
    If there is 'li' in the old section, it means that the bibliography exists and needs <NOT> to be replaced.

    Args:
    html_file (str): Path to the HTML file.
    bbl_file (str): Path to the BBL file.
    """

    with open(html_file) as f:
        soup = BeautifulSoup(f, "html.parser")

    new_section = parse_bbl(bbl_file)

    # 查找并替换整个bibliography
    old_section = soup.find("section", {"class": "ltx_bibliography"})

    if not old_section:
        # 如果没有找到旧的bibliography部分，则报错！
        raise ValueError("No existing bibliography section found.")
    elif old_section and not old_section.find("li"):
        old_section.replace_with(new_section)

    # 保存修改后的HTML文件
    with open(html_file, "w") as f:
        f.write(str(soup))
    print(f"✅ Fixed bibliography in {html_file}.")
