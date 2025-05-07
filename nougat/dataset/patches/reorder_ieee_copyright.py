from bs4 import BeautifulSoup
from pathlib import Path


def contains_text(tag, text):
    """
    递归检查标签是否包含指定文本
    :param tag: BeautifulSoup标签
    :param text: 要检查的文本
    :return: 如果标签包含指定文本，则返回True，否则返回False
    """
    if tag.string and text in tag.string:
        return True
    for child in tag.children:
        if isinstance(child, str):
            continue
        if contains_text(child, text):
            return True
    return False


def reorder_article_children_tags(article_tag):
    """
    重新排序文章标签的子标签，将版权声明放在最前面
    :param article_tag: BeautifulSoup文章标签
    """
    children = list(article_tag.children)
    # 找到版权声明标签，移动到最前面
    for child in children:
        if isinstance(child, str):
            continue
        if contains_text(child, "IEEE Copyright Notice"):
            # 移动版权声明标签到最前面
            article_tag.insert(0, child)
            break


def reorder_ieee_copyright(input_file: Path, output_file: Path = None):
    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

        # 找到所有文章标签
        articles = soup.find_all("article", class_="ltx_document")
        for article in articles:
            # 重新排序文章标签的子标签
            reorder_article_children_tags(article)

        # 将修改后的HTML写入文件
        if output_file is None:
            file.write(str(soup))
        else:
            with open(output_file, "w", encoding="utf-8") as output:
                output.write(str(soup))


if __name__ == "__main__":
    html_file = Path("/home/ninziwei/lyj/nougat/nougat/dataset/parser/test.html")
    reorder_ieee_copyright(input_file=html_file, output_file="result.html")
