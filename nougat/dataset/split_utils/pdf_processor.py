import pypdf
from typing import List


def replace_join_text(text: str) -> str:
    """
    将合字替换为原本的单字
    """
    ligature_mapping = {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ӕ": "ae",
        "œ": "oe",
        "ﬆ": "st",
        "ĳ": "ij",
        "æ": "ae",
        "Æ": "AE",
        "Œ": "OE",
    }
    for ligature, replacement in ligature_mapping.items():
        text = text.replace(ligature, replacement)
    return text


def extract_clean_pdf_lines(pdf: pypdf.PdfReader) -> List[List[str]]:
    """
    从PDF中提取并清理文本行.

    Args:
        pdf: PDF文件对象

    Returns:
        pdf_lines: 每一页的文本行列表
    """
    # 提取每一页的文本并替换合字
    pdf_pages = [replace_join_text(page.extract_text()) for page in pdf.pages]

    pdf_lines = []
    # 处理每一页的文本
    for page in pdf_pages:
        # 分割成行并清理
        page_lines = [line.strip().lower() for line in page.split("\n") if line.strip()]
        pdf_lines.append(page_lines)

    return pdf_lines
