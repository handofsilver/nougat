import re


def squeeze_text(text: str) -> str:
    """
    移除文本中的所有空白字符
    """
    return re.sub(r"[\n\t\xa0 ]", "", text)
