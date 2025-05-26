import jieba
import re
from typing import List, Dict
from nougat.dataset.split_utils.text_cleaner import squeeze_text
from nougat.dataset.split_utils.string_matcher import get_char_match_score
from nougat.dataset.split_utils.pdf_processor import extract_lower_pdf_lines


def build_pdf_words_for_match(words: List[str]) -> List[str]:
    """
    构建单词组合，用于文本匹配.
    """
    words = [word.strip() for word in words if len(word.strip())]
    lens = [len(word) for word in words]
    max_len = max(lens)
    if max_len > 1:
        if len(words) > 10:
            new_words = [" ".join(words[j : j + 3]) for j in range(len(words) - 2)]  # 3-grams
        elif len(words) > 5:
            new_words = [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]  # 2-grams
        else:
            new_words = [word for word in words if len(word) > 1]  # 1-grams
            new_words += [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]  # 2-grams
    elif max_len == 1:
        if len(words) > 3:
            new_words = [" ".join(words[j : j + 4]) for j in range(len(words) - 3)]  # 4-grams
        else:
            new_words = [" ".join(words)]  # 1-grams
    new_words = list(set(new_words))
    return new_words


def build_markdown_words_for_index(words: List[str]) -> List[str]:
    """
    构建单词组合，用于索引.
    """
    words = [word.strip() for word in words if len(word.strip())]
    lens = [len(word) for word in words]
    max_len = max(lens)
    if max_len > 1:
        if len(words) > 8:
            new_words = [" ".join(words[j : j + 3]) for j in range(len(words) - 2)]  # 3-grams
            new_words += [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]  # 2-grams
        else:
            new_words = [word for word in words if len(word) > 1]  # 1-grams
            new_words += [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]  # 2-grams
    elif max_len == 1:
        if len(words) > 3:
            new_words = [" ".join(words[j : j + 4]) for j in range(len(words) - 3)]  # 4-grams
        else:
            new_words = [" ".join(words)]  # 1-grams
    new_words = list(set(new_words))
    return new_words


def build_inverted_index(lines: List[str]) -> Dict[str, List[int]]:
    """
    构建倒排索引，返回一个字典，键为单词，值为该单词在行数组中的索引列表.
    """
    inverted_index = {}
    words_by_line = [jieba.lcut(line.lower()) for line in lines]
    new_words_by_line = [build_markdown_words_for_index(words) for words in words_by_line]
    for i, (words, new_words) in enumerate(zip(words_by_line, new_words_by_line)):
        # i: 行号
        # words: words_by_line, 每行单词列表
        # new_words: new_words_by_line, 每行单词组合列表
        for new_word in new_words:
            inverted_index.setdefault(new_word, []).append(i)
    return inverted_index


def filter_and_match_lines(pdf, doc_lines: List[str], debug=False):
    """
    过滤和匹配PDF中的行与文档中的行

    Args:
        pdf: PDF文件对象
        doc_lines: 文档行列表
        debug: 是否开启调试模式

    Returns:
        valid_lines_of_pages: PDF文档中每一页的有效行列表
    """
    # 获取干净的文档中的纯文本
    strip_doc_lines = [squeeze_text(line).lower() for line in doc_lines]

    # 对Markdown文档构建倒排索引
    doc_inverted_index = build_inverted_index(doc_lines)

    # 获取干净的 pdf 中的纯文本
    raw_lines_of_pages = extract_lower_pdf_lines(pdf)

    # 遍历每一页，获取有效行
    last_line = ""  # 上一行PDF文本
    valid_lines_of_pages = []  # 每一PDF页的有效行
    for page_lines in raw_lines_of_pages:
        single_words_by_line = [jieba.lcut(line) for line in page_lines]
        words_by_line = [build_pdf_words_for_match(words) for words in single_words_by_line]

        valid_lines = []  # 当前PDF页的有效行
        strip_page_lines = [squeeze_text(line) for line in page_lines]

        for i, (line, strip_pdf_line, words) in enumerate(
            zip(page_lines, strip_page_lines, words_by_line)
        ):
            # i: 行号
            # line: PDF行文本
            # strip_line: 去除所有空格后的PDF行文本
            # words: PDF行文本的单词组合

            # 确保 "Thus,"，"Where,"这种超短的单行成段的文本有效
            if strip_pdf_line in strip_doc_lines:
                # 如果去除所有空格后的PDF行文本在Markdown文档中存在
                valid_lines.append(line)
                last_line = strip_pdf_line
                continue

            # 如果行长度太短，跳过
            if len(strip_pdf_line) <= 3:
                last_line = strip_pdf_line
                continue

            # 如果行文本以 "abstract" 开头，并且长度大于30，则去除前8个字符，并去除末尾的 ".—"
            if strip_pdf_line.startswith("abstract") and len(strip_pdf_line) > 30:
                strip_pdf_line = strip_pdf_line[8:]
                strip_pdf_line = strip_pdf_line.strip(".—")

            # 如果行长度大于20，则可以用完美匹配的方式直接判断是否有效
            if len(strip_pdf_line) > 20:
                strip_doc_text = "".join(strip_doc_lines)  # 将所有Markdown行文本拼接成一个字符串
                if strip_pdf_line in strip_doc_text:  # 如果PDF行文本在Markdown文档中存在
                    valid_lines.append(line)
                    last_line = strip_pdf_line
                    continue

            # tmp_line: 去除标点符号后的PDF行文本
            tmp_line = re.sub(r"-|_| ", "", strip_pdf_line)
            tmp_line = tmp_line.strip("{}[]<>()（）")

            # 如果去除标点符号后的PDF行文本是数字，则认为有效
            if tmp_line.isdigit():
                last_line = strip_pdf_line
                continue

            # 如果去除标点符号后的PDF行文本是字母，并且长度小于10，则认为有效
            if tmp_line.isalpha() and len(tmp_line) < 10:
                last_line = strip_pdf_line
                continue

            # 如果去除标点符号后的PDF行文本是字母，并且上一行或下一行长度小于15，则认为有效
            next_line = strip_page_lines[i + 1] if i + 1 < len(strip_page_lines) else ""
            if tmp_line.isalpha() and (
                (last_line and len(last_line) < 15) or (next_line and len(next_line) < 15)
            ):
                last_line = strip_pdf_line
                continue

            # 获取候选行索引(Markdown文档中的行索引)
            valid_num = 0
            candid_doc_line_indices = []
            for word in words:  # 遍历PDF行文本的单词组合
                # doc_inverted_index[word] 表示单词word在Markdown文档中的所有行索引
                if word in doc_inverted_index:
                    candid_doc_line_indices += doc_inverted_index[word]
                    valid_num += 1
                if valid_num == 5:
                    break
            candid_doc_line_indices = list(set(candid_doc_line_indices))  # 去重

            # 获取候选行文本(Markdown行文本)
            candid_doc_lines = [strip_doc_lines[idx] for idx in candid_doc_line_indices]

            # 计算匹配分数
            doc_scores = []
            for doc_line in candid_doc_lines:
                doc_scores.append(get_char_match_score(content=doc_line, query=strip_pdf_line))
                # 如果匹配分数大于0.9,说明找到了很好的匹配,不需要继续计算其他候选行的分数
                if doc_scores[-1] > 0.9:
                    break

            if len(strip_pdf_line) > 30:
                if doc_scores and max(doc_scores) > 0.5:
                    valid_lines.append(line)
            else:
                if doc_scores and max(doc_scores) > 0.75:
                    valid_lines.append(line)

            last_line = strip_pdf_line

        valid_lines_of_pages.append(valid_lines)

    if debug:
        out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
        with open(f"{out_path}/valid_lines.txt", "w", encoding="utf-8") as f:
            for lines in valid_lines_of_pages:
                f.write("\n".join(lines))
                f.write("\n##### @@ #####\n")

    return valid_lines_of_pages
