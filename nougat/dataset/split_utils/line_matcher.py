import jieba
import re
from typing import List, Dict
from nougat.dataset.split_utils.text_cleaner import squeeze_text
from nougat.dataset.split_utils.string_matcher import get_char_match_score, build_partial_match_table
from nougat.dataset.split_utils.pdf_processor import extract_clean_pdf_lines

def build_words(words: List[str]) -> List[str]:
    """
    构建单词组合，用于文本匹配
    """
    words = [word.strip() for word in words if len(word.strip())]
    lens = [len(word) for word in words]
    max_len = max(lens)
    if max_len > 1:
        if len(words) > 10:
            new_words = [" ".join(words[j : j + 3]) for j in range(len(words) - 2)]
        elif len(words) > 5:
            new_words = [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]
        else:
            new_words = [word for word in words if len(word) > 1]
            new_words += [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]
    elif max_len == 1:
        if len(words) > 3:
            new_words = [" ".join(words[j : j + 4]) for j in range(len(words) - 3)]
        else:
            new_words = [" ".join(words)]
    new_words = list(set(new_words))
    return new_words

def build_words_for_index(words: List[str]) -> List[str]:
    """
    构建用于索引的单词组合
    """
    words = [word.strip() for word in words if len(word.strip())]
    lens = [len(word) for word in words]
    max_len = max(lens)
    if max_len > 1:
        if len(words) > 8:
            new_words = [" ".join(words[j : j + 3]) for j in range(len(words) - 2)]
            new_words += [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]
        else:
            new_words = [word for word in words if len(word) > 1]
            new_words += [" ".join(words[j : j + 2]) for j in range(len(words) - 1)]
    elif max_len == 1:
        if len(words) > 3:
            new_words = [" ".join(words[j : j + 4]) for j in range(len(words) - 3)]
        else:
            new_words = [" ".join(words)]
    new_words = list(set(new_words))
    return new_words

def build_inverted_index(lines: List[str]) -> Dict[str, List[int]]:
    """
    构建倒排索引，返回一个字典，键为单词，值为该单词在行数组中的索引列表
    """
    inverted_index = {}
    words_by_line = [jieba.lcut(line) for line in lines]
    new_words_by_line = [build_words_for_index(words) for words in words_by_line]
    for i, (words, new_words) in enumerate(zip(words_by_line, new_words_by_line)):
        for word in new_words:
            inverted_index.setdefault(word, []).append(i)
    return inverted_index

def filter_and_match_lines(pdf, doc_lines: List[str], debug=False):
    """
    过滤和匹配PDF中的行与文档中的行
    
    Args:
        pdf: PDF文件对象
        doc_lines: 文档行列表
        debug: 是否开启调试模式
    
    Returns:
        valid_lines_of_pages: Markdown文档中每一页的有效行列表
    """
    # 获取干净的 pdf 中的纯文本
    strip_doc_lines = [squeeze_text(line) for line in doc_lines]
    doc_inverted_index = build_inverted_index(doc_lines)

    last_line = ""
    valid_lines_of_pages = []
    
    raw_page_lines = extract_clean_pdf_lines(pdf)

    for page_lines in raw_page_lines:
        strip_page_lines = [squeeze_text(line) for line in page_lines]
        valid_lines = []

        words_by_line = [jieba.lcut(line) for line in page_lines]
        words_by_line = [build_words(words) for words in words_by_line]

        for i, (line, strip_line, words) in enumerate(
            zip(page_lines, strip_page_lines, words_by_line)
        ):
            # 确保 "Thus,"，"Where,"这种超短的单行成段的文本有效
            if strip_line in strip_doc_lines:
                valid_lines.append(line)
                last_line = strip_line
                continue

            # 如果行长度太短，跳过
            if len(strip_line) <= 3:
                last_line = strip_line
                continue

            if strip_line.startswith("abstract") and len(strip_line) > 30:
                strip_line = strip_line[8:]
                strip_line = strip_line.strip(".—")

            # 如果行长度大于20，则可以用完美匹配的方式直接判断是否有效
            if len(strip_line) > 20:
                strip_doc_text = "".join(strip_doc_lines)
                if strip_line in strip_doc_text:
                    valid_lines.append(line)
                    last_line = strip_line
                    continue

            tmp_line = "".join(re.split(r"-|_| ", strip_line))
            tmp_line = tmp_line.strip("{}[]<>()（）")

            if tmp_line.isdigit():
                last_line = strip_line
                continue

            next_line = strip_page_lines[i + 1] if i + 1 < len(strip_page_lines) else ""

            if tmp_line.isalpha() and len(tmp_line) < 10:
                last_line = strip_line
                continue

            if tmp_line.isalpha() and (
                (last_line and len(last_line) < 15)
                or (next_line and len(next_line) < 15)
            ):
                last_line = strip_line
                continue

            # 获取候选行索引
            valid_num = 0
            candid_doc_line_idxes = []
            for word in words:
                # 确保用的都是有效单词
                if word in doc_inverted_index:
                    candid_doc_line_idxes += doc_inverted_index[word]
                    valid_num += 1
                if valid_num == 5:
                    break
            candid_doc_line_idxes = list(set(candid_doc_line_idxes))
            # 获取候选行文本
            candid_doc_lines = [strip_doc_lines[idx] for idx in candid_doc_line_idxes]

            # 计算匹配分数
            match_table = build_partial_match_table(strip_line)

            doc_scores = []
            for doc_line in candid_doc_lines:
                doc_scores.append(
                    get_char_match_score(doc_line, strip_line, match_table)
                )
                if doc_scores[-1] > 0.9:
                    break

            if len(strip_line) > 30:
                if doc_scores and max(doc_scores) > 0.5:
                    valid_lines.append(line)
            else:
                if doc_scores and max(doc_scores) > 0.75:
                    valid_lines.append(line)

            last_line = strip_line

            if debug:
                if len(candid_doc_line_idxes) > 50:
                    print(words, line)
                test_str = "whichwillinfluence"
                if test_str and test_str in strip_line:
                    print(words)
                    print(len(candid_doc_line_idxes), doc_scores)
                    print(strip_line)
                    print(doc_scores)
                    print(valid_lines[-1])

        valid_lines_of_pages.append(valid_lines)

    if debug:
        out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
        with open(f"{out_path}/valid_lines.txt", "w", encoding="utf-8") as f:
            for lines in valid_lines_of_pages:
                f.write("\n".join(lines))
                f.write("\n##### @@ #####\n")
                
    return valid_lines_of_pages 