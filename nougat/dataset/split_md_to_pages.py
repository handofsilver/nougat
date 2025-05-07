"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

"""

import os
import re
import json
import pypdf
import argparse
import jieba
import numpy as np

from time import time
from typing import Dict, List, Tuple
from split_utils import encode_formula_in_md
from string_matcher import get_char_match_score, build_partial_match_table
from rasterize import rasterize_paper


def build_words(words):
    words = [word.strip() for word in words if len(word.strip())]
    lens = [len(word) for word in words]
    max_len = max(lens)
    if max_len > 1:
        # words = [word for word in words if len(word) > 1]
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


def build_words_for_index(words):
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


def build_inverted_index(lines):
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


def parse_markdown(doc):
    """
    得到文档中每一类元素的文本内容
    先清洗掉所有 [TABLE:.*] 和 [ENDTABLE] 之间，
        [FIGURE:.*] 和 [ENDFIGURE] 之间，
        [LISTING:.*] 和 [ENDLISTING] 之间，
        [ALGORITHM] 和 [ENDALGORITHM] 之间，
        [FOOTNOTE] 和 [ENDFOOTNOTE] 之间
    的所有文本，其中.*表示任意长度字符
    然后从 doc 中提取出所有 [TEXT] 和 [ENDTEXT] 之间的文本并用 '\n' 将其拼接成一个长文本
    doc 中有多组 [TEXT] 和 [ENDTEXT]，请提取每一对 [TEXT] 和 [ENDTEXT] 之间的文本
    """
    # 清洗掉各类标签对之间的文本
    # 清洗 TABLE 标签及其中间的内容
    # table_matches = re.finditer(r'\[TABLE:.*?\].*?\[ENDTABLE\]', doc, flags=re.DOTALL)
    # for match in table_matches:
    #     print("Found TABLE match:", match.group())
    doc = re.sub(r"\[TABLE:.*?\].*?\[ENDTABLE\]", "", doc, flags=re.DOTALL)
    # 清洗 FIGURE 标签及其中间的内容
    # table_matches = re.finditer(r'\[FIGURE:.*?\].*?\[ENDFIGURE\]', doc, flags=re.DOTALL)
    # for match in table_matches:
    #     print("46 Found FIGURE match:", match.group())
    doc = re.sub(r"\[FIGURE:.*?\].*?\[ENDFIGURE\]", "", doc, flags=re.DOTALL)
    # 清洗 LISTING 标签及其中间的内容
    doc = re.sub(r"\[LISTING:.*?\].*?\[ENDLISTING\]", "", doc, flags=re.DOTALL)
    # 清洗 ALGORITHM 标签及其中间的内容
    doc = re.sub(r"\[ALGORITHM\].*?\[ENDALGORITHM\]", "", doc, flags=re.DOTALL)
    # 清洗 FOOTNOTE 标签及其中间的内容
    doc = re.sub(r"\[FOOTNOTE\].*?\[ENDFOOTNOTE\]", "", doc, flags=re.DOTALL)

    # 使用正则表达式找到所有 [TEXT] 和 [ENDTEXT] 之间的内容
    text_blocks = re.finditer(r"\[TEXT\](.*?)\[ENDTEXT\]", doc, re.DOTALL)

    # 将所有找到的文本块用换行符连接
    doc_text = "\n".join(block.group(1).strip() for block in text_blocks)

    # 如果没有找到任何 [TEXT] 标记，则使用整个文档
    if not doc_text:
        doc_text = doc

    # 去掉[FORMULA]和[ENDFORMULA]标签
    doc_text = doc_text.replace("[FORMULA]", "").replace("[ENDFORMULA]", "")

    return doc_text


def squeeze_text(text: str) -> str:
    text = (
        text.replace("\n", "").replace("\t", "").replace("\xa0", " ").replace(" ", "")
    )
    return text


def replace_join_text(text):
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


def get_span_of_pages(doc_lines, page_start_positions, page_end_positions):
    # 确定分割位置
    start_list, end_list = [], []
    whole_start_list, whole_end_list = [], []
    bad_pages = []  # 如果开头结尾冲突，则认为这两页需要丢弃
    coinside_pages = []  # 两页重合的页码对

    # 辅助参数
    last_end = 0
    last_end_positions = [-1] + page_end_positions[:-1]
    next_start_positions = page_start_positions[1:] + [len(doc_lines)]
    # 获取每一页的开始和结束位置
    for i, (start_idx, end_idx, last_end_idx, next_start_idx) in enumerate(
        zip(
            page_start_positions,
            page_end_positions,
            last_end_positions,
            next_start_positions,
        )
    ):
        # 该页没有行，直接添加占位结果
        if start_idx == -1:
            whole_start_list.append(-1)
            whole_end_list.append(-1)
            continue
        # 该页头尾都失配，直接添加占位结果
        elif start_idx == -2 and end_idx == -2:
            start_list.append(-2)
            end_list.append(-2)
            bad_pages.append(i)
        # 该页头失配，尾匹配成功
        elif start_idx < 0 and end_idx >= 0:
            # 如果该页是第一页，则直接添加0
            if i == 0:
                start_list.append(0)
            # 如果上一页正常有效，则按照上一页来
            elif last_end_idx >= 0:
                start_list.append(last_end + 1)
            # 如果上一页是无效匹配，则直接添加占位结果
            else:
                start_list.append(last_end)
                bad_pages.append(i)
            end_list.append(end_idx)
        # 该页头匹配成功，尾失配
        elif start_idx >= 0 and end_idx < 0:
            start_list.append(start_idx)
            if next_start_idx > 0:
                end_list.append(next_start_idx - 1)
            else:
                end_list.append(start_idx)
                bad_pages.append(i)
        # 该页头尾都匹配成功
        else:
            if end_idx >= start_idx:
                start_list.append(start_idx)
                end_list.append(end_idx)
            else:
                start_list.append(last_end)
                end_list.append(last_end)
                bad_pages.append(i)

        # 更新完整参数
        whole_start_list.append(start_list[-1])
        whole_end_list.append(end_list[-1])

        last_end = end_list[-1]

        # 判断分页是否在同一段
        if end_idx == next_start_idx:
            coinside_pages.append((i, i + 1))

    page_span = [(start, end) for start, end in zip(whole_start_list, whole_end_list)]

    print(246, whole_start_list)
    print(247, whole_end_list)
    print(248, bad_pages)
    print(249, coinside_pages)

    return page_span, coinside_pages, bad_pages


def split_markdown(doc, pdf, figure_info, debug=False) -> Tuple[List[str], Dict]:
    """
    Split a PDF document into Markdown paragraphs.

    Args:
        doc (str): latex 转 html 再转 markdown 后 md 文本内容.
        pdf (pypdf.PdfReader): 用 pypdf 读取 pdf 文件的结果.
        figure_info (Optional[List[Dict]]): 图表信息，每个字典指定一个图表的信息，包括标题、页码和边界框.

    Returns:
        doc_lines_by_pages: 每一页的行数组
        coinside_pages: 两页重合的页码对
        bad_pages: 两页冲突的页码对
    """

    # 用正则表达式从 doc 中提取出所有 [TEXT] 和 [ENDTEXT] 之间的文本
    doc = encode_formula_in_md(doc)
    doc_text = parse_markdown(doc).lower()

    doc_lines = doc_text.split("\n")
    doc_lines = [line.strip() for line in doc_lines if line.strip()]
    # 由于html中会有嵌套的[TEXT][ENDTEXT]标签，所以parse_markdown后的文本中还会有[text]和[endtext]
    doc_lines = [line for line in doc_lines if line != "[text]" and line != "[endtext]"]

    # 对 doc_lines 中的每一行分词后构建从 word 到行号的倒排索引
    doc_inverted_index = build_inverted_index(doc_lines)

    # 去掉空格、换行等分隔符
    strip_doc_text = squeeze_text(doc_text)
    strip_doc_lines = [squeeze_text(line) for line in doc_lines]

    if debug:
        out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
        with open(f"{out_path}/doc_text.txt", "w", encoding="utf-8") as f:
            f.write(doc_text)
        with open(f"{out_path}/strip_doc_lines.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(strip_doc_lines))

    test_str = ""
    # test_str = "ωp=⌊ω"
    test_str = "whichwillinfluence"

    time_start = time()
    # 获取干净的 pdf 中的纯文本
    last_line = ""
    valid_lines_of_pages = []
    raw_lines_by_page = [
        replace_join_text(page.extract_text()).split("\n") for page in pdf.pages
    ]
    for page_lines in raw_lines_by_page:
        page_lines = [line.strip().lower() for line in page_lines if line.strip()]
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
                if strip_line in strip_doc_text:
                    valid_lines.append(line)
                    last_line = strip_line
                    continue

            tmp_line = "".join(re.split(r"-|_| ", strip_line))
            tmp_line.strip("{}[]<>()（）")

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
            # doc_scores = [
            #     get_char_match_score(doc_line, strip_line, match_table)
            #     for doc_line in candid_doc_lines
            # ]
            doc_scores = []
            for doc_line in candid_doc_lines:
                doc_scores.append(
                    get_char_match_score(doc_line, strip_line, match_table)
                )
                if doc_scores[-1] > 0.9:
                    break
            # caption_scores = [get_char_match_score(strip_caption_line, strip_line) for strip_caption_line in candid_caption_lines]

            if len(strip_line) > 30:
                if doc_scores and max(doc_scores) > 0.5:
                    valid_lines.append(line)
            else:
                if doc_scores and max(doc_scores) > 0.75:
                    valid_lines.append(line)

            last_line = strip_line

            if debug:
                if len(candid_doc_line_idxes) > 50:
                    print(324, words, line)
                if test_str and test_str in strip_line:
                    print(375, words)
                    print(383, len(candid_doc_line_idxes), doc_scores)
                    print(384, len(candid_doc_line_idxes), doc_scores)
                    print(385, strip_line)
                    print(doc_scores)
                    print(valid_lines[-1])

        valid_lines_of_pages.append(valid_lines)

    time_end = time()
    print(376, time_end - time_start)

    if debug:
        out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
        with open(f"{out_path}/valid_lines.txt", "w", encoding="utf-8") as f:
            for lines in valid_lines_of_pages:
                f.write("\n".join(lines))
                f.write("\n##### @@ #####\n")

    # 初始化指针和窗口大小
    min_window_size = 30
    start_pointer, end_pointer = 0, 0
    start_window_size, end_window_size = min_window_size, min_window_size
    page_start_positions = []  # 该页第一行文本在 doc_lines 中的索引
    page_end_positions = []  # 该页最后一行文本在 doc_lines 中的索引

    # 获得每一页的开始和结束行在 doc_lines 中的索引
    """
    状态定义
    -1: 该页没有行
    -2: 该行匹配失败
    '{type}:{num}': 匹配到的是哪一类元素中的哪一行
    """
    test_str = "six-dofhapti"
    for page_lines in valid_lines_of_pages:
        # 如果该页没有行，则认为没有有效位置
        if not page_lines:
            page_start_positions.append(-1)
            page_end_positions.append(-1)
            continue

        # 获得该页的开始和结束行
        start_line = squeeze_text(page_lines[0])
        end_line = squeeze_text(page_lines[-1])
        start_line_table = build_partial_match_table(start_line)
        end_line_table = build_partial_match_table(end_line)

        # 计算开始行的匹配分数
        start_window_end = min(start_pointer + start_window_size, len(strip_doc_lines))
        start_scores = [
            get_char_match_score(doc_line, start_line, start_line_table)
            for doc_line in strip_doc_lines[start_pointer:start_window_end]
        ]

        if start_scores and max(start_scores) > 0.85:
            start_idx = start_pointer + np.argmax(start_scores)
            page_start_positions.append(start_idx)
            start_pointer = start_idx + 1
            # 如果匹配成功，则恢复原始窗口大小
            start_window_size = min_window_size
        else:
            # 第一次没有匹配成功，则给个机会扩大窗口再来一次
            start_window_end = min(
                start_pointer + start_window_size + min_window_size,
                len(strip_doc_lines),
            )
            start_scores = [
                get_char_match_score(doc_line, start_line, start_line_table)
                for doc_line in strip_doc_lines[start_pointer:start_window_end]
            ]

            if start_scores and max(start_scores) > 0.85:
                start_idx = start_pointer + np.argmax(start_scores)
                page_start_positions.append(start_idx)
                start_pointer = start_idx + 1
                # 如果匹配成功，则恢复原始窗口大小
                start_window_size = min_window_size
            else:
                page_start_positions.append(-2)
                # 如果匹配失败，则扩大窗口
                start_window_size += min_window_size

        # 计算结束行的匹配分数
        end_window_end = min(end_pointer + end_window_size, len(strip_doc_lines))
        end_scores = [
            get_char_match_score(doc_line, end_line, end_line_table)
            for doc_line in strip_doc_lines[end_pointer:end_window_end]
        ]

        if test_str and test_str in end_line:
            print(264, max(end_scores), end_pointer, end_window_end, end_line)
            idx = np.argmax(end_scores)
            print(idx)
            print(strip_doc_lines[end_pointer:end_window_end][idx])

        if end_scores and max(end_scores) > 0.85:
            end_idx = end_pointer + np.argmax(end_scores)
            page_end_positions.append(end_idx)
            end_pointer = end_idx + 1
            # 如果匹配成功，则恢复原始窗口大小
            end_window_size = min_window_size
        else:
            # 第一次没有匹配成功，则给个机会扩大窗口再来一次
            end_window_end = min(
                end_pointer + end_window_size + min_window_size, len(strip_doc_lines)
            )
            end_scores = [
                get_char_match_score(doc_line, end_line, end_line_table)
                for doc_line in strip_doc_lines[end_pointer:end_window_end]
            ]

            # print(264, max(end_scores), end_pointer, end_window_end, end_line)
            if end_scores and max(end_scores) > 0.85:
                end_idx = end_pointer + np.argmax(end_scores)
                page_end_positions.append(end_idx)
                end_pointer = end_idx + 1
                # 如果匹配成功，则恢复原始窗口大小
                end_window_size = min_window_size
            else:
                page_end_positions.append(-2)
                # 如果匹配失败，则扩大窗口
                end_window_size += min_window_size

        if (
            page_end_positions[-1] > 0
            and page_start_positions[-1] > page_end_positions[-1]
        ):
            page_start_positions[-1] = -2
            start_pointer = page_end_positions[-1] - 1

        if debug:
            print(231, page_start_positions[-1], max(start_scores), start_line)
            print(232, page_end_positions[-1], max(end_scores), end_line)

    if debug:
        print(392, page_start_positions)
        print(393, page_end_positions)

    # 根据分割位置拆分文档
    page_spans, coinside_pages, bad_pages = get_span_of_pages(
        doc_lines, page_start_positions, page_end_positions
    )

    # 根据分割位置拆分文档
    doc_text_by_pages = []
    for start, end in page_spans:
        if start > 0:
            doc_text_by_pages.append("\n".join(doc_lines[start : end + 1]))
        else:
            doc_text_by_pages.append("")

    print(516, len(doc_text_by_pages))
    return doc_text_by_pages, page_spans, coinside_pages, bad_pages


def use_split_markdown():
    """
    主函数，处理命令行参数并执行拆分过程。

    流程：
    1. 解析命令行参数（Markdown文件、PDF文件、输出目录、图表信息等）
    2. 读取Markdown和PDF文件
    3. 调用split_markdown函数进行拆分
    4. 将拆分结果保存到指定目录
    5. 调用rasterize_paper函数将PDF页面转换为图像
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=str, help="Markdown file", required=False)
    parser.add_argument("--pdf", type=str, help="PDF File", required=False)
    parser.add_argument("--out", type=str, help="Out dir", required=False)
    parser.add_argument(
        "--figure",
        type=str,
        help="Figure info JSON",
    )
    parser.add_argument("--dpi", type=int, default=96)
    args = parser.parse_args()

    doc_name = "2303.00058"
    # doc_name = "2402.00041"
    base_path = "/home/ninziwei/lyj/nougat/__test_0"
    args.md = f"{base_path}/markdown/{doc_name}.mmd"
    args.pdf = f"{base_path}/src/{doc_name}.pdf"
    args.figure = f"{base_path}/fig/{doc_name}.json"
    base_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test"
    args.out = f"{base_path}/out"

    md = open(args.md, "r", encoding="utf-8").read().replace("\xa0", " ")
    pdf = pypdf.PdfReader(args.pdf)
    fig_info = json.load(open(args.figure, "r", encoding="utf-8"))

    pages = [pdf.pages[i].extract_text() for i in range(len(pdf.pages))]
    with open(f"{args.out}/{doc_name}-pypdf.txt", "w", encoding="utf-8") as f:
        for page in pages:
            f.write(page)
            f.write("\n##### @@ #####\n")

    pages, page_spans, coinside_pages, bad_pages = split_markdown(md, pdf, fig_info)

    print(563, len(pages))
    with open(f"{args.out}/{doc_name}.txt", "w", encoding="utf-8") as f:
        for page in pages:
            f.write(page)
            f.write("\n##### @@ #####\n")

    if args.out:
        outpath = os.path.join(args.out, os.path.basename(args.pdf).partition(".")[0])
        os.makedirs(outpath, exist_ok=True)
        found_pages = []
        for i, content in enumerate(pages):
            if content:
                with open(
                    os.path.join(outpath, "%02d.mmd" % (i + 1)),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(content)
                found_pages.append(i)
        rasterize_paper(args.pdf, outpath, dpi=args.dpi, pages=found_pages)


if __name__ == "__main__":
    # content = 'of the edge set of the original problem \(E\), i.e., \({|E|^{-1}\cdot\sum_{p=1}^{q}|E_{p}|}\)'
    # query = 'of the edge set of the original problem E, i.e., |E|−1 · Pq'
    # content = 'Givenitspracticalrelevance,numerousstudiesaboutthevehicleroutingproblem(VRP,DantzigandRamser1959)anditsvariantsexist(Vidaletal.2020).Typically,richVRPsoflarge-scalereal-worldapplicationsaresolvedbyheuristics.State-of-the-artmetaheuristicsusemostoftheircomputationtimesearchingforlocalimprovementsinanincumbentsolutionbymodifyingcustomersequenceswithinagiventourorchangingcustomer-vehicleassignments(Vidaletal.2013).Astheproblemsizeincreases,thenumberofpossiblelocalsearch(LS)operationsgrowsexponentially.Inresponse,complexityreductiontechniquesareappliedtolimitthesolutionspace.Decompositionandaggregationmethodsdividetheoriginalproblemintomultiplesmalleronesthataresolvedindependently(Santinietal.2023).PruninglimitstheLSoperators'explorationofnewsolutions(ArnoldandSörensen2019).Thesestrategiesmostlyfollowsimplerulestailoredtoaparticularproblem.Inpractice,however,solutionalgorithmsmustbescalableandadjustabletovariousproblemcharacteristics.Thus,weproposeageneralizablesolutionframeworkcalleddecompose-route-improve(DRI)thatreducesthecomplexityoflarge-scaleroutingproblemsusingdata-baseddecomposition.ThisapproachusesunsupervisedclusteringtosplitthecustomersofaVRPintoseparatesubsets.Itssimilaritymetriccombinescustomers'spatial,temporal,anddemandfeatureswiththeproblem'sobjectivefunctionandconstraints.Theresultingstand-alonesmall-sizedsub-VRPsaresolvedindependently.Thesolutiontotheoverallproblemisthecombinationoftheindividualsolutionsofthesubproblems.Finally,LS,prunedbasedoncustomers'spatial-temporal-demandsimilarity,resolvesunfavorableroutingdecisionsattheperimetersofthesubproblems.Thisapproachdemonstrateshighscalabilityandexpeditiouslyachieveshigh-qualitysolutionsforlarge-scaleVRPs.Thus,ourcontributiontothegrowingresearchfieldofheuristicdecompositionistwofold.'
    # query = 'contributiontothegrowingresearchfieldofheuristicdecompositionistwofold.'
    # print(decode_formula(content))
    # start_time = time.time()
    # print(get_char_match_score(content, query))
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")
    # print(get_char_match_score(decode_formula(content), query))

    use_split_markdown()
