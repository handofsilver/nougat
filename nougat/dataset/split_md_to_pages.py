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
import random
import numpy as np

from typing import Dict, List, Tuple
from itertools import chain
from split_utils import encode_formula_in_md
from string_matcher import get_char_match_score
from rasterize import rasterize_paper


def build_inverted_index(lines):
    """
    构建倒排索引，返回一个字典，键为单词，值为该单词在行数组中的索引列表
    """
    inverted_index = {}
    for i, line in enumerate(lines):
        words = jieba.lcut(line)
        if max([len(word.strip()) for word in words]) > 3:
            words = [word.strip() for word in words if len(word.strip()) > 1]
        else:
            words = [word.strip() for word in words]
        words = list(set(words))
        for word in words:
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


def get_page_result_by_point(doc_lines, page_start_positions, page_end_positions):
    # 确定分割位置
    last_position = 0
    split_positions = []  # 两页的分割位置
    bad_pages = []  # 如果开头结尾冲突，则认为这两页需要丢弃
    coinside_pages = []  # 两页重合的页码对

    for i, (end_idx, start_idx) in enumerate(
        zip(page_end_positions[:-1], page_start_positions[1:])
    ):
        # print(282, end_idx, start_idx)
        if end_idx == -1 and start_idx == -1:
            split_positions.append(last_position)
        elif end_idx == -2 and start_idx == -2:
            split_positions.append(last_position)
        elif end_idx < 0 and start_idx > 0:
            split_positions.append(start_idx - 1)
        elif start_idx < 0 and end_idx > 0:
            split_positions.append(end_idx)
        # 两者都大于0
        else:
            if 0 <= start_idx - end_idx:
                split_positions.append(end_idx)
                if start_idx == end_idx:
                    coinside_pages.append((i, i + 1))
            else:
                split_positions.append(last_position)
                bad_pages += [i, i + 1]
        last_position = split_positions[-1]

    # 根据分割位置拆分文档
    split_positions = [0] + split_positions + [len(doc_lines)]
    print(122, split_positions)
    doc_text_by_pages = []
    for i in range(len(split_positions) - 1):
        doc_text_by_pages.append(
            "\n".join(doc_lines[split_positions[i] : split_positions[i + 1] + 1])
        )

    print(129, len(doc_text_by_pages))

    return doc_text_by_pages, coinside_pages, bad_pages


def get_page_result_by_span(doc_lines, page_start_positions, page_end_positions):
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
            whole_start_list.append(last_end)
            whole_end_list.append(last_end)
            continue
        # 该页头尾都失配，直接添加占位结果
        elif start_idx == -2 and end_idx == -2:
            start_list.append(last_end)
            end_list.append(last_end)
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

    # 根据分割位置拆分文档
    doc_text_by_pages = []
    for start, end in zip(whole_start_list, whole_end_list):
        doc_text_by_pages.append("\n".join(doc_lines[start:end]))

    print(246, whole_start_list)
    print(247, whole_end_list)
    print(248, bad_pages)
    print(249, coinside_pages)
    print(250, len(doc_text_by_pages))

    return doc_text_by_pages, coinside_pages, bad_pages


def split_markdown(
    doc,
    pdf,
    figure_info,
) -> Tuple[List[str], Dict]:
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
    out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
    with open(f"{out_path}/doc_text.txt", "w", encoding="utf-8") as f:
        f.write(doc_text)

    doc_lines = doc_text.split("\n")
    doc_lines = [line.strip() for line in doc_lines if line.strip()]
    # 由于html中会有嵌套的[TEXT][ENDTEXT]标签，所以parse_markdown后的文本中还会有[text]和[endtext]
    doc_lines = [line for line in doc_lines if line != "[text]" and line != "[endtext]"]

    # 提取图表标题
    caption_lines = []
    if "figures" in figure_info:
        caption_lines = [item.get("caption", "") for item in figure_info["figures"]]
        caption_lines = [encode_formula_in_md(line) for line in caption_lines if line]

    # 对 doc_lines 中的每一行分词后构建从 word 到行号的倒排索引
    doc_inverted_index = build_inverted_index(doc_lines)
    caption_inverted_index = build_inverted_index(caption_lines)

    # 去掉空格、换行等分隔符
    strip_doc_text = squeeze_text(doc_text)
    strip_doc_lines = [squeeze_text(line) for line in doc_lines]
    strip_caption_lines = [squeeze_text(line) for line in caption_lines]
    strip_caption_text = "".join(strip_caption_lines)

    out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
    with open(f"{out_path}/strip_doc_lines.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(strip_doc_lines))

    # 获取干净的 pdf 中的纯文本
    last_line = ""
    lines_of_pages = []
    for page in pdf.pages:
        page_text = page.extract_text()
        page_lines = page_text.split("\n")
        page_lines = [line.strip().lower() for line in page_lines if line.strip()]
        strip_page_lines = [squeeze_text(line) for line in page_lines]
        valid_lines = []

        for i, (line, strip_line) in enumerate(zip(page_lines, strip_page_lines)):
            # 如果行长度太短，跳过
            if len(strip_line) <= 3:
                last_line = strip_line
                continue

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

            words = jieba.lcut(line)
            valid_words = [word.strip() for word in words if len(word.strip()) > 1]
            if valid_words and len(valid_words) > 5:
                words = valid_words

            test_str = "ωp=⌊ω"
            if test_str in strip_line:
                print(312, words)

            # 如果分词后的单词数量不足5个，使用所有单词
            if len(words) > 5:
                words = random.sample(words, 5)

            # 获取候选行索引
            candid_doc_line_idxes = set(
                chain(*[doc_inverted_index.get(word, []) for word in words])
            )
            if test_str in strip_line:
                print(324, candid_doc_line_idxes)
            # candid_caption_line_idxes = set(chain(*[caption_inverted_index.get(word, []) for word in words]))

            # 获取候选行文本
            candid_doc_lines = [strip_doc_lines[idx] for idx in candid_doc_line_idxes]
            # candid_caption_lines = [strip_caption_lines[idx] for idx in candid_caption_line_idxes]

            # 计算匹配分数
            doc_scores = [
                get_char_match_score(doc_line, strip_line)
                for doc_line in candid_doc_lines
            ]
            # caption_scores = [get_char_match_score(strip_caption_line, strip_line) for strip_caption_line in candid_caption_lines]

            if len(strip_line) > 30:
                if doc_scores and max(doc_scores) > 0.5:
                    valid_lines.append(line)
            else:
                if doc_scores and max(doc_scores) > 0.75:
                    valid_lines.append(line)

            last_line = strip_line

            if test_str in strip_line:
                print(340, strip_line)
                print(341, candid_doc_lines)
                print("ωp=ω·vpn(31)" in candid_doc_lines)
                print(351, get_char_match_score("ωp=ω·vpn(31)", strip_line))
                print(doc_scores)
                print(valid_lines[-1])

        lines_of_pages.append(valid_lines)

    out_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test/out"
    with open(f"{out_path}/valid_lines.txt", "w", encoding="utf-8") as f:
        for lines in lines_of_pages:
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
    for page_lines in lines_of_pages:
        # 如果该页没有行，则认为没有有效位置
        if not page_lines:
            page_start_positions.append(-1)
            page_end_positions.append(-1)
            continue

        # 获得该页的开始和结束行
        start_line = squeeze_text(page_lines[0])
        end_line = squeeze_text(page_lines[-1])

        # 计算开始行的匹配分数
        start_window_end = min(start_pointer + start_window_size, len(strip_doc_lines))
        start_scores = [
            get_char_match_score(doc_line, start_line)
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
                get_char_match_score(doc_line, start_line)
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
            get_char_match_score(doc_line, end_line)
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
            # 第一次没有匹配成功，则给个机会扩大窗口再来一次
            end_window_end = min(
                end_pointer + end_window_size + min_window_size, len(strip_doc_lines)
            )
            end_scores = [
                get_char_match_score(doc_line, end_line)
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

        print(231, page_start_positions[-1], max(start_scores), start_line)
        print(232, page_end_positions[-1], max(end_scores), end_line)

    print(392, page_start_positions)
    print(393, page_end_positions)

    # 根据分割位置拆分文档
    doc_text_by_pages, coinside_pages, bad_pages = get_page_result_by_span(
        doc_lines, page_start_positions, page_end_positions
    )

    return doc_text_by_pages, coinside_pages, bad_pages


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

    base_path = "/home/ninziwei/projects/nougat_ocr/nougat/dataset/test"
    args.md = f"{base_path}/2402.00041.mmd"
    args.pdf = f"{base_path}/2402.00041.pdf"
    args.figure = f"{base_path}/2402.00041.json"
    args.out = f"{base_path}/out"

    md = open(args.md, "r", encoding="utf-8").read().replace("\xa0", " ")
    pdf = pypdf.PdfReader(args.pdf)
    fig_info = json.load(open(args.figure, "r", encoding="utf-8"))

    pages, coinside_pages, bad_pages = split_markdown(md, pdf, fig_info)

    print(341, len(pages))
    with open(f"{args.out}/2402.00041.txt", "w", encoding="utf-8") as f:
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
