import jieba
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from nougat.dataset.split_utils.text_cleaner import squeeze_text
from nougat.dataset.split_utils.string_matcher import get_char_match_score
from nougat.dataset.split_utils.pdf_processor import extract_lower_pdf_lines

# ================================
# 1. 数据结构定义
# ================================


@dataclass
class MatchResult:
    """单行匹配结果"""

    is_valid: bool
    matched_index: Optional[int] = None
    match_score: Optional[float] = None
    match_type: Optional[str] = None  # 'exact', 'inverted', 'rule'
    content_type: Optional[str] = None  # 'ordered', 'unordered' - 新增


@dataclass
class PageMatchContext:
    """页面匹配上下文"""

    page_lines: List[str]
    strip_page_lines: List[str]
    words_by_line: List[List[str]]
    last_line: str = ""


@dataclass
class DualMatchResult:
    """双重匹配结果 - 新增"""

    pdf_line_index: int
    pdf_line_content: str
    matched_md_line_index: Optional[int] = None
    match_score: Optional[float] = None
    match_type: Optional[str] = None
    content_type: Optional[str] = None  # 'ordered', 'unordered'
    is_valid: bool = False


# ================================
# 2. 文本预处理模块
# ================================


class TextPreprocessor:
    """文本预处理器"""

    @staticmethod
    def build_pdf_words_for_match(words: List[str]) -> List[str]:
        """构建PDF文本的单词组合，用于匹配"""
        words = [word.strip() for word in words if len(word.strip())]
        if not words:
            return []

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

        return list(set(new_words))

    @staticmethod
    def build_markdown_words_for_index(words: List[str]) -> List[str]:
        """构建Markdown文本的单词组合，用于索引"""
        words = [word.strip() for word in words if len(word.strip())]
        if not words:
            return []

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

        return list(set(new_words))

    @staticmethod
    def preprocess_page_lines(page_lines: List[str]) -> PageMatchContext:
        """预处理页面行数据"""
        strip_page_lines = [squeeze_text(line) for line in page_lines]
        single_words_by_line = [jieba.lcut(line) for line in page_lines]
        words_by_line = [
            TextPreprocessor.build_pdf_words_for_match(words) for words in single_words_by_line
        ]

        return PageMatchContext(
            page_lines=page_lines, strip_page_lines=strip_page_lines, words_by_line=words_by_line
        )


# ================================
# 3. 倒排索引模块
# ================================


class InvertedIndexBuilder:
    """倒排索引构建器"""

    @staticmethod
    def build_inverted_index(lines: List[str]) -> Dict[str, List[int]]:
        """构建倒排索引，使用局部索引"""
        inverted_index = {}

        for local_index, line in enumerate(lines):
            words = jieba.lcut(line)
            new_words = TextPreprocessor.build_markdown_words_for_index(words)

            for new_word in new_words:
                inverted_index.setdefault(new_word, []).append(local_index)

        return inverted_index

    @staticmethod
    def get_candidates_from_index(
        words: List[str], inverted_index: Dict[str, List[int]], max_words: int = 5
    ) -> List[int]:
        """从倒排索引中获取候选行索引"""
        candidates = []
        valid_num = 0

        for word in words:
            if word in inverted_index:
                candidates.extend(inverted_index[word])
                valid_num += 1
            if valid_num >= max_words:
                break

        return list(set(candidates))


# ================================
# 4. 匹配规则模块
# ================================


class MatchingRules:
    """匹配规则集合"""

    @staticmethod
    def is_too_short(strip_line: str) -> bool:
        """判断行是否太短"""
        return len(strip_line) <= 4

    @staticmethod
    def preprocess_abstract_line(strip_line: str) -> str:
        """预处理abstract开头的行"""
        if strip_line.startswith("abstract") and len(strip_line) > 30:
            strip_line = strip_line[8:]
            strip_line = strip_line.strip(".—")
        return strip_line

    @staticmethod
    def is_digit_line(strip_line: str) -> bool:
        """判断是否为数字行"""
        tmp_line = re.sub(r"-|_| ", "", strip_line)
        tmp_line = tmp_line.strip("{}[]<>()（）")
        return tmp_line.isdigit()  # 如果去除标点符号后的PDF行文本是数字，则认为有效

    @staticmethod
    def is_short_alpha_line(strip_line: str) -> bool:
        """判断是否为短字母行"""
        tmp_line = re.sub(r"-|_| ", "", strip_line)
        tmp_line = tmp_line.strip("{}[]<>()（）")
        # 如果去除标点符号后的PDF行文本是字母，并且长度小于10，则认为有效
        return tmp_line.isalpha() and len(tmp_line) < 10

    @staticmethod
    def is_alpha_with_short_context(strip_line: str, last_line: str, next_line: str) -> bool:
        """判断是否为有短上下文的字母行"""
        tmp_line = re.sub(r"-|_| ", "", strip_line)
        tmp_line = tmp_line.strip("{}[]<>()（）")

        # 如果去除标点符号后的PDF行文本是字母，并且上一行或下一行长度小于15，则认为有效
        return tmp_line.isalpha() and (
            (last_line and len(last_line) < 15) or (next_line and len(next_line) < 15)
        )

    @staticmethod
    def should_accept_by_score(strip_line: str, max_score: float) -> bool:
        """根据分数判断是否接受"""
        if len(strip_line) > 30:
            return max_score > 0.5
        else:
            return max_score > 0.75


# ================================
# 5. 核心匹配器
# ================================


class LineMatcher:
    """行匹配器 - 统一使用双重索引方案"""

    def __init__(self, ordered_lines, unordered_lines):
        """初始化匹配器

        Args:
            ordered_lines: 有序内容行（ContentLine列表）
            unordered_lines: 无序内容行（ContentLine列表）
        """
        self.ordered_lines = ordered_lines
        self.unordered_lines = unordered_lines

        # 合并所有清理后的内容作为完整文档（用于子串匹配）
        all_content_lines = ordered_lines + unordered_lines
        self.strip_doc_lines = [line for line in all_content_lines if line.content.strip()]

        # 构建有序内容的倒排索引
        ordered_text_lines = [line.content for line in ordered_lines]
        self.ordered_inverted_index = InvertedIndexBuilder.build_inverted_index(ordered_text_lines)

        # 构建无序内容的倒排索引
        unordered_text_lines = [line.content for line in unordered_lines]
        self.unordered_inverted_index = InvertedIndexBuilder.build_inverted_index(
            unordered_text_lines
        )

        print(f"🔍 双重索引匹配器初始化:")
        print(f"   - 有序内容: {len(ordered_lines)} 行")
        print(f"   - 无序内容: {len(unordered_lines)} 行")
        print(f"   - 合并后总行数: {len(self.strip_doc_lines)} 行")

    def match_single_line(
        self,
        pdf_line: str,
        strip_pdf_line: str,
        words: List[str],
        context: PageMatchContext,
        line_index: int,
    ) -> MatchResult:
        """匹配单个PDF行"""

        # 1. 快速规则检查
        if MatchingRules.is_too_short(strip_pdf_line):
            return MatchResult(is_valid=False)

        # 2. 预处理特殊行
        strip_pdf_line = MatchingRules.preprocess_abstract_line(strip_pdf_line)

        # 3. 子串匹配检查（包含精确匹配）
        substring_result = self._check_substring_match(strip_pdf_line)
        if substring_result.is_valid:
            return substring_result

        # 4. 规则匹配
        if MatchingRules.is_digit_line(strip_pdf_line):
            return MatchResult(is_valid=False)  # 数字行跳过

        if MatchingRules.is_short_alpha_line(strip_pdf_line):
            return MatchResult(is_valid=False)  # 短字母行跳过

        # 获取上下文
        next_line = ""
        if line_index + 1 < len(context.strip_page_lines):
            next_line = context.strip_page_lines[line_index + 1]

        if MatchingRules.is_alpha_with_short_context(strip_pdf_line, context.last_line, next_line):
            return MatchResult(is_valid=False)  # 有短上下文的字母行跳过

        # 5. 双重倒排索引匹配
        return self._match_with_dual_inverted_index(strip_pdf_line, words)

    def _match_with_dual_inverted_index(self, strip_pdf_line: str, words: List[str]) -> MatchResult:
        """使用双重倒排索引进行匹配"""
        # 先尝试有序内容匹配
        ordered_result = self._match_with_index(
            strip_pdf_line, words, self.ordered_inverted_index, self.ordered_lines, 'ordered'
        )

        # 如果有序内容匹配成功且分数较高，直接返回
        if (
            ordered_result.is_valid
            and ordered_result.match_score
            and ordered_result.match_score > 0.8
        ):
            return ordered_result

        # 尝试无序内容匹配
        unordered_result = self._match_with_index(
            strip_pdf_line, words, self.unordered_inverted_index, self.unordered_lines, 'unordered'
        )

        # 选择更好的匹配结果
        if not ordered_result.is_valid and not unordered_result.is_valid:
            return MatchResult(is_valid=False)
        elif not ordered_result.is_valid:
            return unordered_result
        elif not unordered_result.is_valid:
            return ordered_result
        else:
            # 两个都有效，选择分数更高的
            if (unordered_result.match_score or 0) > (ordered_result.match_score or 0):
                return unordered_result
            else:
                return ordered_result

    def _match_with_index(
        self,
        strip_pdf_line: str,
        words: List[str],
        inverted_index: Dict,
        content_lines: List,  # List[ContentLine]
        content_type: str,
    ) -> MatchResult:
        """使用指定的倒排索引进行匹配"""
        # 获取候选行
        candidates = InvertedIndexBuilder.get_candidates_from_index(words, inverted_index)

        if not candidates:
            return MatchResult(is_valid=False)

        # 计算匹配分数
        scores = []
        for idx in candidates:
            if idx < len(content_lines):
                content_line = content_lines[idx]
                score = get_char_match_score(content=content_line.content, query=strip_pdf_line)
                scores.append(score)
                # 早停优化
                if score > 0.9:
                    break

        if not scores:
            return MatchResult(is_valid=False)

        max_score = max(scores)
        is_valid = MatchingRules.should_accept_by_score(strip_pdf_line, max_score)

        if is_valid:
            best_idx = scores.index(max_score)
            local_matched_index = candidates[best_idx]
            # 转换为原始文档中的行索引
            original_line_index = content_lines[local_matched_index].line_index

            return MatchResult(
                is_valid=True,
                matched_index=original_line_index,
                match_score=max_score,
                match_type='dual_inverted',
                content_type=content_type,
            )

        return MatchResult(is_valid=False)

    def _check_substring_match(self, strip_pdf_line: str) -> MatchResult:
        """检查子串匹配，区分单行匹配和跨行匹配"""

        # 1. 检查是否为某个文档行的完全匹配或子串
        for idx, content_line in enumerate(self.strip_doc_lines):
            strip_doc_line = squeeze_text(content_line.content).lower()

            if strip_pdf_line == strip_doc_line:
                # 完全匹配
                return MatchResult(
                    is_valid=True,
                    matched_index=content_line.line_index,
                    match_score=1.0,
                    match_type='exact_substring',
                    content_type=content_line.content_type,
                )
            elif strip_pdf_line in strip_doc_line:
                # 单行子串匹配
                return MatchResult(
                    is_valid=True,
                    matched_index=content_line.line_index,
                    match_score=1.0,
                    match_type='single_line_substring',
                    content_type=content_line.content_type,
                )
            elif (
                not MatchingRules.is_too_short(strip_doc_line) and strip_doc_line in strip_pdf_line
            ):
                # PDF行包含整个文档行
                return MatchResult(
                    is_valid=True,
                    matched_index=content_line.line_index,
                    match_score=1.0,
                    match_type='reverse_substring',
                    content_type=content_line.content_type,
                )

        return MatchResult(is_valid=False)


# ================================
# 6. 主要接口函数
# ================================


def filter_and_match_lines(pdf, ordered_lines, unordered_lines):
    """
    过滤和匹配PDF中的行与文档中的行（统一双重索引方案）

    Args:
        pdf: PDF文件对象
        ordered_lines: 有序内容行（ContentLine列表）
        unordered_lines: 无序内容行（ContentLine列表）

    Returns:
        (valid_lines_of_pages, detailed_mappings): 有效行列表和详细映射信息
    """
    # 初始化匹配器
    matcher = LineMatcher(ordered_lines, unordered_lines)

    # 获取PDF页面数据
    raw_lines_of_pages = extract_lower_pdf_lines(pdf)

    # 处理每一页
    valid_lines_of_pages = []
    detailed_mappings = []

    for page_idx, page_lines in enumerate(raw_lines_of_pages):
        # 预处理当前页
        context = TextPreprocessor.preprocess_page_lines(page_lines)
        valid_lines = []
        page_mappings = []

        # 匹配每一行
        for i, (pdf_line, strip_pdf_line, words) in enumerate(
            zip(context.page_lines, context.strip_page_lines, context.words_by_line)
        ):

            # 执行匹配
            result = matcher.match_single_line(pdf_line, strip_pdf_line, words, context, i)

            # 记录详细映射
            mapping = DualMatchResult(
                pdf_line_index=i,
                pdf_line_content=pdf_line,
                matched_md_line_index=result.matched_index,
                match_score=result.match_score,
                match_type=result.match_type,
                content_type=result.content_type,
                is_valid=result.is_valid,
            )
            page_mappings.append(mapping)

            # 记录有效行
            if result.is_valid:
                valid_lines.append(pdf_line)

            # 更新上下文
            context.last_line = strip_pdf_line

        valid_lines_of_pages.append(valid_lines)
        detailed_mappings.append(page_mappings)

    return valid_lines_of_pages, detailed_mappings


# ================================
# 7. 向后兼容的函数别名
# ================================


def build_inverted_index(lines: List[str]) -> Dict[str, List[int]]:
    """向后兼容的倒排索引构建函数"""
    return InvertedIndexBuilder.build_inverted_index(lines)


def build_pdf_words_for_match(words: List[str]) -> List[str]:
    """向后兼容的PDF单词构建函数"""
    return TextPreprocessor.build_pdf_words_for_match(words)


def build_markdown_words_for_index(words: List[str]) -> List[str]:
    """向后兼容的Markdown单词构建函数"""
    return TextPreprocessor.build_markdown_words_for_index(words)
