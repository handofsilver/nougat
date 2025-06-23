import jieba
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from nougat.dataset.split_utils.text_cleaner import squeeze_text
from nougat.dataset.split_utils.string_matcher import calculate_similarity_score
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
    content_type: Optional[str] = None  # 'ordered', 'unordered'
    local_index: Optional[int] = None  # 在ordered或unordered中的局部索引
    matched_content: Optional[str] = None  # 匹配的行内容


@dataclass
class PageMatchContext:
    """页面匹配上下文"""

    page_lines: List[str]
    strip_page_lines: List[str]
    words_by_line: List[List[str]]
    last_line: str = ""
    current_ordered_pointer: int = 0  # 用于窗口限制的ordered指针


@dataclass
class DualMatchResult:
    """双重匹配结果"""

    pdf_line_index: int
    pdf_line_content: str
    matched_doc_line_index: Optional[int] = None
    matched_doc_line_content: Optional[str] = None
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
    def preprocess_page_lines(
        page_lines: List[str], current_ordered_pointer: int
    ) -> PageMatchContext:
        """预处理页面行数据"""
        strip_page_lines = [squeeze_text(line) for line in page_lines]
        single_words_by_line = [jieba.lcut(line) for line in page_lines]
        words_by_line = [
            TextPreprocessor.build_pdf_words_for_match(words) for words in single_words_by_line
        ]

        return PageMatchContext(
            page_lines=page_lines,
            strip_page_lines=strip_page_lines,
            words_by_line=words_by_line,
            current_ordered_pointer=current_ordered_pointer,
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
        return len(strip_line) <= 8

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
        return tmp_line.isdigit()

    @staticmethod
    def should_accept_by_score(strip_line: str, max_score: float) -> bool:
        """根据分数判断是否接受"""
        if len(strip_line) > 30:
            return max_score > 0.5
        else:
            return max_score > 0.75

    @staticmethod
    def calculate_length_penalty(
        pdf_content: str, mmd_content: str, min_ratio: float = 0.03
    ) -> float:
        """
        计算长度比例惩罚，用于single_line_substring匹配

        Args:
            pdf_content: PDF行内容
            mmd_content: MMD行内容
            min_ratio: 最小长度比例阈值，默认0.03

        Returns:
            length_penalty: 长度惩罚因子 (0.1 到 1.0)
        """

        pdf_len = len(pdf_content.strip())
        mmd_len = len(mmd_content.strip())

        length_ratio = pdf_len / mmd_len

        if length_ratio >= min_ratio:
            return 1.0  # 长度比例合理，无惩罚
        else:
            # 线性惩罚：比例越小，惩罚越大
            penalty = max(length_ratio / min_ratio, 0.1)  # 最小惩罚因子为0.1
            return penalty


# ================================
# 5. 核心匹配器
# ================================


class LineMatcher:
    """行匹配器 - 统一使用双重索引方案，支持窗口限制"""

    def __init__(self, ordered_lines, unordered_lines, window_size: int = 10):
        """初始化匹配器"""
        self.ordered_lines = ordered_lines
        self.unordered_lines = unordered_lines
        self.window_size = window_size

        # 构建倒排索引
        ordered_text_lines = [line.content for line in ordered_lines]
        self.ordered_inverted_index = InvertedIndexBuilder.build_inverted_index(ordered_text_lines)

        unordered_text_lines = [line.content for line in unordered_lines]
        self.unordered_inverted_index = InvertedIndexBuilder.build_inverted_index(
            unordered_text_lines
        )

        print(
            f"🔍 双重索引匹配器初始化: 有序{len(ordered_lines)}行, 无序{len(unordered_lines)}行, 窗口{window_size}"
        )

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

        strip_pdf_line = MatchingRules.preprocess_abstract_line(strip_pdf_line)

        # 2. 规则匹配
        if MatchingRules.is_digit_line(strip_pdf_line):
            return MatchResult(is_valid=False)

        # 3. 子串匹配检查（包含精确匹配）
        substring_result = self._check_substring_match_with_priority_windowed(
            strip_pdf_line, context
        )
        if substring_result.is_valid:
            return substring_result

        # 4. 双重倒排索引匹配（兜底策略）
        return self._match_with_dual_inverted_index_windowed(strip_pdf_line, words, context)

    def _check_substring_matches(
        self, strip_pdf_line: str, content_lines: List
    ) -> List[MatchResult]:
        """通用子串匹配逻辑，返回MatchResult列表"""
        matches = []

        for content_line in content_lines:
            strip_doc_line = squeeze_text(content_line.content)
            match_type = None
            match_score = 1.0

            if strip_pdf_line == strip_doc_line:
                match_type = 'exact_substring'
            elif strip_pdf_line in strip_doc_line:
                match_type = 'single_line_substring'
                # 对single_line_substring应用长度比例惩罚
                length_penalty = MatchingRules.calculate_length_penalty(
                    strip_pdf_line, strip_doc_line
                )
                match_score = 1.0 * length_penalty
            elif (
                not MatchingRules.is_too_short(strip_doc_line) and strip_doc_line in strip_pdf_line
            ):
                match_type = 'reverse_substring'

            if match_type:
                # 检查惩罚后的分数是否仍然满足阈值
                if MatchingRules.should_accept_by_score(strip_pdf_line, match_score):
                    result = MatchResult(
                        is_valid=True,
                        matched_index=content_line.line_index,
                        local_index=content_line.local_index,
                        matched_content=content_line.content,
                        match_score=match_score,
                        match_type=match_type,
                        content_type=content_line.content_type,
                    )
                    matches.append(result)

        return matches

    def _check_substring_match_with_priority_windowed(
        self, strip_pdf_line: str, context: PageMatchContext
    ) -> MatchResult:
        """带优先级的子串匹配：优先ordered（按距离排序） > unordered"""
        # 获取所有匹配
        ordered_matches = self._check_substring_matches(strip_pdf_line, self.ordered_lines)
        unordered_matches = self._check_substring_matches(strip_pdf_line, self.unordered_lines)

        # 优先处理ordered匹配：按距离排序
        if ordered_matches:
            windowed_ordered_matches = [
                match
                for match in ordered_matches
                if match.local_index >= context.current_ordered_pointer - self.window_size
                and match.local_index <= context.current_ordered_pointer + self.window_size
            ]
            if not windowed_ordered_matches:
                return MatchResult(is_valid=False)

            ordered_matches = windowed_ordered_matches

            def distance_key(match: MatchResult, dist_penalty: int = self.window_size // 2):
                if match.local_index >= context.current_ordered_pointer:
                    return match.local_index - context.current_ordered_pointer
                else:
                    return context.current_ordered_pointer - match.local_index + dist_penalty

            ordered_matches.sort(key=distance_key)
            ordered_result = ordered_matches[0]

            self._update_ordered_pointer(context, ordered_result.local_index)
            return ordered_result

        # 如果没有ordered匹配，尝试unordered
        if unordered_matches:
            unordered_result = unordered_matches[0]
            return unordered_result

        return MatchResult(is_valid=False)

    def _match_with_dual_inverted_index_windowed(
        self, strip_pdf_line: str, words: List[str], context: PageMatchContext
    ) -> MatchResult:
        """使用双重倒排索引进行匹配（兜底策略，支持窗口限制）"""
        # 分别尝试ordered和unordered匹配
        ordered_result = self._match_with_index(
            strip_pdf_line,
            words,
            self.ordered_inverted_index,
            self.ordered_lines,
            'ordered',
            context,
        )
        unordered_result = self._match_with_index(
            strip_pdf_line,
            words,
            self.unordered_inverted_index,
            self.unordered_lines,
            'unordered',
            None,
        )

        # 选择最佳结果
        if not ordered_result.is_valid and not unordered_result.is_valid:
            return MatchResult(is_valid=False)
        if not ordered_result.is_valid and unordered_result.is_valid:
            return unordered_result
        if not unordered_result.is_valid and ordered_result.is_valid:
            self._update_ordered_pointer(context, ordered_result.local_index)
            return ordered_result

        # 两个都有效，选择分数更高的
        if (unordered_result.match_score or 0) > (ordered_result.match_score or 0):
            return unordered_result
        else:
            self._update_ordered_pointer(context, ordered_result.local_index)
            return ordered_result

    def _match_with_index(
        self,
        strip_pdf_line: str,
        words: List[str],
        inverted_index: Dict,
        content_lines: List,
        content_type: str,
        context: Optional[PageMatchContext],
    ) -> MatchResult:
        """使用倒排索引进行匹配，支持可选的窗口限制"""
        candidates = InvertedIndexBuilder.get_candidates_from_index(words, inverted_index)
        if not candidates:
            return MatchResult(is_valid=False)

        # 对ordered内容应用窗口过滤
        if content_type == 'ordered' and context:
            # current_ordered_pointer <= idx <= current_ordered_pointer + window_size
            windowed_candidates = [
                idx
                for idx in candidates
                if context.current_ordered_pointer <= idx
                and idx <= context.current_ordered_pointer + self.window_size
            ]
            if not windowed_candidates:
                return MatchResult(is_valid=False)

            candidates = windowed_candidates

        # 计算匹配分数
        scores = []
        for local_index in candidates:
            if local_index < len(content_lines):
                score = calculate_similarity_score(
                    content=content_lines[local_index].content, query=strip_pdf_line
                )
                scores.append(score)
                if score > 0.9:  # 早停优化
                    break

        if not scores:
            return MatchResult(is_valid=False)

        max_score = max(scores)
        if MatchingRules.should_accept_by_score(strip_pdf_line, max_score):
            best_idx = scores.index(max_score)
            local_index = candidates[best_idx]
            original_line_index = content_lines[local_index].line_index

            return MatchResult(
                is_valid=True,
                matched_index=original_line_index,
                local_index=local_index,
                matched_content=content_lines[local_index].content,
                match_score=max_score,
                match_type='dual_inverted',
                content_type=content_type,
            )

        return MatchResult(is_valid=False)

    def _update_ordered_pointer(self, context: PageMatchContext, new_pointer: int):
        """更新context中的ordered指针"""
        context.current_ordered_pointer = new_pointer


# ================================
# 6. 主要接口函数
# ================================


def filter_and_match_lines(pdf, ordered_lines, unordered_lines):
    """
    过滤和匹配PDF中的行与文档中的行（统一双重索引方案，支持窗口限制）

    Args:
        pdf: PDF文件对象
        ordered_lines: 有序内容行（ContentLine列表）
        unordered_lines: 无序内容行（ContentLine列表）

    Returns:
        (valid_lines_of_pages, index_mappings): 有效行列表和详细映射信息
    """
    # 初始化匹配器（可以调整window_size参数）
    matcher = LineMatcher(ordered_lines, unordered_lines, window_size=20)

    # 获取PDF页面数据
    raw_lines_of_pages = extract_lower_pdf_lines(pdf)

    # 处理每一页
    valid_lines_of_pages = []
    index_mappings = []

    current_ordered_pointer = 0
    for page_idx, page_lines in enumerate(raw_lines_of_pages):
        # 预处理当前页
        context = TextPreprocessor.preprocess_page_lines(page_lines, current_ordered_pointer)
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
                matched_doc_line_index=result.matched_index,
                matched_doc_line_content=result.matched_content,
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
        index_mappings.append(page_mappings)

        current_ordered_pointer = context.current_ordered_pointer
        # 为下一页保持ordered指针的连续性
        print(
            f"📄 页面 {page_idx}: 处理了 {len(page_lines)} 行PDF，{len(valid_lines)} 行有效，ordered指针: {context.current_ordered_pointer}"
        )

    return valid_lines_of_pages, index_mappings


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
