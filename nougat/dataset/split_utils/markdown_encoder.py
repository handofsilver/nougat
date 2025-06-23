import re
from typing import List

# 希腊字母
greek_letters = {
    r"\\alpha": "α",
    r"\\beta": "β",
    r"\\gamma": "γ",
    r"\\delta": "δ",
    r"\\epsilon": "ε",
    r"\\zeta": "ζ",
    r"\\eta": "η",
    r"\\theta": "θ",
    r"\\iota": "ι",
    r"\\kappa": "κ",
    r"\\lambda": "λ",
    r"\\mu": "μ",
    r"\\nu": "ν",
    r"\\xi": "ξ",
    r"\\omicron": "ο",
    r"\\pi": "π",
    r"\\rho": "ρ",
    r"\\sigma": "σ",
    r"\\tau": "τ",
    r"\\upsilon": "υ",
    r"\\phi": "φ",
    r"\\chi": "χ",
    r"\\psi": "ψ",
    r"\\omega": "ω",
    r"\\Gamma": "Γ",
    r"\\Delta": "Δ",
    r"\\Theta": "Θ",
    r"\\Lambda": "Λ",
    r"\\Xi": "Ξ",
    r"\\Pi": "Π",
    r"\\Sigma": "Σ",
    r"\\Phi": "Φ",
    r"\\Psi": "Ψ",
    r"\\Omega": "Ω",
    r"\\varepsilon": "ε",
    r"\\vartheta": "ϑ",
    r"\\varpi": "ϖ",
    r"\\varrho": "ϱ",
    r"\\varsigma": "ς",
    r"\\varphi": "φ",
}

# 数学运算符
math_operators = {
    r"\\times": "×",
    r"\\div": "÷",
    r"\\pm": "±",
    r"\\mp": "∓",
    r"\\cdot": "·",
    r"\\ast": "∗",
    r"\\star": "⋆",
    r"\\circ": "∘",
    r"\\bullet": "•",
    r"\\oplus": "⊕",
    r"\\otimes": "⊗",
    r"\\odot": "⊙",
    r"\\cap": "∩",
    r"\\cup": "∪",
    r"\\wedge": "∧",
    r"\\vee": "∨",
    r"\\dagger": "†",
    r"\\ddagger": "‡",
    r"\\amalg": "⨿",
    r"\\setminus": "∖",
    r"\\wr": "≀",
    r"\\diamond": "⋄",
    r"\\bigtriangleup": "△",
    r"\\bigtriangledown": "▽",
    r"\\triangleleft": "◁",
    r"\\triangleright": "▷",
    r"\\lhd": "⊲",
    r"\\rhd": "⊳",
    r"\\unlhd": "⊴",
    r"\\unrhd": "⊵",
}

# 比较运算符和关系
comparison_operators = {
    r"\\leq": "≤",
    r"\\geq": "≥",
    r"\\neq": "≠",
    r"\\approx": "≈",
    r"\\equiv": "≡",
    r"\\cong": "≅",
    r"\\sim": "∼",
    r"\\simeq": "≃",
    r"\\subset": "⊂",
    r"\\supset": "⊃",
    r"\\subseteq": "⊆",
    r"\\supseteq": "⊇",
    r"\\in": "∈",
    r"\\ni": "∋",
    r"\\notin": "∉",
    r"\\propto": "∝",
    r"\\parallel": "∥",
    r"\\perp": "⊥",
    r"\\mid": "∣",
    r"\\prec": "≺",
    r"\\succ": "≻",
    r"\\preceq": "⪯",
    r"\\succeq": "⪰",
    r"\\ll": "≪",
    r"\\gg": "≫",
    r"\\doteq": "≐",
    r"\\bowtie": "⋈",
    r"\\asymp": "≍",
    r"\\vdash": "⊢",
    r"\\dashv": "⊣",
    r"\\models": "⊨",
}

# 箭头和范围符号
arrows = {
    r"\\leftarrow": "←",
    r"\\rightarrow": "→",
    r"\\Leftarrow": "⇐",
    r"\\Rightarrow": "⇒",
    r"\\leftrightarrow": "↔",
    r"\\Leftrightarrow": "⇔",
    r"\\mapsto": "↦",
    r"\\uparrow": "↑",
    r"\\downarrow": "↓",
    r"\\updownarrow": "↕",
    r"\\nearrow": "↗",
    r"\\searrow": "↘",
    r"\\swarrow": "↙",
    r"\\nwarrow": "↖",
    r"\\longleftarrow": "⟵",
    r"\\longrightarrow": "⟶",
    r"\\Longleftarrow": "⟸",
    r"\\Longrightarrow": "⟹",
    r"\\longleftrightarrow": "⟷",
    r"\\Longleftrightarrow": "⟺",
    r"\\longmapsto": "⟼",
    r"\\hookrightarrow": "↪",
    r"\\hookleftarrow": "↩",
}

# 大型运算符
big_operators = {
    r"\\sum": "∑",
    r"\\prod": "∏",
    r"\\coprod": "∐",
    r"\\int": "∫",
    r"\\oint": "∮",
    r"\\bigcap": "⋂",
    r"\\bigcup": "⋃",
    r"\\bigvee": "⋁",
    r"\\bigwedge": "⋀",
    r"\\bigoplus": "⨁",
    r"\\bigotimes": "⨂",
    r"\\iint": "∬",
    r"\\iiint": "∭",
    r"\\iiiint": "⨌",
}

# 分式、根号等
fractions = {
    r"\\frac12": "½",
    r"\\frac14": "¼",
    r"\\frac34": "¾",
    r"\\frac13": "⅓",
    r"\\frac23": "⅔",
    r"\\frac15": "⅕",
    r"\\frac25": "⅖",
    r"\\frac35": "⅗",
    r"\\frac45": "⅘",
    r"\\frac16": "⅙",
    r"\\frac56": "⅚",
    r"\\frac18": "⅛",
    r"\\frac38": "⅜",
    r"\\frac58": "⅝",
    r"\\frac78": "⅞",
    r"\\sqrt": "√",
}

# 其他数学符号
other_symbols = {
    r"\\infty": "∞",
    r"\\nabla": "∇",
    r"\\partial": "∂",
    r"\\forall": "∀",
    r"\\exists": "∃",
    r"\\nexists": "∄",
    r"\\emptyset": "∅",
    r"\\varnothing": "∅",
    r"\\therefore": "∴",
    r"\\because": "∵",
    r"\\ldots": "…",
    r"\\cdots": "⋯",
    r"\\vdots": "⋮",
    r"\\ddots": "⋱",
    r"\\aleph": "ℵ",
    r"\\Re": "ℜ",
    r"\\Im": "ℑ",
    r"\\wp": "℘",
    r"\\prime": "′",
    r"\\backprime": "‵",
    r"\\square": "□",
    r"\\blacksquare": "■",
    r"\\bigcirc": "○",
    r"\\blacktriangledown": "▼",
    r"\\blacktriangle": "▲",
    r"\\blacktriangleright": "▶",
    r"\\blacktriangleleft": "◀",
    r"\\diamondsuit": "♦",
    r"\\heartsuit": "♥",
    r"\\clubsuit": "♣",
    r"\\spadesuit": "♠",
    r"\\neg": "¬",
    r"\\flat": "♭",
    r"\\natural": "♮",
    r"\\sharp": "♯",
    r"\\angle": "∠",
    r"\\measuredangle": "∡",
    r"\\sphericalangle": "∢",
    r"\\top": "⊤",
    r"\\bot": "⊥",
    r"\\ell": "ℓ",
    r"\\hbar": "ℏ",
    r"\\triangle": "△",
    r"\\vartriangle": "△",
    r"\\triangledown": "▽",
}


def encode_math_symbol(text):
    """
    将文本中用转义符描述的数学符号尽可能还原为对应的Unicode字符
    例如希腊字母、求和、积分、逻辑符号等
    """
    # 合并所有替换规则
    all_replacements = {}
    all_replacements.update(greek_letters)
    all_replacements.update(math_operators)
    all_replacements.update(comparison_operators)
    all_replacements.update(arrows)
    all_replacements.update(big_operators)
    all_replacements.update(fractions)
    all_replacements.update(other_symbols)

    # 应用所有替换规则
    for pattern, replacement in all_replacements.items():
        text = re.sub(pattern, replacement, text)

    # 处理上下标 (简单情况)
    text = re.sub(r"\^(\d)", r"^\1", text)  # 上标数字
    text = re.sub(r"_(\d)", r"_\1", text)  # 下标数字

    # 移除数学环境标记
    text = re.sub(r"\\\(|\\\)", "", text)

    return text


def clear_semantic_symbol(text):
    """
    清洗一些常见的md公式符号
    """
    # 临时替换转义的括号，以便保留它们
    escaped_brackets = {
        r"\{": "%%LEFT_BRACE%%",
        r"\}": "%%RIGHT_BRACE%%",
        r"\[": "%%LEFT_BRACKET%%",
        r"\]": "%%RIGHT_BRACKET%%",
        r"\(": "%%LEFT_PAREN%%",
        r"\)": "%%RIGHT_PAREN%%",
    }

    # 保留转义括号
    for pattern, replacement in escaped_brackets.items():
        text = re.sub(re.escape(pattern), replacement, text)

    # 处理数学字体样式命令，保留括号内的内容
    text = re.sub(r"\\math(?:frak|cal|bb|sf|tt|bf|it|rm|scr|normal)\{([^}]*)\}", r"\1", text)

    # 处理\operatorname和\operatorname*命令
    text = re.sub(r"\\operatorname\*?\{([^}]*)\}", r"\1", text)

    # 处理空格命令
    text = re.sub(r"\\(?:enspace|quad|qquad|,|thinspace|:|medspace|;|thickspace)", " ", text)

    # 处理其他结构化命令，保留内容
    text = re.sub(
        r"\\(?:text|mbox|displaystyle|textstyle|scriptstyle|scriptscriptstyle)\{([^}]*)\}",
        r"\1",
        text,
    )
    text = re.sub(r"\\(?:mathop|mathrel|mathbin)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:limits|nolimits)", "", text)
    text = re.sub(r"\\(?:left|right)(.)", r"\1", text)
    text = re.sub(r"\\(?:overset|underset|stackrel)\{([^}]*)\}\{([^}]*)\}", r"\2", text)

    # 处理通用的数学命令格式 \command{content}
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)  # 处理无括号的命令如\alpha, \beta等

    # 移除数学标记的各种括号，保留带转义符的括号，移除掉表示数学公式编码的普通括号
    text = re.sub(r"\\", "", text)

    # 改进：递归处理花括号，直到没有更多变化
    old_text = ""
    while old_text != text:
        old_text = text
        # 处理上标中的花括号
        text = re.sub(r"\^\{([^{}]*)\}", r"^\1", text)
        # 处理下标中的花括号
        text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
        # 移除嵌套的花括号
        text = re.sub(r"\{([^{}]*)\}", r"\1", text)

    # 处理上标（更通用的方式，不仅仅是数字）
    text = re.sub(r"\^(\(.*?\)|\w)", r"\1", text)
    text = re.sub(r"\^", "", text)  # 清除剩余的上标符号

    # 处理下标
    text = re.sub(r"_(\(.*?\)|\w)", r"\1", text)
    text = re.sub(r"_", "", text)  # 清除剩余的下标符号

    # 将临时标记恢复为原始括号
    reverse_escaped_brackets = {
        "%%LEFT_BRACE%%": "{",
        "%%RIGHT_BRACE%%": "}",
        "%%LEFT_BRACKET%%": "[",
        "%%RIGHT_BRACKET%%": "]",
        "%%LEFT_PAREN%%": "(",
        "%%RIGHT_PAREN%%": ")",
    }

    for pattern, replacement in reverse_escaped_brackets.items():
        text = text.replace(pattern, replacement)

    return text


def encode_formula_in_markdown(doc_lines: List[str], debug=False):
    """
    将md中的公式尽可能转换为编译后的文本
    1. 根据转义方括号或圆括号提取出doc中的所有公式
    2. 调用 encode_math_symbol 将公式中的转义符尽可能还原为对应的Unicode字符
    3. 调用 clear_semantic_symbol 清洗公式中用来表示语法的特殊字符
    4. 将doc中的公式替换为编译后的文本
    5. 返回编译后的文本
    """

    doc = "\n".join(doc_lines)

    def process_formula(match):
        formula = match.group(1)  # 提取公式内容
        # 处理公式内容
        encoded_formula = encode_math_symbol(formula)
        simplified_formula = clear_semantic_symbol(encoded_formula)
        if debug:
            print(f"原始公式: {formula}")
            print(f"编译后公式: {simplified_formula}")
        return simplified_formula

    # 定义替换模式
    patterns = [
        # 单独成行的公式 \[...\]
        (r"\\\[(.*?)\\\]", process_formula),
        # 行内公式 \(...\)
        (r"\\\((.*?)\\\)", process_formula),
        # 单行内公式 $...$
        (r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", process_formula),
        # 多行公式 $$...$$
        (r"\$\$(.*?)\$\$", process_formula),
    ]

    # 应用所有替换
    for pattern, replacement_func in patterns:
        doc = re.sub(pattern, lambda m: replacement_func(m), doc, flags=re.DOTALL)

    return doc.split("\n")


if __name__ == "__main__":
    txt = r'such as \(\|{\mathbf{Z}}\odot({\mathbf{Y}}-{\mathbf{B}}{\mathbf{S}}^{({\mathcal{L}})})\|\)", which will influence the learned and matrices via backpropagation.'
    print(encode_formula_in_markdown([txt]))
