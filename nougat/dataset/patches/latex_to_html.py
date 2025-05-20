import re


def latex_to_html(latex: str) -> str:
    """
    将一段 LaTeX 格式的参考文献条目，转换成适合 HTML 的文本。
    这里把所有小的转换步骤依次拼起来。
    """
    # 先把空白合并一下
    latex = re.sub(r"\s+", " ", latex.strip())

    # 先去掉纯排版命令，比如 `\/`
    latex = _remove_latex_spacing_commands(latex)

    # 把 `{\sc Something}` 这种改为 `\sc{Something}`；同理 `{\em ...}`
    latex = _convert_braces_for_sc_and_em(latex)

    # 把 `\em{` -> `\emph{`
    latex = _unify_em_to_emph(latex)

    # 把 `\newblock` 去掉/改空格
    latex = _convert_newblock(latex)

    # 处理 \url{ ... }，只保留里面的内容
    latex = _parse_url_with_stack(latex)

    # 处理 \emph{...} 嵌套
    latex = _parse_emph_with_stack(latex)

    # 把 \sc{...} 变成大写
    latex = _parse_sc_with_stack(latex)

    # 去掉那些只包含 ASCII/标点/数字/空格之类的单层大括号
    latex = _remove_useless_braces(latex)

    # 做特殊字符替换
    latex = _replace_special_chars(latex)

    return latex


def _remove_latex_spacing_commands(text: str) -> str:
    """
    去掉常见的只用于排版的命令，比如 \/, \!, \,, \:, \; 等。
    如果你还遇到更多类似的命令，可以依葫芦画瓢往下加。
    """
    # 这里演示最简，直接把它们干掉
    spacing_cmds = [r"\\/", r"\\!", r"\\,", r"\\;", r"\\:", r"\\\.", r"\\\ "]  # 可按需增补
    for cmd in spacing_cmds:
        text = re.sub(cmd, "", text)
    return text


def _convert_braces_for_sc_and_em(text: str) -> str:
    """
    将 `{\sc Something}` 统一改为 `\sc{Something}`，
    将 `{\em Something}` 统一改为 `\em{Something}`，
    这样后续用同一种解析方式就方便多了。
    也可以将 `\em` 全部替换成 `\emph`，根据喜好来。
    """
    # 比如匹配 `{\sc ...}`
    text = re.sub(r"\{\s*\\sc\s+([^}]+)\}", r"\\sc{\1}", text)
    text = re.sub(r"\{\s*\\em\s+([^}]+)\}", r"\\em{\1}", text)
    text = re.sub(r"\{\s*\\emph\s+([^}]+)\}", r"\\emph{\1}", text)
    return text


def _unify_em_to_emph(text: str) -> str:
    """
    如果你希望把 `\em` 和 `\emph` 统一处理，
    可以先把所有 \em{...} 全部替换成 \emph{...}。
    """
    text = re.sub(r"\\em\s*\{", r"\\emph{", text)
    return text


def _parse_emph_with_stack(text: str) -> str:
    """
    用手动堆栈的方式，递归处理 \emph{ ... }，支持嵌套花括号。
    最后把内容变成 <em>...</em>。
    """
    pattern = r"\\emph\s*\{"
    while True:
        match = re.search(pattern, text)
        if not match:
            break

        start_idx = match.start()

        # 从 `\emph{` 这个大括号开始
        brace_open_idx = match.end() - 1  # 指向 '{'
        brace_stack = ["{"]

        i = brace_open_idx
        n = len(text)
        while i < n - 1 and brace_stack:
            i += 1
            if text[i] == "{":
                brace_stack.append("{")
            elif text[i] == "}":
                brace_stack.pop()

        if brace_stack:
            # 如果到最后还没匹配完，说明花括号不完整；直接break或raise都行
            break

        # i 现在是与起始 '{' 对应的 '}'
        inner_content = text[brace_open_idx + 1 : i]

        # 递归处理内层可能还包含 \emph{...}
        inner_content_converted = _parse_emph_with_stack(inner_content)

        # 用 <em> 包裹起来
        replace_str = f"<em>{inner_content_converted}</em>"

        text = text[:start_idx] + replace_str + text[i + 1 :]

    return text


def _parse_sc_with_stack(text: str) -> str:
    pattern = r"\\sc\s*\{"
    while True:
        match = re.search(pattern, text)
        if not match:
            break

        start_idx = match.start()  # 指向 `\sc{`
        brace_open_idx = match.end() - 1  # 指向 `{`
        brace_stack = ["{"]

        i = brace_open_idx
        n = len(text)

        while i < n - 1 and brace_stack:
            i += 1
            if text[i] == "{":
                brace_stack.append("{")
            elif text[i] == "}":
                brace_stack.pop()

        if brace_stack:
            # 花括号不匹配，就直接 break 或 raise
            break

        # i 现在就是与起始 '{' 相配对的 '}'
        inner_content = text[brace_open_idx + 1 : i]

        # 递归处理里头，假如还要考虑下一级 \sc{ } 或别的命令，就继续 parse
        # 如果你只想单纯转大写，这里也可以一股脑 upper()
        # 但要注意，如果还有 \emph{...} 之类，也得先做处理
        # 这里示例：我先把嵌套的 \sc 也处理掉，然后再 upper
        inner_converted = _parse_sc_with_stack(inner_content)
        inner_upper = inner_converted.upper()

        # 替换
        text = text[:start_idx] + inner_upper + text[i + 1 :]
    return text


def _parse_url_with_stack(text: str) -> str:
    pattern = r"\\url\s*\{"
    while True:
        match = re.search(pattern, text)
        if not match:
            break

        start_idx = match.start()  # `\url`
        brace_open_idx = match.end() - 1  # 指向 '{'
        brace_stack = ["{"]

        i = brace_open_idx
        n = len(text)

        while i < n - 1 and brace_stack:
            i += 1
            if text[i] == "{":
                brace_stack.append("{")
            elif text[i] == "}":
                brace_stack.pop()

        if brace_stack:
            break

        inner_content = text[brace_open_idx + 1 : i]

        # 递归处理内部可能也有 \url{...}？（很罕见，但反正一视同仁）
        inner_converted = _parse_url_with_stack(inner_content)

        # 只保留内容
        replaced_str = inner_converted

        text = text[:start_idx] + replaced_str + text[i + 1 :]

    return text


def _convert_newblock(text: str) -> str:
    """
    LaTeX 的 \\newblock 通常是换行/段落的意思，
    这里简单替换成空格或者直接删掉。
    """
    text = re.sub(r"\\newblock\s*", " ", text)
    return text


def _replace_special_chars(text: str) -> str:
    """
    这里演示一些特殊字符处理，
    可根据实际需要往里扩充。
    """
    text = re.sub(r"~", " ", text)
    text = re.sub(r"(?<!`)``(?!`)", '"', text)
    text = re.sub(r"(?<!')''(?!')", '"', text)

    text = re.sub(r"{\\&}", "&amp;", text)
    text = re.sub(r"{\\\'I}", "Í", text)
    text = re.sub(r'{\\"o}', "ö", text)
    text = re.sub(r'{\\"a}', "ä", text)
    text = re.sub(r"{\\\'u}", "ú", text)
    text = re.sub(r"{\\\'e}", "é", text)
    text = re.sub(r"{\\\'a}", "á", text)
    text = re.sub(r"{\\\'o}", "ó", text)
    text = re.sub(r"{\\\'i}", "í", text)
    text = re.sub(r"{\\\'E}", "É", text)
    text = re.sub(r"{\\\'A}", "Á", text)
    text = re.sub(r"{\\\'O}", "Ó", text)
    text = re.sub(r"{\\\'I}", "Í", text)
    text = re.sub(r"{\\\'U}", "Ú", text)
    text = re.sub(r"{\\\'c}", "ç", text)
    text = re.sub(r"{\\\'C}", "Ç", text)
    text = re.sub(r"{\\\'n}", "ñ", text)
    text = re.sub(r"{\\\'N}", "Ñ", text)

    return text


def _remove_useless_braces(text: str) -> str:
    """
    去掉那些只包含 ASCII/标点/数字/空格之类的单层大括号。
    例如 "Latent {D}irichlet" -> "Latent Dirichlet"
         "Relation between {PLSA} and {NMF}" -> "Relation between PLSA and NMF"
    会循环替换，直到没得可替换。
    """
    # 这里用一个 while 循环，反复移除可去掉的 brace。
    pattern = re.compile(r"\{([A-Za-z0-9_.,;:\-\'\"\(\)\!\?\+\= /]+)\}")
    # 说明：
    # - [A-Za-z0-9_.,;:\-\'\"\(\)\!\?\+\= /]+ 是“常见”的字符范围。可根据需要再扩充。
    # - 没包括反斜线 '\\'、花括号等，所以不会误删掉有命令或嵌套的场景。

    while True:
        new_text = pattern.sub(r"\1", text)
        if new_text == text:
            break
        text = new_text

    return text


if __name__ == "__main__":
    # 测试一下
    latex = r"""
    \bibitem{Yamada.2006}
    H.~Yamada and S.~Hirose, ``Study on the 3{D} shape of active cord mechanism,''
    in \emph{IEEE International Conference on Robotics and Automation}, 2006, pp.
    2890--2895.

    \bibitem{Mochiyama.1999}
    H.~Mochiyama, E.~Shimemura, and H.~Kobayashi, ``Shape control of manipulators
    with hyper degrees of freedom,'' \emph{The International Journal of Robotics
    Research}, vol.~18, no.~6, pp. 584--600, 1999.
    """
    print(latex_to_html(latex))
