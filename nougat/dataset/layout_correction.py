"""
Optional first-page layout correction using layout_parser (DiT visual detection).

Original design (from pdf_parser/TitleAuthorDetector):
  - For each PDF, scan up to `max_depth` pages to find a "title page"
  - Run DiT on those pages, look for 'title' and 'author' detections
  - If found, extract the text from that bbox region (via OCR on the cropped image)
  - Return {pdf_path: {title: [...], author: [...], note: [...], page_num: N}}

Integration into nougat data pipeline:
  - We already HAVE the per-page MMD text (from split_htmls_to_pages), so we don't need
    OCR on cropped images. The question is: does the MMD already contain the correct
    semantic tags ([TITLE], [AUTHOR])?
  - The problem is: LaTeXML sometimes fails to produce proper title/author sections
    (e.g. when \\maketitle is missing, or the .tex uses non-standard frontmatter).
    In such cases the MMD has the TEXT CONTENT but with [TEXT] tags instead of [TITLE]/[AUTHOR].
  - So the correction is: "the visual detector sees a title region, but the MMD has no
    [TITLE] tag => retag the positionally-matching [TEXT] as [TITLE]". The text content
    is NOT injected—it's already there, just mistagged.

Current implementation:
  - Only runs on the first page (01.mmd + 01.png)
  - DiT detects 20 categories; only a configurable subset triggers retagging
  - Default correctable set: TITLE, AUTHOR (matching the original TitleAuthorDetector scope)
  - Easily extensible: add tags to CORRECTABLE_TAGS to enable more categories

Requires layout_parser dependencies (detectron2, DiT).
Called from split_htmls_to_pages.process_paper() when --layout-parser is set.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

logger = logging.getLogger(__name__)

VISUAL_LABEL_TO_MMD_TAG = {
    "文章标题": "TITLE",    "title": "TITLE",
    "作者": "AUTHOR",       "author": "AUTHOR",
    "副标题": "ABSTRACT",   "assistant_title": "ABSTRACT",
    "正文": "TEXT",          "text": "TEXT",
    "子标题": "SECTION",    "subtitle": "SECTION",
    "公式": "EQUATION",     "formula": "EQUATION",
    "图片": "FIGURE",       "figure": "FIGURE",
    "图片标题": "CAPTION",  "figure_title": "CAPTION",
    "表格": "TABLE",        "table": "TABLE",
    "表格标题": "CAPTION",  "table_title": "CAPTION",
    "注释": "FOOTNOTE",     "bottom_note": "FOOTNOTE",
    "算法": "ALGORITHM",    "algorithm": "ALGORITHM",
    "参考文献": "BIBLIOGRAPHY", "reference": "BIBLIOGRAPHY",
}

CORRECTABLE_TAGS: Set[str] = {"TITLE", "AUTHOR"}

TAG_PATTERN = re.compile(r"^\[([A-Z_]+)(?::[^\]]*)?\].*\[END_\1\]$")


def _extract_mmd_tags(page_content: str) -> List[str]:
    tags = []
    for line in page_content.split("\n"):
        line = line.strip()
        m = TAG_PATTERN.match(line)
        if m:
            tags.append(m.group(1))
    return tags


def _get_visual_tag_sequence(detected_elements: List[Dict]) -> List[str]:
    sorted_elements = sorted(detected_elements, key=lambda e: e.get("y", 0))
    tags = []
    for elem in sorted_elements:
        label = elem.get("label", "")
        mmd_tag = VISUAL_LABEL_TO_MMD_TAG.get(label)
        if mmd_tag:
            tags.append(mmd_tag)
    return tags


def _retag_first_matching_text(
    page_content: str, new_tag: str, after_tag: str = None
) -> Tuple[str, str, str]:
    """
    Find the first [TEXT]...[END_TEXT] line (optionally after a line containing [after_tag])
    and retag it as [new_tag]...[END_new_tag].

    Returns (new_page_content, line_before, line_after) for logging.
    If no change, returns (page_content, "", "").
    """
    lines = page_content.split("\n")
    past_anchor = after_tag is None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not past_anchor:
            if f"[{after_tag}]" in stripped:
                past_anchor = True
            continue
        if stripped.startswith("[TEXT]") and stripped.endswith("[END_TEXT]"):
            inner = stripped[len("[TEXT]"):-len("[END_TEXT]")]
            new_line = f"[{new_tag}]{inner}[END_{new_tag}]"
            lines[i] = new_line
            return "\n".join(lines), stripped, new_line
    return page_content, "", ""


def _write_log(line: str, log_file=None) -> None:
    """Write a line to log_file only (no console)."""
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def correct_first_page(
    page_content: str,
    first_page_image_path: str,
    layout_parser,
    correctable_tags: Set[str] = None,
    mmd_file_path: Optional[str] = None,
    log_file_path: Optional[str] = None,
) -> str:
    """
    Apply layout-parser-based correction to the first page's MMD content.

    When log_file_path is set, all layout_correction output is written to that file
    only (no console). Otherwise nothing is printed for the detailed summary.

    Args:
        page_content: The first page's MMD text (with semantic tags).
        first_page_image_path: Path to the rendered first page .png.
        layout_parser: An initialized LayoutParserApi instance.
        correctable_tags: Set of MMD tags to correct. Default: {"TITLE", "AUTHOR"}.
        mmd_file_path: Optional path to the .mmd file (for logging).
        log_file_path: Optional path to .log file; when set, write all detail there (no console).
    """
    from PIL import Image

    if correctable_tags is None:
        correctable_tags = CORRECTABLE_TAGS

    log_file = None
    if log_file_path:
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_file_path, "w", encoding="utf-8")

    try:
        image = Image.open(first_page_image_path).convert("RGB")
        layout_list, _ = layout_parser.predict([image])
        detected = layout_list[0] if layout_list else []

        if not detected:
            _write_log("[layout_correction] Layout parser returned no detections for first page", log_file)
            return page_content

        visual_tags = _get_visual_tag_sequence(detected)
        mmd_tags_before = _extract_mmd_tags(page_content)

        _write_log(
            "[layout_correction] visual tags: %s | MMD tags (before): %s"
            % (visual_tags, mmd_tags_before),
            log_file,
        )

        corrected = page_content
        applied: List[Dict[str, str]] = []

        if "TITLE" in correctable_tags:
            if "TITLE" in visual_tags and "TITLE" not in mmd_tags_before:
                corrected, line_before, line_after = _retag_first_matching_text(corrected, "TITLE")
                if line_before:
                    applied.append({"tag": "TITLE", "line_before": line_before, "line_after": line_after})

        if "AUTHOR" in correctable_tags:
            if "AUTHOR" in visual_tags and "AUTHOR" not in _extract_mmd_tags(corrected):
                after = "TITLE" if "TITLE" in _extract_mmd_tags(corrected) else None
                corrected, line_before, line_after = _retag_first_matching_text(
                    corrected, "AUTHOR", after_tag=after
                )
                if line_before:
                    applied.append({"tag": "AUTHOR", "line_before": line_before, "line_after": line_after})

        if "ABSTRACT" in correctable_tags:
            if "ABSTRACT" in visual_tags and "ABSTRACT" not in _extract_mmd_tags(corrected):
                after = "AUTHOR" if "AUTHOR" in _extract_mmd_tags(corrected) else "TITLE"
                if after in _extract_mmd_tags(corrected):
                    corrected, line_before, line_after = _retag_first_matching_text(
                        corrected, "ABSTRACT", after_tag=after
                    )
                    if line_before:
                        applied.append({
                            "tag": "ABSTRACT",
                            "line_before": line_before,
                            "line_after": line_after,
                        })

        if applied:
            mmd_tags_after = _extract_mmd_tags(corrected)
            _log_correction_summary(
                mmd_file_path=mmd_file_path or first_page_image_path.replace(".png", ".mmd"),
                mmd_tags_before=mmd_tags_before,
                mmd_tags_after=mmd_tags_after,
                applied=applied,
                log_file=log_file,
            )
        return corrected
    finally:
        if log_file is not None:
            log_file.close()


def _log_correction_summary(
    mmd_file_path: str,
    mmd_tags_before: List[str],
    mmd_tags_after: List[str],
    applied: List[Dict[str, str]],
    max_snippet_len: int = 80,
    log_file=None,
) -> None:
    """Write layout correction summary to log_file only (no console)."""
    _write_log("[layout_correction] === applied to file: %s" % mmd_file_path, log_file)
    _write_log("[layout_correction]   MMD tags before: %s" % mmd_tags_before, log_file)
    _write_log("[layout_correction]   MMD tags after:  %s" % mmd_tags_after, log_file)
    for i, item in enumerate(applied, 1):
        tag = item["tag"]
        before = item["line_before"]
        after = item["line_after"]
        before_short = before if len(before) <= max_snippet_len else before[: max_snippet_len - 3] + "..."
        after_short = after if len(after) <= max_snippet_len else after[: max_snippet_len - 3] + "..."
        _write_log("[layout_correction]   change %d (%s): [TEXT] -> [%s]" % (i, tag, tag), log_file)
        _write_log("[layout_correction]     before: %s" % before_short, log_file)
        _write_log("[layout_correction]     after:  %s" % after_short, log_file)
    _write_log("[layout_correction] === end (%d change(s))" % len(applied), log_file)


def try_correct_first_page(
    page_content: str,
    first_page_image_path: str,
    correctable_tags: Set[str] = None,
    mmd_file_path: Optional[str] = None,
    log_file_path: Optional[str] = None,
) -> str:
    """
    Attempt layout correction; return original content if layout_parser unavailable.
    When log_file_path is set, all layout_correction output (including errors) is
    written to that file only, not to the console.
    """
    def write_error(msg: str) -> None:
        if log_file_path:
            Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write("[layout_correction] " + msg + "\n")
        else:
            logger.warning("Layout correction: %s", msg)

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "layout_parser"))
        from layout_parser_api import LayoutParserApi
        lp = LayoutParserApi()
        return correct_first_page(
            page_content,
            first_page_image_path,
            lp,
            correctable_tags=correctable_tags,
            mmd_file_path=mmd_file_path,
            log_file_path=log_file_path,
        )
    except ImportError:
        write_error("layout_parser not available, skipping first page correction")
        return page_content
    except Exception as e:
        write_error("failed: %s" % e)
        return page_content
