#!/usr/bin/env python3
"""
离线评测脚本：基于 test_results_*.json 的 predictions/ground_truths，
在统一清洗后计算 NED、公式 Exact Match（基于公式精确匹配的召回/精确率/F1）和 BLEU，
支持单文件评估与 base/finetuned 双轨对比。

用法示例：

  # 单个结果文件
  python scripts/evaluate.py --results ocr_data/test_results_base.json

  # 基座 vs 微调 双轨对比
  python scripts/evaluate.py \
    --base ocr_data/test_results_base.json \
    --finetuned ocr_data/test_results_finetuned_v2.json

  # 输出汇总结果为 JSON
  python scripts/evaluate.py --results ocr_data/test_results_base.json --output ocr_data/eval_base_summary.json
"""

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_TOKEN = "[IMAGE_TOKEN]"


def _collapse_newlines(text: str) -> str:
    # 合并连续换行为单个换行
    return re.sub(r"\n+", "\n", text)


def _fullwidth_to_halfwidth(text: str) -> str:
    # 简单的全角转半角实现，覆盖常见 ASCII 范围
    res = []
    for ch in text:
        code = ord(ch)
        # 全角空格
        if code == 0x3000:
            res.append(" ")
        # 全角 ASCII 字符
        elif 0xFF01 <= code <= 0xFF5E:
            res.append(chr(code - 0xFEE0))
        else:
            res.append(ch)
    return "".join(res)


def _normalize_figures(text: str) -> str:
    """
    将所有 figure 相关内容统一成占位符，避免 ID/坐标差异影响编辑距离。
    参考:
      - nougat/dataset/patches/inject_coords_to_mmd.py
      - nougat/dataset/split_utils/markdown_parser.py
    """
    # [FIGURE:... ]...[END_FIGURE] 整块替换为占位符
    text = re.sub(r"\[FIGURE:.*?\].*?\[END_FIGURE\]", IMAGE_TOKEN, text, flags=re.S)
    # [FIGURE_COORDS](...)[END_FIGURE_COORDS]
    text = re.sub(
        r"\[FIGURE_COORDS\]\(.*?\)\[END_FIGURE_COORDS\]", IMAGE_TOKEN, text, flags=re.S
    )
    # 其它潜在的图片占位符，如 [IMAGE_xxx]
    text = re.sub(r"\[IMAGE[^\]]*\]", IMAGE_TOKEN, text)
    return text


def normalize_for_eval(text: str, enable_fullwidth: bool = True) -> str:
    """
    统一清洗逻辑：
      - strip
      - 合并连续换行
      - 全角 -> 半角（可选）
      - figure/图片占位符统一为 [IMAGE_TOKEN]
    """
    if text is None:
        return ""
    s = text.strip()
    s = _collapse_newlines(s)
    if enable_fullwidth:
        s = _fullwidth_to_halfwidth(s)
    s = _normalize_figures(s)
    return s


INLINE_FORMULA_PATTERNS = [
    r"\$(.+?)(?<!\\)\$",  # $...$
    r"\\\((.+?)(?<!\\)\\\)",  # \(...\)
]

DISPLAY_FORMULA_PATTERNS = [
    r"\$\$(.+?)(?<!\\)\$\$",  # $$...$$
    r"\\\[(.+?)(?<!\\)\\\]",  # \[...\]
]


def extract_formulas(text: str) -> List[str]:
    """
    从字符串中提取所有行内/独立公式，返回公式字符串列表。
    """
    formulas: List[str] = []
    for pat in INLINE_FORMULA_PATTERNS + DISPLAY_FORMULA_PATTERNS:
        for m in re.findall(pat, text, flags=re.S):
            # 去除首尾空白，保持内部原样
            formulas.append(m.strip())
    return formulas


@dataclass
class MetricsResult:
    n_samples: int
    ned: Optional[float]
    bleu: Optional[float]
    formula_precision: Optional[float]
    formula_recall: Optional[float]
    formula_f1: Optional[float]


def _safe_mean(values: List[float]) -> Optional[float]:
    arr = [v for v in values if v is not None and not math.isnan(v)]
    if not arr:
        return None
    return float(np.mean(arr))


def compute_sample_metrics(pred: str, gt: str, minlen: int = 4) -> Dict[str, float]:
    """
    在已清洗后的文本上计算 NED 和 BLEU 所需的中间值。
    返回:
      - ned: 1 - edit_distance / max_len
      - bleu: sentence_bleu 值
      若长度太短则返回空 dict。
    """
    if len(pred) < minlen or len(gt) < minlen:
        return {}
    max_len = max(len(pred), len(gt))
    if max_len == 0:
        return {}
    # 使用 rapidfuzz 的 Levenshtein（C++ 实现），比 nltk.edit_distance 快一个数量级以上
    ned = 1.0 - Levenshtein.distance(pred, gt) / max_len

    ref_tokens = gt.split()
    hyp_tokens = pred.split()
    bleu = 0.0
    if ref_tokens and hyp_tokens:
        bleu = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=SmoothingFunction().method1,
        )
    return {"ned": ned, "bleu": bleu}


def compute_formula_metrics(
    preds: List[str], gts: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    基于清洗后的文本，计算公式层面的全局 P/R/F1。
    统计方式：对每条样本的公式多重集做 matching，总结到全局计数。
    """
    total_gt = 0
    total_pred = 0
    total_match = 0

    for pred, gt in zip(preds, gts):
        gt_forms = extract_formulas(gt)
        pred_forms = extract_formulas(pred)
        if not gt_forms and not pred_forms:
            continue

        total_gt += len(gt_forms)
        total_pred += len(pred_forms)

        # 多重集匹配：按字符串 exact match 统计匹配次数
        gt_counts: Dict[str, int] = {}
        for f in gt_forms:
            gt_counts[f] = gt_counts.get(f, 0) + 1
        pred_counts: Dict[str, int] = {}
        for f in pred_forms:
            pred_counts[f] = pred_counts.get(f, 0) + 1

        for f, c_gt in gt_counts.items():
            c_pred = pred_counts.get(f, 0)
            total_match += min(c_gt, c_pred)

    if total_gt == 0 and total_pred == 0:
        return None, None, None

    precision = total_match / total_pred if total_pred > 0 else 0.0
    recall = total_match / total_gt if total_gt > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_single_result(
    predictions: List[str],
    ground_truths: List[str],
    enable_fullwidth: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Eval",
) -> MetricsResult:
    """
    对单个 results JSON 的 predictions/ground_truths 做评估。
    """
    assert len(predictions) == len(
        ground_truths
    ), "predictions 与 ground_truths 长度不一致"

    norm_preds: List[str] = []
    norm_gts: List[str] = []
    ned_list: List[float] = []
    bleu_list: List[float] = []

    it = zip(predictions, ground_truths)
    if show_progress:
        it = tqdm(it, total=len(predictions), desc=progress_desc, unit="sample")

    for pred, gt in it:
        p = normalize_for_eval(pred, enable_fullwidth=enable_fullwidth)
        g = normalize_for_eval(gt, enable_fullwidth=enable_fullwidth)
        norm_preds.append(p)
        norm_gts.append(g)
        m = compute_sample_metrics(p, g)
        if m:
            ned_list.append(m["ned"])
            bleu_list.append(m["bleu"])

    formula_p, formula_r, formula_f1 = compute_formula_metrics(norm_preds, norm_gts)

    return MetricsResult(
        n_samples=len(predictions),
        ned=_safe_mean(ned_list),
        bleu=_safe_mean(bleu_list),
        formula_precision=formula_p,
        formula_recall=formula_r,
        formula_f1=formula_f1,
    )


def _load_results(path: Path) -> Tuple[List[str], List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions")
    gts = data.get("ground_truths")
    if preds is None or gts is None:
        raise ValueError(f"{path} 缺少 predictions/ground_truths 字段")
    if len(preds) != len(gts):
        # 仍允许不等长，但只对齐最短部分
        n = min(len(preds), len(gts))
        preds = preds[:n]
        gts = gts[:n]
    return preds, gts


def _print_metrics(name: str, m: MetricsResult) -> None:
    def fmt(x: Optional[float]) -> str:
        return f"{x:.4f}" if x is not None else "n/a"

    print(f"=== {name} ===")
    print(f"Samples          : {m.n_samples}")
    print(f"NED (mean)      : {fmt(m.ned)}")
    print(f"BLEU (mean)     : {fmt(m.bleu)}")
    print(f"Formula P/R/F1  : P={fmt(m.formula_precision)}, R={fmt(m.formula_recall)}, F1={fmt(m.formula_f1)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Nougat predictions against ground truths with NED, formula EM and BLEU."
    )
    parser.add_argument(
        "--results",
        type=str,
        help="单个结果 JSON 路径（含 predictions/ground_truths）",
    )
    parser.add_argument(
        "--base",
        type=str,
        help="基座模型结果 JSON 路径（含 predictions/ground_truths）",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        help="微调模型结果 JSON 路径（含 predictions/ground_truths）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="可选：将汇总结果写入 JSON 文件路径",
    )
    parser.add_argument(
        "--no-clean-fullwidth",
        action="store_true",
        help="关闭全角->半角转换（其余清洗仍启用）",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="不显示进度条（适用于管道或日志重定向）",
    )
    args = parser.parse_args()

    enable_fullwidth = not args.no_clean_fullwidth
    show_progress = not args.no_progress

    if args.results and (args.base or args.finetuned):
        parser.error("不能同时使用 --results 与 (--base/--finetuned)，请二选一。")
    if not args.results and not (args.base and args.finetuned):
        parser.error("需指定 --results 或同时指定 --base 与 --finetuned。")

    summary: Dict[str, Dict[str, Optional[float]]] = {}

    if args.results:
        path = Path(args.results).resolve()
        preds, gts = _load_results(path)
        metrics = evaluate_single_result(
            preds, gts, enable_fullwidth=enable_fullwidth, show_progress=show_progress
        )
        _print_metrics(path.name, metrics)
        summary[path.name] = {
            "n_samples": metrics.n_samples,
            "ned": metrics.ned,
            "bleu": metrics.bleu,
            "formula_precision": metrics.formula_precision,
            "formula_recall": metrics.formula_recall,
            "formula_f1": metrics.formula_f1,
        }
    else:
        base_path = Path(args.base).resolve()
        ft_path = Path(args.finetuned).resolve()
        base_preds, base_gts = _load_results(base_path)
        ft_preds, ft_gts = _load_results(ft_path)

        # 对齐：以两个结果中较短的长度为准，并假定 gt 序列相同
        n = min(len(base_preds), len(ft_preds), len(base_gts), len(ft_gts))
        base_preds = base_preds[:n]
        base_gts = base_gts[:n]
        ft_preds = ft_preds[:n]
        ft_gts = ft_gts[:n]

        base_metrics = evaluate_single_result(
            base_preds,
            base_gts,
            enable_fullwidth=enable_fullwidth,
            show_progress=show_progress,
            progress_desc="BASE",
        )
        ft_metrics = evaluate_single_result(
            ft_preds,
            ft_gts,
            enable_fullwidth=enable_fullwidth,
            show_progress=show_progress,
            progress_desc="FINETUNED",
        )

        _print_metrics("BASE", base_metrics)
        _print_metrics("FINETUNED", ft_metrics)

        def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            return b - a

        summary["base"] = {
            "n_samples": base_metrics.n_samples,
            "ned": base_metrics.ned,
            "bleu": base_metrics.bleu,
            "formula_precision": base_metrics.formula_precision,
            "formula_recall": base_metrics.formula_recall,
            "formula_f1": base_metrics.formula_f1,
        }
        summary["finetuned"] = {
            "n_samples": ft_metrics.n_samples,
            "ned": ft_metrics.ned,
            "bleu": ft_metrics.bleu,
            "formula_precision": ft_metrics.formula_precision,
            "formula_recall": ft_metrics.formula_recall,
            "formula_f1": ft_metrics.formula_f1,
        }
        summary["delta_finetuned_minus_base"] = {
            "ned": delta(base_metrics.ned, ft_metrics.ned),
            "bleu": delta(base_metrics.bleu, ft_metrics.bleu),
            "formula_precision": delta(
                base_metrics.formula_precision, ft_metrics.formula_precision
            ),
            "formula_recall": delta(
                base_metrics.formula_recall, ft_metrics.formula_recall
            ),
            "formula_f1": delta(base_metrics.formula_f1, ft_metrics.formula_f1),
        }

    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

