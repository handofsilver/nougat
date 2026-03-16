#!/usr/bin/env python3
"""
统计训练/验证/测试集中「一页」对应的 token 数（按当前 tokenizer 对 markdown 字段编码）。
用法:
  python scripts/stats_page_tokens.py --dataset /path/to/train.jsonl --tokenizer /path/to/nougat-base/tokenizer.json
  python scripts/stats_page_tokens.py --dataset /path/to/train.jsonl --tokenizer /path/to/nougat-base  # 自动找 tokenizer.json
"""
import argparse
import json
import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="统计 jsonl 中每页 markdown 的 token 数")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="jsonl 路径，如 train.jsonl / validation.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="tokenizer.json 路径或模型目录（内含 tokenizer.json）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="最多统计多少条，-1 表示全部",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default="50,90,95,99",
        help="输出的百分位数，逗号分隔，如 50,90,95,99",
    )
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer).resolve()
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: tokenizer not found: {tokenizer_path}", file=sys.stderr)
        sys.exit(1)

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.pad_token = "<pad>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f"Error: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # 若存在 seek.map 则按 seek 顺序读；否则按行顺序读
    seek_path = dataset_path.parent / (dataset_path.stem + ".seek.map")
    if seek_path.exists():
        positions = json.loads(seek_path.read_text(encoding="utf-8"))
    else:
        positions = None

    lengths = []
    n_skip = 0
    with dataset_path.open("r", encoding="utf-8") as f:
        if positions is not None:
            for pos in positions:
                f.seek(pos)
                line = f.readline()
                if args.max_samples >= 0 and len(lengths) >= args.max_samples:
                    break
                try:
                    data = json.loads(line)
                except Exception:
                    n_skip += 1
                    continue
                md = data.get("markdown") or data.get("path")
                if md is None:
                    # 有的 jsonl 只有 path 指向 .mmd 文件
                    path_mmd = data.get("path")
                    if path_mmd and isinstance(path_mmd, str):
                        root = dataset_path.parent
                        root_name = (data.get("root_name") or "").strip() or ""
                        full = root / root_name / path_mmd.replace(".png", ".mmd") if path_mmd.endswith(".png") else root / root_name / path_mmd
                        if full.exists():
                            md = full.read_text(encoding="utf-8")
                    if md is None:
                        n_skip += 1
                        continue
                if isinstance(md, bytes):
                    md = md.decode("utf-8")
                enc = tokenizer.encode(md, add_special_tokens=False)
                lengths.append(len(enc))
        else:
            for i, line in enumerate(f):
                if args.max_samples >= 0 and len(lengths) >= args.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    n_skip += 1
                    continue
                md = data.get("markdown")
                if md is None:
                    path_mmd = data.get("path")
                    if path_mmd and isinstance(path_mmd, str):
                        root = dataset_path.parent
                        root_name = (data.get("root_name") or "").strip() or ""
                        p = path_mmd.replace(".png", ".mmd") if path_mmd.endswith(".png") else path_mmd
                        full = root / root_name / p
                        if full.exists():
                            md = full.read_text(encoding="utf-8")
                    if md is None:
                        n_skip += 1
                        continue
                if isinstance(md, bytes):
                    md = md.decode("utf-8")
                enc = tokenizer.encode(md, add_special_tokens=False)
                lengths.append(len(enc))

    if not lengths:
        print("No valid pages with markdown/path found.", file=sys.stderr)
        sys.exit(1)

    import numpy as np

    arr = np.array(lengths, dtype=np.int64)
    n = len(arr)
    try:
        pcts = [int(x) for x in args.percentiles.split(",")]
    except Exception:
        pcts = [50, 90, 95, 99]

    print(f"Dataset: {dataset_path.name}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Samples (valid): {n}  (skipped: {n_skip})")
    print()
    print("Per-page token count (add_special_tokens=False):")
    print(f"  min       = {arr.min()}")
    print(f"  max       = {arr.max()}")
    print(f"  mean      = {arr.mean():.1f}")
    print(f"  median    = {np.median(arr):.1f}")
    print(f"  std       = {arr.std():.1f}")
    for p in pcts:
        print(f"  p{p:<2}       = {np.percentile(arr, p):.0f}")
    print()
    # 与常用上限对比
    for limit in [1500, 2048, 4096]:
        over = (arr > limit).sum()
        print(f"  Pages > {limit} tokens: {over} ({100.0 * over / n:.1f}%)")


if __name__ == "__main__":
    main()
