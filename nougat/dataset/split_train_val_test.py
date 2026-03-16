"""
按 paper_id 将 all_pages.jsonl 划分为 train / validation / test 三个 jsonl。
用于微调前数据准备，保证同一论文不会同时出现在训练集和测试集。

用法（在 base_dir 下执行）：
  python -m nougat.dataset.split_train_val_test --input all_pages.jsonl --out_dir .
  或指定比例：--train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(description="Split jsonl by paper_id into train/val/test.")
    parser.add_argument("--input", type=Path, default=Path("all_pages.jsonl"), help="Full index jsonl from create_index.")
    parser.add_argument("--out_dir", type=Path, default=Path("."), help="Directory to write train.jsonl, validation.jsonl, test.jsonl.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    lines_by_paper = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # image 如 "out/2303.00058/01.png" 或 "2303.00058/01.png"
            parts = rec["image"].replace("\\", "/").split("/")
            if len(parts) >= 2:
                paper_id = parts[-2]
            else:
                paper_id = rec["image"].split("/")[0] if "/" in rec["image"] else "unknown"
            lines_by_paper[paper_id].append(line)

    papers = list(lines_by_paper.keys())
    random.shuffle(papers)
    n = len(papers)
    t = int(args.train_ratio * n)
    v = int(args.val_ratio * n)
    s = n - t - v
    train_papers = set(papers[:t])
    val_papers = set(papers[t : t + v])
    test_papers = set(papers[t + v :])

    def write_split(name: str, paper_set):
        path = args.out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as out:
            for pid in sorted(paper_set):
                for line in lines_by_paper[pid]:
                    out.write(line + "\n")
        count = sum(len(lines_by_paper[pid]) for pid in paper_set)
        print(f"  {path}: {len(paper_set)} papers, {count} samples")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Splits (by paper_id):")
    write_split("train", train_papers)
    write_split("validation", val_papers)
    write_split("test", test_papers)
    print("Done. Run: python -m nougat.dataset.gen_seek train.jsonl validation.jsonl test.jsonl (from out_dir).")


if __name__ == "__main__":
    main()
