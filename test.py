"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nougat import NougatModel
from nougat.metrics import compute_metrics
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import NougatDataset
from nougat.utils.device import move_to_device
from lightning_module import NougatDataPLModule


def test(args):
    pretrained_model = NougatModel.from_pretrained(args.checkpoint)
    pretrained_model = move_to_device(pretrained_model)
    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    else:
        logging.warning("Results can not be saved. Please provide a -o/--save_path")

    predictions = []
    ground_truths = []
    metrics = defaultdict(list)

    dataset = NougatDataset(
        dataset_path=args.dataset,
        nougat_model=pretrained_model,
        max_length=pretrained_model.config.max_length,
        split=args.split,
        root_name=args.root_name,
    )

    # path_to_root = jsonl 所在目录（即 BASE_DIR）；root_name 默认为 ""，图片路径 = path_to_root / root_name / image
    sd = dataset.dataset
    path_to_root = sd.path_to_root
    root_name = sd.root_name or ""
    logging.info(
        "Dataset: path_to_root=%s, root_name=%r, len=%d",
        path_to_root, root_name, len(dataset),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.num_workers > 0),
        shuffle=args.shuffle,
        collate_fn=NougatDataPLModule.ignore_none_collate,
    )

    def run_one_batch(sample, batch_idx):
        if sample is None:
            return
        image_tensors, decoder_input_ids, _ = sample
        if image_tensors is None:
            return
        if args.num_samples > 0 and len(predictions) >= args.num_samples:
            return
        ground_truth = pretrained_model.decoder.tokenizer.batch_decode(
            decoder_input_ids, skip_special_tokens=True
        )
        outputs = pretrained_model.inference(
            image_tensors=image_tensors,
            return_attentions=False,
        )["predictions"]
        predictions.extend(outputs)
        ground_truths.extend(ground_truth)
        # 在主进程内逐条计算 metrics，避免 multiprocessing.Pool 与 CUDA/fork 冲突
        for pred, gt in zip(outputs, ground_truth):
            m = compute_metrics(pred, gt)
            for key, value in m.items():
                metrics[key].append(value)

    def save_results(path: str, preds: list, gts: list, m: dict):
        """写当前 predictions/ground_truths/metrics 到 JSON，便于查看中间结果或断点续跑."""
        if not path or not preds:
            return
        out = {}
        for metric, vals in m.items():
            out[f"{metric}_accuracies"] = vals
            out[f"{metric}_accuracy"] = np.mean(vals) if vals else np.nan
        out["predictions"] = preds
        out["ground_truths"] = gts
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    save_every = getattr(args, "save_every", 0)

    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            if idx == 0:
                logging.warning(
                    "Batch %d is None (all samples skipped). "
                    "Ensure images exist at path_to_root + root_name + image; "
                    "e.g. use --root_name \"\" when data is in base_dir/out/.",
                    idx,
                )
            continue
        run_one_batch(sample, idx)
        if save_every > 0 and args.save_path and (idx + 1) % save_every == 0:
            save_results(args.save_path, predictions, ground_truths, metrics)
            logging.info("Saved intermediate results: %d samples -> %s", len(predictions), args.save_path)
        if args.num_samples > 0 and len(predictions) >= args.num_samples:
            break

    scores = {}
    for metric, vals in metrics.items():
        scores[f"{metric}_accuracies"] = vals
        scores[f"{metric}_accuracy"] = np.mean(vals) if vals else np.nan

    if len(predictions) == 0:
        logging.warning(
            "No samples were evaluated. All batches were skipped. "
            "Ensure images exist at path_to_root + root_name + image (e.g. use --root_name \"\" when data is in base_dir/out/)."
        )
    else:
        try:
            print(
                "Total number of samples: %d, Edit Distance (ED) based accuracy score: %s, BLEU score: %s, METEOR score: %s"
                % (
                    len(predictions),
                    scores.get("edit_dist_accuracy", "N/A"),
                    scores.get("bleu_accuracy", "N/A"),
                    scores.get("meteor_accuracy", "N/A"),
                )
            )
        except Exception:
            pass

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=Path, default=None)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--save_path", "-o", type=str, default=None, help="json file to save results to"
    )
    parser.add_argument("--num_samples", "-N", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="原 4090 常用 4，高配可 16~24")
    parser.add_argument(
        "--root_name",
        type=str,
        default="",
        help='Subdir between path_to_root and image path. Use "" when data is in base_dir/out/.',
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers，0 最稳；高配可试 4")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="每处理 N 个 batch 将当前结果写入 save_path 一次（0=仅结束时写）；便于看中间结果与断点保留",
    )
    args, _ = parser.parse_known_args()
    args.checkpoint = get_checkpoint(args.checkpoint)

    test(args)
