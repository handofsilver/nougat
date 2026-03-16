<div align="center">
<h1>Nougat: Neural Optical Understanding for Academic Documents</h1>

English | [中文](README_cn.md)

[![Paper](https://img.shields.io/badge/Paper-arxiv.2308.13418-white)](https://arxiv.org/abs/2308.13418)
[![GitHub](https://img.shields.io/github/license/facebookresearch/nougat)](https://github.com/facebookresearch/nougat)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

This repository is a fork of [Meta Nougat](https://github.com/facebookresearch/nougat) with **data-engineering** extensions: a full pipeline for building high-quality training data—from arXiv `.tex` sources to page-aligned `.mmd` / `.png` pairs.

Full technical design: **[docs/data_engineering_design.md](docs/data_engineering_design.md)**

---

## Differences from upstream Nougat

| Module | Upstream Nougat | This fork |
|--------|-----------------|-----------|
| `parser/latexml_parser.py` | HTML parsing skeleton | Deep changes: algorithm env detection, caption injection, citation handling |
| `parser/markdown.py` | Basic Markdown formatting | **Rewritten**: full semantic tag set |
| `split_utils/` (11 files) | **Absent** | **New**: page-alignment algorithm |
| `split_md_to_pages.py` | Minimal splitting | **Completely rewritten** |
| `split_htmls_to_pages.py` | Original entrypoint | **Rewritten**: new API, coordinate injection, layout_parser integration |
| `patches/` (5 files) | **Absent** | **New**: TeX preprocessing, bibliography/citation fixes, coordinate injection |
| `preprocess_pipeline.py` | **Absent** | **New**: unified preprocessing from .zip to HTML |
| `layout_correction.py` | **Absent** | **New**: first-page tag correction via DiT layout detection |
| `layout_parser/` | **Absent** | **New**: DiT layout detection model (optional) |

---

## Quick start

### One-command environment setup

```bash
bash setup_env.sh          # creates conda env named nougat by default
bash setup_env.sh myenv    # or specify env name
```

The script runs: create conda env → install PyTorch → install nougat + dataset deps → check LaTeXML / pdffigures2, with status output at each step.

### Manual install

```bash
conda create -n nougat python=3.10 -y && conda activate nougat
pip install torch torchvision          # choose CUDA build if needed
pip install -e ".[dataset]"
pip install jieba
```

External tools (needed for preprocessing):

- **[LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/)**: `.tex → .html`; on Ubuntu: `sudo apt-get install latexml`
- **[pdffigures2](https://github.com/allenai/pdffigures2)**: PDF figure extraction; requires Java + sbt, then:
  ```bash
  export PDFFIGURES_PATH="/path/to/pdffigures2-assembly.jar"
  ```

---

## Data pipeline

The pipeline has **two stages**, each with its own command. Example paper ID: `2308.13418`.

### Stage 0: Preprocessing (.zip/.pdf → HTML)

```bash
python -m nougat.dataset.preprocess_pipeline --base-dir /path/to/data_root
```

#### Input

Place each paper’s **PDF** and **source archive** under `data_root/src/`:

```
data_root/
└── src/
    ├── 2308.13418.pdf          ← arXiv PDF
    ├── 2308.13418.zip          ← arXiv source (.zip / .tar.gz)
    ├── 2309.xxxxx.pdf
    ├── 2309.xxxxx.zip
    └── ...
```

#### Steps (automated)

| Step | Action | Code |
|------|--------|------|
| 1 | Extract `.zip` → `src/2308.13418/` | `preprocess_pipeline.py` |
| 2 | Find main `.tex`, copy/rename to `main.tex` | `preprocess_pipeline.py` → `find_main_tex()` |
| 3 | Clean `.tex` (comments, useless commands) | `patches/preprocess_tex.py` |
| 4 | Run LaTeXML: `main.tex → html/2308.13418/2308.13418.html` | LaTeXML (external) |
| 5 | Fix bibliography & citations in HTML from `.bbl` | `patches/fix_bibliography.py`, `patches/fix_citations.py` |
| 6 | Run pdffigures2: `src/2308.13418.pdf → fig/2308.13418.json` | `pdffigures.py` → pdffigures2 (external) |

> **PDF path**: `split_htmls_to_pages` looks for PDF at `src/2308.13418.pdf` (flat path), **not** inside the extracted directory. The extracted `.tex`/`.bbl` are only used in preprocessing.

#### Output

```
data_root/
├── src/
│   ├── 2308.13418.pdf
│   ├── 2308.13418.zip
│   └── 2308.13418/             ← extracted TeX
│       ├── main.tex
│       ├── main.bbl
│       └── ...
├── html/
│   └── 2308.13418/
│       ├── 2308.13418.html
│       ├── 2308.13418.xml
│       └── ...
└── fig/
    └── 2308.13418.json
```

### Stage 1: Page alignment (HTML → per-page .mmd / .png)

```bash
python -m nougat.dataset.split_htmls_to_pages \
    --html  data_root/html \
    --pdfs  data_root/src \
    --out   data_root/out \
    --figure data_root/fig \
    --markdown data_root/markdown
```

#### Steps

| Step | Action | Code |
|------|--------|------|
| 1 | HTML → semantic element tree | `parser/latexml_parser.py` |
| 2 | Element tree → tagged MMD | `parser/markdown.py` |
| 3 | Inject figure coordinates into MMD | `patches/inject_coords_to_mmd.py` |
| 4 | Page alignment (core 5-step algorithm) | `split_md_to_pages.py` + `split_utils/` |
| 5 | Render PDF pages to PNG | `rasterize.py` |
| 6 | (Optional) First-page layout correction | `layout_correction.py` + `layout_parser/` |

To enable first-page correction: add `--layout-parser` (requires detectron2 and DiT weights; see [layout_parser/README.md](layout_parser/README.md)).

#### Common options

| Option | Description |
|--------|-------------|
| `--recompute` | Recompute all splits |
| `--markdown DIR` | Save intermediate full MMD (before/after split) |
| `--workers N` | Number of worker processes |
| `--dpi N` | Page render resolution (default 96) |
| `--timeout SEC` | Max time per paper (seconds) |
| `--layout-parser` | Enable DiT first-page correction (optional) |

#### Output

```
data_root/
├── out/
│   └── 2308.13418/
│       ├── 01.mmd
│       ├── 01.png
│       ├── 02.mmd
│       ├── 02.png
│       └── ...
└── markdown/                   ← when --markdown is set
    ├── 2308.13418.mmd
    └── 2308.13418_processed.mmd
```

### Final directory layout

After both stages, `data_root/` should look like this:

```
data_root/
├── src/                 ← Sources: flat .pdf + extracted TeX dirs
├── html/                ← LaTeXML HTML
├── fig/                 ← pdffigures2 JSON
├── out/                 ← Final per-page .mmd + .png
└── markdown/            ← (optional) intermediate MMD
```

---

## Index, split, training & evaluation (this fork)

After Stage 0–1 you have `data_root/out/` with per-page `.mmd` and `.png`. Build the index, **split by paper** into train/val/test, then train and evaluate.

```bash
# 1. Index all pages
python -m nougat.dataset.create_index --dir data_root/out --root data_root --out data_root/all_pages.jsonl

# 2. Split by paper_id (train/val/test)
python -m nougat.dataset.split_train_val_test --input data_root/all_pages.jsonl --out_dir data_root --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 3. Seek maps for training
python -m nougat.dataset.gen_seek data_root/train.jsonl data_root/validation.jsonl data_root/test.jsonl

# 4. Download base model (use GitHub 0.1.0-base, not HuggingFace) and export for training
python -m nougat.utils.checkpoint base /path/to/nougat-base
# then export aligned_model_weights.pth (see docs/从base_dir到推理与微调全流程操作指南.md)

# 5. Train (config: config/train_nougat_my.yaml; val_with_generation=false for fast validation)
python train.py --config config/train_nougat_my.yaml

# 6. Inference on test set
python test.py --checkpoint /path/to/nougat-base --dataset data_root/test.jsonl --split test --save_path data_root/test_results_base.json --root_name ""

# 7. Offline evaluation (NED, formula exact match, BLEU)
python scripts/evaluate.py --base data_root/test_results_base.json --finetuned data_root/test_results_finetuned.json
```

**Training changes in this fork:** validation can be **loss-only** (`val_with_generation: false` in config) to avoid slow per-epoch generation; checkpoint keeps **top-k** by `val/loss`; inference uses `repetition_penalty` and configurable `max_new_tokens`. See `docs/微调验证与checkpoint优化复盘.md` for details.

**Scripts:** `scripts/stats_page_tokens.py` — token stats per page for choosing `max_new_tokens`; `scripts/evaluate.py` — NED, formula match, BLEU on `test_results_*.json`.

Full step-by-step (paths, env, commands): **docs/从base_dir到推理与微调全流程操作指南.md**.

---

## Docs and scripts index

| Doc / script | Purpose |
|--------------|--------|
| **docs/data_engineering_design.md** | Full data-pipeline design (TeX→HTML→MMD→page alignment, tags, layout_parser). |
| **docs/从base_dir到推理与微调全流程操作指南.md** | End-to-end: model download, index/split/seek, inference, fine-tuning commands. |
| **docs/当前配置与运行要点总览.md** | Env, paths, train config, token stats, inference settings. |
| **docs/Nougat 模型微调后评测与量化指标生成指南.md** | Evaluation blueprint: dual-run inference, cleaning, NED/formula/BLEU, resume wording. |
| **docs/推理输入输出与JSON格式说明.md** | test.jsonl, test_results_*.json sources, commands, JSON layout. |
| **docs/推理与微调结果评估总结.md** | Data vs baseline/finetuned output, evaluation plan and results. |
| **docs/推理结果评估方向与格式核对.md** | Format differences, unified cleaning, metric definitions. |
| **docs/微调验证与checkpoint优化复盘.md** | Why validation was slow; val_with_generation, save_top_k, repetition_penalty. |
| **docs/第一次推理与训练完成日志.md** | First-run log and notes. |
| **scripts/evaluate.py** | Offline eval: normalize text, NED, formula exact match, BLEU; single file or base vs finetuned. |
| **scripts/stats_page_tokens.py** | Token counts per page (train.jsonl) for max_new_tokens tuning. |

---

## Using the pretrained model (PDF → Markdown)

Upstream prediction is unchanged:

```bash
pip install nougat-ocr
nougat path/to/file.pdf -o output_directory
```

API: run `nougat_api`, then POST to `http://127.0.0.1:8503/predict/`.

---

## Project layout

```
nougat/
├── nougat/
│   └── dataset/
│       ├── preprocess_pipeline.py   ← Stage 0 entry: zip → HTML
│       ├── split_htmls_to_pages.py  ← Stage 1 entry: HTML → per-page mmd/png
│       ├── split_md_to_pages.py     ← Page-splitting orchestration
│       ├── split_utils/             ← Splitting submodules
│       ├── parser/                  ← HTML → tagged MMD
│       ├── patches/                 ← TeX preprocessing, citation fixes, coordinate injection
│       ├── layout_correction.py     ← First-page correction (calls layout_parser)
│       ├── rasterize.py             ← PDF → PNG
│       ├── pdffigures.py            ← pdffigures2 wrapper
│       ├── create_index.py          ← Training index JSONL
│       ├── split_train_val_test.py  ← Train/val/test split by paper_id
│       └── gen_seek.py              ← Seek map
├── scripts/                         ← evaluate.py, stats_page_tokens.py
├── layout_parser/                   ← DiT layout model (optional)
├── docs/
│   └── data_engineering_design.md
├── setup_env.sh                     ← One-command env setup
├── setup.py
└── README.md
```

All processing steps are part of the pipeline (no orphan scripts):

- `patches/preprocess_tex.py` → Stage 0 step 3
- `patches/fix_bibliography.py` + `latex_to_html.py` → Stage 0 step 5
- `patches/fix_citations.py` → Stage 0 step 5
- `patches/inject_coords_to_mmd.py` → Stage 1 step 3
- `layout_correction.py` + `layout_parser/` → Stage 1 step 6 (optional)

---

## Citation

```
@misc{blecher2023nougat,
      title={Nougat: Neural Optical Understanding for Academic Documents},
      author={Lukas Blecher and Guillem Cucurull and Thomas Scialom and Robert Stojnic},
      year={2023},
      eprint={2308.13418},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License

Nougat codebase is licensed under MIT. Nougat model weights are licensed under CC-BY-NC.
