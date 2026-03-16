<div align="center">
<h1>Nougat: Neural Optical Understanding for Academic Documents</h1>

[English](README.md) | 中文

[![Paper](https://img.shields.io/badge/Paper-arxiv.2308.13418-white)](https://arxiv.org/abs/2308.13418)
[![GitHub](https://img.shields.io/github/license/facebookresearch/nougat)](https://github.com/facebookresearch/nougat)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

本仓库 fork 自 [Meta Nougat](https://github.com/facebookresearch/nougat)，在原版基础上做了**数据工程方向**的扩展——构建一条完整的 **高质量训练数据集生产流水线**，从 arXiv `.tex` 源码出发，产出按页对齐的 `.mmd` / `.png` 训练对。

完整技术设计文档：**[docs/data_engineering_design.md](docs/data_engineering_design.md)**

---

## 与原版 Nougat 的差异

| 模块 | 原版 Nougat | 本 Fork 贡献 |
|------|------------|-------------|
| `parser/latexml_parser.py` | HTML 解析骨架 | 深度修改：算法环境检测、caption 标签注入、引用处理 |
| `parser/markdown.py` | 基础 Markdown 格式化 | **重写**：全套语义标签生成 |
| `split_utils/` (11 个文件) | **不存在** | **全新**——分页对齐算法 |
| `split_md_to_pages.py` | 极简分页 | **完全重写** |
| `split_htmls_to_pages.py` | 原版入口 | **重写**——新 API、坐标注入、layout_parser 集成 |
| `patches/` (5 个文件) | **不存在** | **全新**——TeX 预处理、参考文献/引用修复、坐标注入 |
| `preprocess_pipeline.py` | **不存在** | **全新**——从 .zip 到 HTML 的统一预处理脚本 |
| `layout_correction.py` | **不存在** | **全新**——基于 DiT 版面检测的首页标签修正 |
| `layout_parser/` | **不存在** | **全新**——DiT 版面检测模型（可选） |

---

## 快速开始

### 一键环境配置

```bash
bash setup_env.sh          # 默认创建 conda 环境名 nougat
bash setup_env.sh myenv    # 或指定环境名
```

脚本会依次完成：创建 conda 环境 → 安装 PyTorch → 安装 nougat + dataset 依赖 → 检查 LaTeXML / pdffigures2，每一步都有状态输出。

### 手动安装

```bash
conda create -n nougat python=3.10 -y && conda activate nougat
pip install torch torchvision          # 根据 CUDA 版本选择
pip install -e ".[dataset]"
pip install jieba
```

外部工具（预处理阶段需要）：

- **[LaTeXML](https://math.nist.gov/~BMiller/LaTeXML/)**：`.tex → .html`，Ubuntu 下 `sudo apt-get install latexml`
- **[pdffigures2](https://github.com/allenai/pdffigures2)**：PDF 图表抽取，需 Java + sbt 构建，然后：
  ```bash
  export PDFFIGURES_PATH="/path/to/pdffigures2-assembly.jar"
  ```

---

## 数据处理流水线

整条流水线分为 **两个阶段**，对应两个独立入口命令。以 arXiv 论文 `2308.13418` 为例：

### 阶段 0：预处理（.zip/.pdf → HTML）

```bash
python -m nougat.dataset.preprocess_pipeline --base-dir /path/to/data_root
```

#### 输入要求

在 `data_root/src/` 下放置每篇论文的 **PDF** 和 **源码包**：

```
data_root/
└── src/
    ├── 2308.13418.pdf          ← arXiv PDF
    ├── 2308.13418.zip          ← arXiv 源码包 (.zip / .tar.gz)
    ├── 2309.xxxxx.pdf
    ├── 2309.xxxxx.zip
    └── ...
```

#### 处理步骤（脚本自动完成）

| 步骤 | 操作 | 涉及代码 |
|------|------|---------|
| 1 | 解压 `.zip` → `src/2308.13418/` | `preprocess_pipeline.py` |
| 2 | 识别主 `.tex` 文档，复制/重命名为 `main.tex` | `preprocess_pipeline.py` → `find_main_tex()` |
| 3 | 清理 `.tex`（删注释、无用命令） | `patches/preprocess_tex.py` |
| 4 | 运行 LaTeXML：`main.tex → html/2308.13418/2308.13418.html` | LaTeXML (外部) |
| 5 | 从 `.bbl` 修复 HTML 中的参考文献与引用 | `patches/fix_bibliography.py`, `patches/fix_citations.py` |
| 6 | 运行 pdffigures2：`src/2308.13418.pdf → fig/2308.13418.json` | `pdffigures.py` → pdffigures2 (外部) |

> **关于 PDF 路径**：`split_htmls_to_pages` 查找 PDF 的路径是 `src/2308.13418.pdf`（扁平文件），**不是**解压目录内的副本。解压目录内的 `.tex` / `.bbl` 仅供预处理阶段使用。

#### 产出

```
data_root/
├── src/
│   ├── 2308.13418.pdf
│   ├── 2308.13418.zip
│   └── 2308.13418/             ← 解压后的 TeX 源码
│       ├── main.tex            ← 已识别并重命名
│       ├── main.bbl
│       └── ...
├── html/
│   └── 2308.13418/
│       ├── 2308.13418.html     ← LaTeXML 产出
│       ├── 2308.13418.xml
│       └── ...
└── fig/
    └── 2308.13418.json         ← pdffigures2 图表信息
```

### 阶段 1：分页对齐（HTML → 每页 .mmd / .png）

```bash
python -m nougat.dataset.split_htmls_to_pages \
    --html  data_root/html \
    --pdfs  data_root/src \
    --out   data_root/out \
    --figure data_root/fig \
    --markdown data_root/markdown
```

#### 处理步骤

| 步骤 | 操作 | 涉及代码 |
|------|------|---------|
| 1 | HTML → 语义元素树 | `parser/latexml_parser.py` |
| 2 | 元素树 → 带语义标签的 MMD | `parser/markdown.py` |
| 3 | 注入图坐标到 MMD | `patches/inject_coords_to_mmd.py` |
| 4 | 分页对齐（核心 5 步算法） | `split_md_to_pages.py` + `split_utils/` |
| 5 | 渲染 PDF 各页为 PNG | `rasterize.py` |
| 6 | （可选）首页版面检测修正 | `layout_correction.py` + `layout_parser/` |

启用首页修正：加 `--layout-parser` 参数（需先安装 detectron2 及 DiT 权重，详见 [layout_parser/README.md](layout_parser/README.md)）。

#### 常用参数

| 参数 | 说明 |
|------|------|
| `--recompute` | 重新计算所有分页 |
| `--markdown DIR` | 保存中间完整 MMD（分页前与分页后） |
| `--workers N` | 并行进程数 |
| `--dpi N` | 页面渲染分辨率（默认 96） |
| `--timeout SEC` | 单篇论文最大处理时间 |
| `--layout-parser` | 启用 DiT 首页修正（可选） |

#### 产出

```
data_root/
├── out/
│   └── 2308.13418/
│       ├── 01.mmd              ← 第 1 页带语义标签的 Markdown
│       ├── 01.png              ← 第 1 页渲染图像
│       ├── 02.mmd
│       ├── 02.png
│       └── ...
└── markdown/                   ← --markdown 时产出
    ├── 2308.13418.mmd          ← 分页前的完整 MMD
    └── 2308.13418_processed.mmd ← 分页后拼接
```

### 最终数据目录一览

跑完两个阶段后，`data_root/` 应具有以下结构（所有目录均必要）：

```
data_root/
├── src/                 ← 源文件：.pdf（扁平）+ 解压后的 TeX 目录
│   ├── 2308.13418.pdf
│   └── 2308.13418/
├── html/                ← LaTeXML 产出的 HTML
│   └── 2308.13418/
├── fig/                 ← pdffigures2 产出的图表 JSON
│   └── 2308.13418.json
├── out/                 ← 最终产出：每页 .mmd + .png
│   └── 2308.13418/
└── markdown/            ← （可选）中间 MMD 文件
```

---

## 索引、划分、训练与评测（本 fork）

流水线产出 `data_root/out/` 后，需先建索引、**按论文划分** train/val/test，再训练与评测。

```bash
# 1. 建全量页级索引
python -m nougat.dataset.create_index --dir data_root/out --root data_root --out data_root/all_pages.jsonl

# 2. 按 paper_id 划分 train/val/test
python -m nougat.dataset.split_train_val_test --input data_root/all_pages.jsonl --out_dir data_root --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

# 3. 生成 seek map（训练用）
python -m nougat.dataset.gen_seek data_root/train.jsonl data_root/validation.jsonl data_root/test.jsonl

# 4. 下载基座模型（须用 GitHub 0.1.0-base，与 HuggingFace 格式不兼容）并导出训练用权重
python -m nougat.utils.checkpoint base /path/to/nougat-base
# 再导出 aligned_model_weights.pth（见 docs/pipeline_and_training_guide.md）

# 5. 微调（配置：config/train_nougat_my.yaml；val_with_generation=false 可大幅加快验证）
python train.py --config config/train_nougat_my.yaml

# 6. 在测试集上推理
python test.py --checkpoint /path/to/nougat-base --dataset data_root/test.jsonl --split test --save_path data_root/test_results_base.json --root_name ""

# 7. 离线评测（NED、公式精确匹配、BLEU）
python scripts/evaluate.py --base data_root/test_results_base.json --finetuned data_root/test_results_finetuned.json
```

**本 fork 训练相关修改**：验证可仅算 loss（配置中 `val_with_generation: false`），避免每轮对全量验证集做生成；checkpoint 按 `val/loss` 保留 **top-k**；推理使用 `repetition_penalty` 与可配置 `max_new_tokens`。详见 **docs/validation_and_checkpoint_notes.md**。

**脚本**：`scripts/stats_page_tokens.py` — 统计每页 token 数，用于设定 `max_new_tokens`；`scripts/evaluate.py` — 对 `test_results_*.json` 做 NED、公式匹配、BLEU。  

完整步骤（路径、环境、命令）：**docs/pipeline_and_training_guide.md**。

---

## 文档与脚本索引

| 文档 / 脚本 | 用途 |
|-------------|------|
| **docs/data_engineering_design.md** | 数据流水线完整设计（TeX→HTML→MMD→分页对齐、语义标签、layout_parser）。 |
| **docs/pipeline_and_training_guide.md** | 端到端：模型下载、索引/划分/seek、推理与微调命令（环境见 README setup_env）。 |
| **docs/inference_io_and_json_format.md** | test.jsonl、test_results_*.json 来源、生成命令、JSON 结构。 |
| **docs/evaluation_results_and_summary.md** | 数据与 baseline/微调输出差异、评测方案、结果与简历话术。 |
| **docs/validation_and_checkpoint_notes.md** | 验证极慢原因；val_with_generation、save_top_k、repetition_penalty；早停与学习率。 |
| **scripts/evaluate.py** | 离线评测：文本规范化、NED、公式精确匹配、BLEU；单文件或 base vs 微调对比。 |
| **scripts/stats_page_tokens.py** | 按页统计 token 数（train.jsonl），用于设定 max_new_tokens。 |

---

## 使用预训练模型做 PDF → Markdown 预测

原版 Nougat 的预测功能保留不变：

```bash
pip install nougat-ocr
nougat path/to/file.pdf -o output_directory
```

API 方式：`nougat_api` 启动服务后 POST 到 `http://127.0.0.1:8503/predict/`。

---

## 项目结构

```
nougat/
├── nougat/
│   └── dataset/
│       ├── preprocess_pipeline.py   ← 阶段 0 入口：zip → HTML
│       ├── split_htmls_to_pages.py  ← 阶段 1 入口：HTML → 每页 mmd/png
│       ├── split_md_to_pages.py     ← 分页算法编排
│       ├── split_utils/             ← 分页算法各子模块
│       ├── parser/                  ← HTML → 带标签 MMD
│       ├── patches/                 ← TeX 预处理、引用修复、坐标注入
│       ├── layout_correction.py     ← 首页修正（调用 layout_parser）
│       ├── rasterize.py             ← PDF → PNG
│       ├── pdffigures.py            ← pdffigures2 封装
│       ├── create_index.py          ← 生成训练索引 JSONL
│       ├── split_train_val_test.py  ← 按 paper_id 划分 train/val/test
│       └── gen_seek.py              ← 生成 seek map
├── scripts/                         ← evaluate.py、stats_page_tokens.py
├── layout_parser/                   ← DiT 版面检测模型（可选）
│   ├── README.md
│   └── ...
├── docs/
│   ├── data_engineering_design.md   ← 完整技术设计文档
│   ├── pipeline_and_training_guide.md
│   ├── inference_io_and_json_format.md
│   ├── evaluation_results_and_summary.md
│   └── validation_and_checkpoint_notes.md
├── setup_env.sh                     ← 一键环境配置
├── setup.py
└── README.md                        ← 本文件
```

**所有处理步骤都归属到流水线的具体阶段，没有离散补丁**：

- `patches/preprocess_tex.py` → 阶段 0 步骤 3
- `patches/fix_bibliography.py` + `latex_to_html.py` → 阶段 0 步骤 5
- `patches/fix_citations.py` → 阶段 0 步骤 5
- `patches/inject_coords_to_mmd.py` → 阶段 1 步骤 3
- `layout_correction.py` + `layout_parser/` → 阶段 1 步骤 6（可选）

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
