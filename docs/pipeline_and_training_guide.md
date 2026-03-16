# 从 base_dir 到推理与微调全流程操作指南

本文档面向「已有与 `__test_0/out` 同构的数据目录（.mmd + .png + meta.json，约 33474 条），且该目录的父目录作为 BASE_DIR」的场景，系统梳理：**模型下载 → 数据划分与训练格式 → 推理流程 → 微调流程** 及具体命令。**按章节顺序执行即可**；每一步的输入是上一步的输出。训练/推理/测试用到的**原始数据**只有该同构目录（如 `out`）里的内容；BASE_DIR 是它的父目录，并用来存放生成的 jsonl、.seek.map 等。

---

## 阅读前：两个关键路径（请先确定）

后文所有命令会用到这两个路径，请先定好并替换文档中的占位符：

| 变量 | 含义 | 示例（请改成你的实际路径） |
|------|------|-----------------------------|
| **BASE_DIR** | **与 `out` 同构的那一层的父目录**（不是 out 本身）。即：你有一个目录（通常叫 `out`），里面是 `out/<paper_id>/01.png`、`01.mmd`、`meta.json`；BASE_DIR 就是放这个 `out` 的上一级目录。训练/推理/测试用到的**原始数据**只有这个 `out` 目录里的内容；BASE_DIR 还会用来存放生成的索引文件（jsonl、.seek.map）并作为路径解析的根。 | `/data/my_nougat_data`（此时真实数据在 `/data/my_nougat_data/out/<paper_id>/...`） |
| **MODEL_DIR** | 存放 **GitHub 0.1.0-base** 的本地目录（推理与微调统一用此目录） | `/root/autodl-tmp/models/nougat-base` |

文档中出现的 `/path/to/base_dir`、`/path/to/nougat-base` 等，请一律替换为你的 `BASE_DIR`、`MODEL_DIR`。

### 环境准备（首次操作前执行一次）

- **环境与依赖**：见项目根目录 **README** 的「快速开始 / setup_env」一节（中英文 README 均已包含 `setup_env.sh` 用法）。
- 进入本仓库根目录，并安装依赖（若未使用 setup_env，可手动）：
  ```bash
  cd /path/to/nougat   # 你的仓库根目录
  pip install -e .     # 或以 README / setup_env.sh 为准安装依赖
  ```
- 后续所有「在本仓库根目录执行」的命令，都默认已经 `cd` 到了仓库根目录。

---

## 一、前提约定

- **BASE_DIR**：即上表的 BASE_DIR，是「和 `__test_0/out` 同构的那一层」的**父目录**（不是 out 本身）。
- **实际数据所在目录**：训练/推理/测试用到的图片和标注，只来自 **BASE_DIR 下的那个子目录**（文档中假定其名为 `out`，与 `__test_0/out` 同构）：
  - `BASE_DIR/out/<paper_id>/01.png`, `01.mmd`, `02.png`, `02.mmd`, …
  - 每个 `<paper_id>` 目录下需要有 **meta.json**（见第三节；若无则用脚本生成）。
- **规模**：约 33474 个 `.mmd` 文件（即 33474 个「页」样本）。
- **目标**：跑通 nougat-base 推理、跑通 Nougat 微调，并为后续评测预留约 3.2k 测试集。

---

## 二、模型下载与权重导出（统一使用 GitHub 0.1.0-base）

**重要**：本仓库的推理（test.py）与微调（train.py）**只与同一套模型格式兼容**。请**统一使用 GitHub Release 的 nougat-base（0.1.0-base）**，不要使用 HuggingFace 的 `facebook/nougat-base` 目录——后者保存格式与本仓库不一致，会导致「部分权重未加载」、推理/训练无效。

### 2.1 下载与本仓库兼容的 nougat-base（推荐唯一方式）

**步骤 1：确定 MODEL_DIR 并清空旧内容（若之前用过 HuggingFace 的 nougat-base）**

```bash
export MODEL_DIR=/root/autodl-tmp/models/nougat-base   # 或你的实际路径
mkdir -p "$MODEL_DIR"
# 若该目录下已有从 HuggingFace 下载的文件，请先备份或删除，再执行下一步
```

**步骤 2：从 GitHub Release 下载 0.1.0-base 到 MODEL_DIR**

在本仓库根目录执行（需能访问 GitHub）：

```bash
cd /path/to/nougat   # 你的仓库根目录
python -m nougat.utils.checkpoint base "$MODEL_DIR"
```

执行后，`$MODEL_DIR` 下会有：`config.json`、`pytorch_model.bin`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`。

**步骤 3：确认目录内容**

```bash
ls "$MODEL_DIR"
# 应至少包含：config.json  pytorch_model.bin  tokenizer.json  tokenizer_config.json  special_tokens_map.json
```

之后**推理与微调**都使用该 `MODEL_DIR` 即可。

---

### 2.2 为训练导出 aligned_model_weights.pth

训练脚本会先按 `model_path` 构建模型，再加载 **state_dict** 文件 `aligned_model_weights.pth`，因此需从当前 MODEL_DIR 导出一份。

在**本仓库根目录**执行（MODEL_DIR 为 2.1 中已下载好的 GitHub 0.1.0-base 目录）：

```bash
cd /path/to/nougat   # 你的仓库根目录
export MODEL_DIR=/root/autodl-tmp/models/nougat-base   # 与 2.1 一致

python -c "
from pathlib import Path
import torch
from nougat import NougatModel

model_path = Path('$MODEL_DIR')
model = NougatModel.from_pretrained(str(model_path))
torch.save(model.state_dict(), model_path / 'aligned_model_weights.pth')
print('Saved to', model_path / 'aligned_model_weights.pth')
"
```

训练 config 中填写：`model_path: "$MODEL_DIR"`、`pretrained_weight_path: "$MODEL_DIR/aligned_model_weights.pth"`、`tokenizer: "$MODEL_DIR/tokenizer.json"`（见第七节）。

---

## 三、base_dir 数据结构与 meta.json

与 `__test_0/out` 同构即：

- `base_dir/out/<paper_id>/<page>.png`、`<page>.mmd`（如 `01.png`, `01.mmd`）。
- 训练用索引脚本 **create_index** 会读取每个 `<paper_id>` 下的 **meta.json**，且要求：
  - 存在 `num_pages`；
  - 可选：`pdffigures.figures` 等（用于 bbox 等元数据）。

**若你当前没有 meta.json，或格式不兼容**，需要先为每个 paper 目录生成一个最小 meta（仅 `num_pages` 即可）。在**本仓库根目录**执行（把 `BASE_DIR` 换成你的数据根目录）：

```bash
cd /home/eliosilver/nougat
export BASE_DIR=/path/to/your/data   # 替换为实际 base_dir
python -c "
from pathlib import Path
import json
out = Path('$BASE_DIR') / 'out'
for d in out.iterdir():
    if not d.is_dir():
        continue
    n = len(list(d.glob('*.mmd')))
    if n == 0:
        continue
    meta = {'num_pages': n, 'pdffigures': {'figures': []}}
    (d / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('meta.json generated under', out)
"
```

---

## 四、数据划分：训练集 / 验证集 / 测试集

建议按 **论文（paper_id）** 划分，避免同一篇论文同时出现在训练和测试中。比例建议 **80 : 10 : 10**，测试集约 3200 条。

**重要**：以下 4.1～4.3 的命令都要在「能 import 到 `nougat`」的环境下执行。推荐先 `cd` 到**本仓库根目录**，再在命令里用 `BASE_DIR` 指定数据位置；若已 `pip install -e .`，也可以先 `cd $BASE_DIR`，再执行 `python -m nougat.dataset.*`（此时需保证当前环境能找到 nougat 包）。

### 4.1 生成全量索引（jsonl）

在**本仓库根目录**执行（把 `BASE_DIR` 换成你的数据根目录）：

```bash
cd /home/eliosilver/nougat
export BASE_DIR=/path/to/your/data   # 替换为实际 base_dir

python -m nougat.dataset.create_index \
  --dir "$BASE_DIR/out" \
  --root "$BASE_DIR" \
  --out "$BASE_DIR/all_pages.jsonl" \
  --workers 8
```

- `--dir`：所有论文子目录的父目录（即 `base_dir/out`）。
- `--root`：索引里记录的 `image` 路径会相对这个 root，这样生成的每行里是 `out/2303.00058/01.png` 这类相对 base_dir 的路径。
- 若某 paper 目录缺少 meta.json，该 paper 会被跳过；可先用第三节脚本补全 meta。

执行后应得到 `$BASE_DIR/all_pages.jsonl`。

### 4.2 按 paper 划分 train / val / test

仍在**本仓库根目录**执行：

```bash
cd /home/eliosilver/nougat
export BASE_DIR=/path/to/your/data   # 与 4.1 一致

python -m nougat.dataset.split_train_val_test \
  --input "$BASE_DIR/all_pages.jsonl" \
  --out_dir "$BASE_DIR" \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
  --seed 42
```

会在 `$BASE_DIR` 下生成 `train.jsonl`、`validation.jsonl`、`test.jsonl`。

### 4.3 生成 seek map（训练/测试读取索引用）

仍在**本仓库根目录**执行：

```bash
cd /home/eliosilver/nougat
export BASE_DIR=/path/to/your/data

python -m nougat.dataset.gen_seek \
  "$BASE_DIR/train.jsonl" \
  "$BASE_DIR/validation.jsonl" \
  "$BASE_DIR/test.jsonl"
```

会在**各 jsonl 所在目录**生成同名 stem 的 `.seek.map`：`train.seek.map`、`validation.seek.map`、`test.seek.map`。

保持 `base_dir/out/<paper_id>/` 的树形结构即可，无需展平或改文件名；按 paper_id 划分即可正确得到互不重叠的训练集、验证集、测试集。

---

## 五、训练/测试时图片路径：root_name 必须设为空（重要）

### 5.1 为什么会有「arxiv」？必须改吗？

代码里有一个参数 **root_name**，默认值是 **"arxiv"**。这是历史原因：官方/早期数据可能把图片放在名为 `arxiv` 的子目录下，所以「图片完整路径」是按下面公式拼出来的：

```
实际访问的图片路径 = path_to_root / root_name / jsonl 里每行的 "image" 字段
```

- **path_to_root**：就是你放 `train.jsonl` 的目录，即 **BASE_DIR**（例如 `/data/my_nougat_data`）。
- **jsonl 里每行的 "image"**：由第四节 `create_index` 生成，形如 `out/2303.00058/01.png`（相对 base_dir）。

因此：

- 若 **root_name 保持默认 "arxiv"**，程序会去找：  
  `base_dir/arxiv/out/2303.00058/01.png`  
  而你的真实文件在 `base_dir/out/2303.00058/01.png`，**没有 arxiv 这一层**，就会报错「找不到图片」。
- 所以**我们的目录结构下，必须把 root_name 设成空字符串 `""`**，这样：  
  `base_dir + "" + "out/2303.00058/01.png"` = `base_dir/out/2303.00058/01.png`，才和真实路径一致。

**总结**：没有任何要求「必须叫 arxiv」；只是代码默认多拼了一层 `arxiv/`。我们的数据在 `base_dir/out/...`，所以要设 `root_name: ""`。

### 5.2 你需要做的

- **训练**：在 `config/train_nougat.yaml`（或你的副本）里增加一行：
  ```yaml
  root_name: ""
  ```
- **测试/推理**：`test.py` 已默认 `root_name=""`；请用**绝对路径**指定 `--dataset`。若出现「All batches skipped」，当前实现已改为 `num_workers=0`、`pin_memory=False`，直接重跑即可。
  ```bash
  --dataset /path/to/base_dir/test.jsonl
  ```

同时，训练 config 里的 `train_dataset_paths` / `val_dataset_paths` 要用**绝对路径**指向 base_dir 下的 jsonl，例如（把路径换成你的 BASE_DIR）：

```yaml
train_dataset_paths: ["/data/my_nougat_data/train.jsonl"]
val_dataset_paths: ["/data/my_nougat_data/validation.jsonl"]
```

---

## 六、推理流程（nougat-base 批量跑图 → .mmd）

当前仓库的 `predict.py` 是**按 PDF** 推理的；你的数据是 **png + 已有 .mmd 作为 GT**，用 **test.py** 读 test.jsonl 做推理最直接。

### 6.1 使用 test.py（读 jsonl，适合已有 test 集时）

前提：已按第四节在 BASE_DIR 下生成 `test.jsonl` 与 `test.seek.map`。

**模型**：请使用第二节中下载的 **MODEL_DIR**（GitHub 0.1.0-base），勿用 HuggingFace 的 `facebook/nougat-base`。

在**本仓库根目录**执行（把 `BASE_DIR`、`MODEL_DIR` 换成你的路径）：

```bash
cd /path/to/nougat
export BASE_DIR=/root/autodl-tmp/ocr_data
export MODEL_DIR=/root/autodl-tmp/models/nougat-base

python test.py \
  --checkpoint "$MODEL_DIR" \
  --dataset "$BASE_DIR/test.jsonl" \
  --split test \
  --save_path "$BASE_DIR/test_results.json" \
  --batch_size 8 \
  --root_name ""
```

- **root_name**：必须为 `""`（test.py 已默认），否则会去找 `base_dir/arxiv/out/...`，图片会找不到（见第五节）。
- **dataset**：建议用**绝对路径**指向 `$BASE_DIR/test.jsonl`；图片路径 = `test.jsonl 所在目录` + `root_name` + jsonl 中的 `image` 字段（如 `out/0704.1780/01.png`），即数据需在 `base_dir/out/...`。

### 6.2 批量推理脚本思路（产出 pred 目录的 .mmd）

评测指南要求：对 3200 张测试图分别用 base 和 finetuned 模型推理，结果写到 `./data/pred_base/` 和 `./data/pred_finetuned/`。可单独写一个「图片列表 → 模型 → 输出 .mmd」的脚本，例如：

- 输入：一个「图片路径列表」（可从 test.jsonl 解析出），或直接一个目录递归所有 .png。
- 加载模型：`NougatModel.from_pretrained(checkpoint)`，`move_to_device`，`model.eval()`。
- 对每张图：`model.encoder.prepare_input(PIL.Image.open(path))` → `model.inference(image_tensors=...)` → 取 `predictions[0]`，可选 `markdown_compatible(...)`，写入 `out_dir/<与输入同名的>.mmd`。
- 可加 batch 循环以提速（与 `predict.py` 中 LazyDataset 的 batch 类似，但输入为图片路径列表）。

这样即可得到 `data/pred_base/` 与 `data/pred_finetuned/`，再与 `data/test_gt/` 做 NED、公式 Exact Match 等（见 [evaluation_results_and_summary.md](evaluation_results_and_summary.md)）。

### 6.3 推理失败排查（No samples were evaluated / Some weights were not initialized）

- **“No samples were evaluated. All batches were skipped”**  
  - 原因：程序按 `path_to_root + root_name + image` 拼图路径，找不到文件。  
  - 处理：`path_to_root` = test.jsonl 所在目录（即 BASE_DIR）；`image` 来自 jsonl 每行的 `"image"` 字段（如 `out/0704.1780/01.png`）。数据必须在 `BASE_DIR/out/...`，且 `--root_name ""`（默认已是空）。用绝对路径传 `--dataset "$BASE_DIR/test.jsonl"`。

- **“Some weights of NougatModel were not initialized from the model checkpoint”**  
  - 原因：当前 checkpoint 是 HuggingFace 的 vision-encoder-decoder 格式，state_dict 的 key 与本仓库 NougatModel 不兼容，大量权重未加载（相当于随机初始化），推理无意义。  
  - 处理：换用与本仓库匹配的 checkpoint（如本仓库 GitHub Release 提供的权重），或使用 HuggingFace 的 `transformers` 做推理，不要用该 HF 目录作为本仓库的 `--checkpoint`。

---

## 七、微调流程与具体命令

### 7.1 配置文件

复制一份配置再改（避免覆盖原配置）：

```bash
cd /home/eliosilver/nougat
cp config/train_nougat.yaml config/train_nougat_my.yaml
```

用编辑器打开 `config/train_nougat_my.yaml`，**至少**修改以下项（全部写成绝对路径，避免歧义）：

| 配置项 | 含义 | 示例（请改成你的路径） |
|--------|------|------------------------|
| `model_path` | 存放 GitHub 0.1.0-base 的 MODEL_DIR | `"/root/autodl-tmp/models/nougat-base"` |
| `pretrained_weight_path` | 上一步导出的 `aligned_model_weights.pth` 的完整路径 | `"/root/autodl-tmp/models/nougat-base/aligned_model_weights.pth"` |
| `tokenizer` | 同上目录下的 `tokenizer.json` | `"/root/autodl-tmp/models/nougat-base/tokenizer.json"` |
| `train_dataset_paths` | base_dir 下的 train.jsonl | `["/data/my_nougat_data/train.jsonl"]` |
| `val_dataset_paths` | base_dir 下的 validation.jsonl | `["/data/my_nougat_data/validation.jsonl"]` |
| `result_path` | 实验输出根目录（checkpoint、日志会写在这里） | `"/data/experiments/nougat"` |
| `root_name` | **必须**写 `""`，否则训练时找不到图片（见第五节） | `""` |

示例（仅作格式参考，路径请全部替换）：

```yaml
model_path: "/data/models/nougat-base"
pretrained_weight_path: "/data/models/nougat-base/aligned_model_weights.pth"
tokenizer: "/data/models/nougat-base/tokenizer.json"
train_dataset_paths: ["/data/my_nougat_data/train.jsonl"]
val_dataset_paths: ["/data/my_nougat_data/validation.jsonl"]
result_path: "/data/experiments/nougat"
root_name: ""
# 其余如 train_batch_sizes、max_epochs、max_length 等可保持默认
```

### 7.2 启动训练

在**本仓库根目录**执行：

```bash
cd /home/eliosilver/nougat
python train.py --config config/train_nougat_my.yaml
```

可选：`--exp_version v1`、`--debug`（用 TensorBoard 等）。

### 7.3 输出与后续

- 权重与日志会落在 `result_path/exp_name/exp_version/`（如 `result_path/nougat_finetune/v1/`）。
- 最佳 checkpoint 由 `ModelCheckpoint` 的 `monitor="val/edit_dist"` 决定；评测时用该 checkpoint 作为「微调后模型」，做批量推理并与 baseline 对比（见 [evaluation_results_and_summary.md](evaluation_results_and_summary.md)）。

---

## 八、操作清单小结

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 模型下载 / 导出 | 下载 nougat-base，并导出 `aligned_model_weights.pth` 供训练 |
| 2 | meta.json | 若缺失，按第三节为每个 paper 目录生成最小 meta |
| 3 | 全量索引 | 从仓库根目录执行 create_index，`--dir $BASE_DIR/out`、`--root $BASE_DIR`、`--out $BASE_DIR/all_pages.jsonl` |
| 4 | 划分 train/val/test | 按 paper_id 划分，写出 train/validation/test.jsonl |
| 5 | seek map | `gen_seek train.jsonl validation.jsonl test.jsonl` |
| 6 | root_name | 训练/测试时 `root_name=""`，保证图片路径 = base_dir + image |
| 7 | 推理 | 用 test.py 或自写「图片→mmd」脚本，得到 pred_base / pred_finetuned |
| 8 | 微调 | 修改 config → `train.py --config ...` |
| 9 | 评测 | 按 [evaluation_results_and_summary.md](evaluation_results_and_summary.md) 做 NED、公式 Exact Match 等 |

按上述顺序执行即可从 base_dir 跑通推理与微调，并预留测试集用于最终量化评测。
