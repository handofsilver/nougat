# 推理输入/输出与 JSON 格式说明

本文档面向**需要对推理结果做进一步评估的工程师**，说明以下三份文件的来源、生成命令、推理参数，以及 JSON/JSONL 格式与数据对应关系。不依赖项目背景即可据此做评测脚本或指标计算。

---

## 一、文件清单与用途

| 文件路径 | 用途 |
|----------|------|
| `.../ocr_data/test.jsonl` | **测试集索引**：每行一条样本，含图片路径与标准答案（markdown）。推理时按该顺序喂入模型。 |
| `.../ocr_data/test_results_base.json` | **基座模型（零样本）推理结果**：对 test.jsonl 用未微调的 Nougat 0.1.0-base 做批量推理后的输出。 |
| `.../ocr_data/test_results_finetuned_v2.json` | **微调模型（v2）推理结果**：对同一 test.jsonl 用微调 checkpoint（epoch=02-val_loss=0.0461）做批量推理后的输出。 |

三份文件均位于项目内的 `ocr_data` 目录（或你部署时的等价路径），路径前缀可能为 `.../nougat/ocr_data` 或 `$BASE_DIR`。

---

## 二、来源与生成命令

### 2.1 test.jsonl

**来源**：由数据管线生成，并非单条命令直接产出。

- 上游为「按论文划分的页级索引」：先对 `out/` 下所有论文页做 `create_index` 得到 `all_pages.jsonl`，再经 `split_train_val_test` 按 paper_id 划分得到 `train.jsonl`、`validation.jsonl`、**test.jsonl**。
- 生成后通常还会对三个 jsonl 跑 `gen_seek` 得到对应的 `.seek.map`（训练/验证用；推理仅需 jsonl）。

**未提供单条复现命令**：若你只有这三份文件，可直接使用现有 `test.jsonl`，无需重跑数据管线。若需从原始数据重建，需在项目根目录执行与 [pipeline_and_training_guide.md](pipeline_and_training_guide.md) 中第四节一致的流程（create_index → split_train_val_test → gen_seek）。

---

### 2.2 test_results_base.json（基座零样本）

**生成命令**（在项目根目录、已激活对应 conda 环境时）：

```bash
export BASE_DIR=/path/to/ocr_data   # 即 test.jsonl 所在目录
export MODEL_DIR=/path/to/nougat-base   # 0.1.0-base 权重目录

python test.py \
  --checkpoint "$MODEL_DIR" \
  --dataset "$BASE_DIR/test.jsonl" \
  --split test \
  --save_path "$BASE_DIR/test_results_base.json" \
  --batch_size 16 \
  --root_name ""
```

**重要参数说明**：

| 参数 | 取值 | 说明 |
|------|------|------|
| `--checkpoint` | 基座模型目录 | 须为 GitHub Release 0.1.0-base 格式（含 config.json、pytorch_model.bin、tokenizer.json 等），与 HuggingFace 的 facebook/nougat-base 不兼容。 |
| `--dataset` | test.jsonl 的绝对路径 | 推荐绝对路径；程序用其所在目录作为 `path_to_root`，与每行的 `image` 字段拼接得到图片路径。 |
| `--split` | test | 与文件名 test.jsonl 对应，用于加载该 split。 |
| `--root_name` | "" | 必须为空。图片路径 = path_to_root + root_name + image；若不为空会多出一层子目录导致找不到图。 |
| `--batch_size` | 16 | 推理 batch 大小，影响显存与吞吐。 |
| `--num_workers` | 默认 0 | 可选；大于 0 时可加快数据加载，部分环境可能触发多进程问题。 |

**推理过程关键逻辑**（代码层面，便于你理解结果）：

- 每条样本：读入对应图片 → encoder 编码 → decoder 自回归生成文本（不喂 GT）。
- 生成阶段使用：**max_new_tokens**（来自 checkpoint 的 config，未显式设时多用 `max_length`，如 2048）、**repetition_penalty=1.15**、**无 no_repeat_ngram_size**（避免破坏 LaTeX/公式中的合法重复）。
- 生成结果与 test.jsonl 中该条的 `markdown`（经 tokenizer 解码后）逐条算 edit_dist、bleu、meteor 等；若某条 pred 或 GT 长度 &lt; 4，该条不参与 per-sample 指标，仅仍写入 predictions/ground_truths。

---

### 2.3 test_results_finetuned_v2.json（微调 v2）

**生成命令**：

```bash
export BASE_DIR=/path/to/ocr_data

python test.py \
  --checkpoint "/path/to/nougat_exp/nougat_finetune/v2/epoch=02-val_loss=0.0461.ckpt" \
  --dataset "$BASE_DIR/test.jsonl" \
  --split test \
  --save_path "$BASE_DIR/test_results_finetuned_v2.json" \
  --batch_size 16 \
  --root_name ""
```

**与基座差异**：仅 `--checkpoint` 和 `--save_path` 不同；其余参数（dataset、split、root_name、batch_size、推理内部 max_new_tokens/repetition_penalty 等）与基座一致。checkpoint 为 Lightning 保存的 `.ckpt`，程序会从中加载权重并在同一 test.jsonl 上按相同顺序推理。

---

## 三、JSON / JSONL 格式与对应关系

### 3.1 test.jsonl（输入）

- **格式**：每行一个 JSON 对象，UTF-8，换行分隔（JSONL）。
- **行数**：3491（本批数据）。
- **每行字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | string | 相对 path_to_root 的图片路径，如 `out/0704.1280/01.png`。完整路径 = path_to_root + image（root_name 为空时）。 |
| `markdown` | string | 该页的标准答案（Ground Truth），与推理结果中的 `ground_truths[i]` 对应（见下）。可能含 [TITLE]、[AUTHOR]、[FIGURE] 等结构化标签。 |
| `meta` | array/object | 页级元数据（如 bbox 等），评测时一般不用。 |

**示例（第 1 条，仅展示结构；markdown 已截断）**：

```json
{
  "image": "out/0704.1280/01.png",
  "markdown": "[TITLE]# Controllable Quantum Switchboard[END_TITLE]\n[AUTHOR]D. Kaszlikowski,1 L. C. Kwek,2 C. H. Lai,1 and V. Vedral3[END_AUTHOR]\n[AUTHOR]1Department of Physics, National University of Singapore, 2 Science Drive 3, Singapore 117542[END_AUTHOR]\n...",
  "meta": []
}
```

**示例（第 2 条）**：

```json
{
  "image": "out/0704.1280/02.png",
  "markdown": "[FIGURE]\n[FIGURE_TITLE]Figure 1: Suppose Alice wishes to send her auxiliary qubit to Bob. ...",
  "meta": []
}
```

**与结果文件的对应**：`test.jsonl` 第 `i` 行（0-based）对应推理结果中的 `predictions[i]` 与 `ground_truths[i]`。若推理时有个别样本被跳过（如读图失败），结果列表长度可能略小于 3491，此时仍按顺序一一对应到 test.jsonl 中**被成功推理**的那几行（通常与 jsonl 行序一致，仅可能少 1～2 条）。

---

### 3.2 test_results_base.json / test_results_finetuned_v2.json（输出）

- **格式**：单个 JSON 对象，UTF-8；含标量汇总、逐条指标数组、以及两条长字符串数组。
- **顶层键**（两个结果文件结构相同）：

| 键 | 类型 | 说明 |
|----|------|------|
| `edit_dist_accuracy` | float | 全表归一化编辑距离均值（仅对 pred/gt 长度≥4 的条计算）。 |
| `edit_dist_accuracies` | array of float | 每条样本的归一化编辑距离，长度可能略小于 predictions（因 minlen=4 会跳过部分条）。 |
| `bleu_accuracy` | float | 全表 BLEU 均值。 |
| `bleu_accuracies` | array of float | 每条 BLEU。 |
| `meteor_accuracy` | float | 全表 METEOR 均值（环境未装 METEOR 语料时常为 nan）。 |
| `meteor_accuracies` | array of float | 每条 METEOR。 |
| `precision_accuracy`, `precision_accuracies` | float / array | 词级 precision 均值与逐条。 |
| `recall_accuracy`, `recall_accuracies` | float / array | 词级 recall 均值与逐条。 |
| `f_measure_accuracy`, `f_measure_accuracies` | float / array | F1 均值与逐条。 |
| `predictions` | array of string | 模型生成的文本，与 test.jsonl 行序一致，长度通常为 3490 或 3491。 |
| `ground_truths` | array of string | 每条对应的标准答案（来自 test.jsonl 的 markdown，经 tokenizer 解码后的字符串）。 |

**长度说明**：`predictions` 与 `ground_truths` 长度相同（如 3490），且与 test.jsonl 行序一一对应。`*_accuracies` 数组长度可能略小（例如 3445），因为 `compute_metrics` 对 `len(pred)<4` 或 `len(gt)<4` 的条不写入任何指标，该条仍会出现在 `predictions`/`ground_truths` 中。**逐条指标顺序**：`*_accuracies` 的 k 号元素对应的是「第 k 个满足 minlen 条件的样本」（按推理顺序），与 `predictions` 的下标 i 无直接 1:1 对应。做逐条评测时建议以 `predictions[i]` 与 `ground_truths[i]` 自算指标；若直接使用 `*_accuracies`，需自行维护「参与计算的样本」与 test.jsonl 行号的映射。

**数据举例（5 条）**：以下为从 `test_results_base.json` 中取出的前 5 条，仅展示 `predictions` / `ground_truths` 的预览与长度；实际 JSON 中为完整字符串。

```json
[
  {
    "index": 0,
    "prediction_preview": "\n\n# Controllable Quantum Switchboard\n\nD. Kaszlikowski\n\nDepartment of Physics, National University of Singapore, 2 Science Drive 3, Singapore 117542\n\nL. C. Kwek\n\nNanyang Technological University...",
    "ground_truth_preview": "[TITLE]# Controllable Quantum Switchboard[END_TITLE]\n[AUTHOR]D. Kaszlikowski,1 L. C. Kwek,2 C. H. Lai,1 and V. Vedral3[END_AUTHOR]\n[AUTHOR]1Department of Physics, National University of Singapore...",
    "pred_len": 5640,
    "gt_len": 5794
  },
  {
    "index": 1,
    "prediction_preview": "measurement to Bob. Using the information from Charlene, Bob can perfectly recover the state of the Alice's auxiliary qubit.\n\nThe situation is entirely symmetric...",
    "ground_truth_preview": "[FIGURE]\n[FIGURE_TITLE]Figure 1: Suppose Alice wishes to send her auxiliary qubit to Bob...",
    "pred_len": 4947,
    "gt_len": 5320
  },
  {
    "index": 2,
    "prediction_preview": "which he can transform back to the state \\(|\\alpha\\rangle\\) by applying the inverse unitary transformation \\(U^{\\dagger}_{mn,kl}\\)...",
    "ground_truth_preview": "[TEXT]which he can transform back to the state \\(|\\alpha\\rangle\\) by applying the inverse unitary transformation \\(U_{mn,kl}^{\\dagger}\\)...",
    "pred_len": 4497,
    "gt_len": 4857
  },
  {
    "index": 3,
    "prediction_preview": "\n\n# On the spectral functions of scalar mesons\n\nFrancesco Giacosa and Giuseppe Pagliara\n\nInstitut fur Theoretische Physik...",
    "ground_truth_preview": "[TITLE]# On the spectral functions of scalar mesons[END_TITLE]\n[AUTHOR]Francesco Giacosa and Giuseppe Pagliara[END_AUTHOR]...",
    "pred_len": 5704,
    "gt_len": 5102
  },
  {
    "index": 4,
    "prediction_preview": "the corresponding spectral function shows consistent deviations from the usual Breit-Wigner one. In Section III we turn to the two-channel case...",
    "ground_truth_preview": "[TEXT]the corresponding spectral function shows consistent deviations from the usual Breit-Wigner one. In Section III we turn to the two-channel case...",
    "pred_len": 4009,
    "gt_len": 4346
  }
]
```

说明：上表为便于阅读的「预览 + 长度」；真实文件中 `predictions` 与 `ground_truths` 为两个一维字符串数组，无 `prediction_preview` 等键。GT 中常见 `[TITLE]`、`[AUTHOR]`、`[FIGURE]`、`[TEXT]`、`[SUBTITLE]` 等标签；模型预测可能不带这些标签或格式略有差异，评测时如需可先做规范化或标签对齐。

---

## 四、进一步评估时需要注意的点

1. **顺序一致**：同一 test.jsonl 产生的 base 与 finetuned 结果，`predictions`/`ground_truths` 与 test.jsonl 行序一致；**base 与 finetuned 的同一 index 对应同一条样本**，可直接按 index 做对比（如同一 index 的 base pred vs finetuned pred vs GT）。
2. **文本清洗**：做 NED、公式 Exact Match 等前，建议对 pred/GT 做统一清洗（去首尾空白、合并连续换行、全角转半角、图片占位符泛化等），否则指标会偏悲观。可参考 [evaluation_results_and_summary.md](evaluation_results_and_summary.md) 中的清洗规则。
3. **指标复现**：当前实现中，`edit_dist` = 编辑距离 / max(len(pred), len(gt))；BLEU 为句子级、带 smoothing。若你自算指标，需与 `nougat/metrics.py` 中 `compute_metrics` 的约定一致，否则与文件中的 `*_accuracy` 会有数值差异。
4. **缺失/异常**：若某条推理失败被跳过，结果列表中会少一条，此时 test.jsonl 行号与 index 的对应需以「实际参与推理的行」为准（通常顺序一致、仅尾部可能少 1～2 条）。

以上三份文件的来源、命令、参数与格式说明完毕，可直接用于编写评测脚本或报告。
