# 微调验证极慢与 Checkpoint 策略优化复盘

本文档记录：**验证阶段极慢**的原因诊断、代码修改、对训练效果的影响、以及 **checkpoint 保存策略** 的调整与操作说明，便于后续复盘与复现。

---

## 一、问题现象

- 训练时 **Validation DataLoader** 进度极慢（例如 58/2836 要跑数小时）。
- 日志中高频出现 `WARNING:root:Found repetitions in sample 0`。
- 单轮验证耗时与「验证集条数 × 单条自回归生成时间」相当，导致每个 Epoch 的验证阶段成为瓶颈。

---

## 二、根因诊断

### 2.1 验证阶段在做完整自回归生成

在 `lightning_module.py` 的 `validation_step` 中，**每个 validation batch** 都会调用：

```python
preds = self.model.inference(image_tensors=..., return_attentions=False)["predictions"]
```

而 `model.inference()` 内部会执行 `decoder.model.generate(..., max_length=config.max_length)`（例如 2048），即**对每张验证图做一整段自回归生成**。

- 验证集约 2836 条、`val_batch_sizes: [1]` 时，每轮验证 = **2836 次** 完整 generate。
- 单次 generate 在遇到重复或长序列时可能需数十秒，导致整轮验证耗时达到数小时级别。
- **结论**：瓶颈来自「验证阶段对全量验证集做生成」的设计，而非单卡算力不足；升级 A100 等无法根本解决。

### 2.2 “Found repetitions” 的含义

- 在 `nougat/model.py` 的 `inference()` 中，`generate()` 返回后会对 logits 做方差分析，若判定为「重复生成」，则对该条序列做**截断**并打日志 `WARNING: Found repetitions in sample i`。
- 截断可以避免无意义的超长输出，但**在截断之前**，模型已经按步执行了多步自回归，因此单条仍可能很慢；且验证条数多，总时间依然巨大。
- 重复惩罚（见下文）用于在**生成过程中**抑制重复，减少冗长生成与耗时。

### 2.3 与训练逻辑的对比

- **训练**：使用 Teacher Forcing，一次前向传播即可得到 loss，不做自回归生成。
- **验证（修改前）**：对每条验证样本做完整 `inference()` → 自回归生成 → 再算 BLEU/edit_dist 等。
- 工程上更合理的做法是：**训练中的验证以“快速评估拟合程度”为主**，用 loss 即可；耗时的生成与 BLEU/NED 等指标放在**训练结束后**在测试集上跑一次即可。

---

## 三、代码修改概要

### 3.1 验证阶段可配置为「只算 loss、不生成」

| 文件 | 修改内容 |
|------|----------|
| `lightning_module.py` | 读取 `config.val_with_generation`。当为 `false` 时，`validation_step` 仅做一次前向（Teacher Forcing），计算并记录 `val/loss`，不调用 `model.inference()`。 |
| `train.py` | 根据 `val_with_generation` 选择 checkpoint 的 monitor：`val_with_generation=false` 时用 `val/loss`（mode=min）和对应 filename；为 true 时仍用 `val/edit_dist`。 |
| `config/train_nougat_my.yaml` | 增加 `val_with_generation: false`，验证默认只算 loss。 |

效果：每轮验证由「2836 次 generate」变为「若干次前向」，验证耗时从数小时量级降到数分钟量级。

### 3.2 推理阶段防复读：软惩罚 + 长度硬兜底（不用 N-gram，避免破坏 LaTeX）

| 文件 | 修改内容 |
|------|----------|
| `nougat/model.py` | 在 `decoder.model.generate()` 中：**不添加** `no_repeat_ngram_size`（LaTeX 表格/公式中合法重复的 5-gram 很多，该参数会误杀并导致公式错乱）；仅采用 **`repetition_penalty=1.15`**（软惩罚，抑制复读不封杀合法重复）+ **`max_new_tokens=1500`**（单页生成上限，触达即截断，防止死循环耗算力）。 |

### 3.3 Checkpoint 多存几份

| 文件 | 修改内容 |
|------|----------|
| `train.py` | 使用 `config.get("save_top_k", 5)`，按 monitor 指标保留**最优的 N 个** checkpoint（默认 5）。 |
| `config/train_nougat_my.yaml` | 增加 `save_top_k: 5`。 |

当前策略：

- **save_last=True**：始终保存 `last.ckpt`（最后一个 epoch 结束时的权重）。
- **save_top_k=5**：按 `val/loss`（当 `val_with_generation=false` 时）保留指标最优的 5 个 ckpt，文件名形如 `epoch=XX-val_loss=0.xxxx.ckpt`。
- 如需更多或更少，只需在 yaml 中修改 `save_top_k` 即可。

---

## 四、对训练效果的影响

### 4.1 验证只算 loss、不生成

- **训练过程本身未改**：仍为 Teacher Forcing + 反向传播，优化目标与数据不变，**训练效果（收敛、拟合能力）不受影响**。
- **唯一变化**：训练过程中不再得到「val/edit_dist、BLEU」等生成类指标，checkpoint 的“最优”由 **val/loss** 决定（越低越好）。
- **合理性**：val/loss 与模型在验证集上的拟合程度强相关，一般 val/loss 更低时，在测试集上做生成评测也会更好；用 val/loss 选 best 是常见做法。
- **最终评测**：训练结束后，用 `test.py` 或自写脚本在 **test 集**上做一次完整生成，计算 BLEU、NED 等，用于报告和对比 base vs 微调。

### 4.2 推理防复读策略（repetition_penalty + max_new_tokens，不用 no_repeat_ngram_size）

- 仅作用于 **inference/generate**，不参与训练，**不改变训练效果**。
- **为何不用 no_repeat_ngram_size**：LaTeX 中表格对齐、公式等天然存在大量合法重复 N-gram（如 `\\ \hline & \quad`），硬性禁止会迫使模型为规避惩罚而输出错乱，破坏公式解析能力。
- **当前方案**：**软惩罚** `repetition_penalty=1.15`（概率层面压制复读）+ **长度硬兜底** `max_new_tokens=1500`（单页最大生成 token 数，到顶即停），在防死循环、提速的同时不误伤 LaTeX 结构。

### 4.3 多存 checkpoint（save_top_k=5）

- 不改变训练曲线或收敛行为。
- 好处：可事后用不同 epoch 的 ckpt（如 best、epoch-10、epoch-20）在测试集上跑评测，选表现最好的或做曲线分析。

---

## 五、Checkpoint 保存策略小结

| 项目 | 说明 |
|------|------|
| **last.ckpt** | 每个 run 都会保存，对应最后一个 epoch 结束时的权重。 |
| **按指标保存** | 当 `val_with_generation: false` 时，按 **val/loss** 保留最优的 **save_top_k** 个（默认 5），文件名含 epoch 与 val_loss。 |
| **保存目录** | `result_path/exp_name/exp_version/`，例如 `nougat_exp/nougat_finetune/v1/`。 |
| **如何多存** | 在 `train_nougat_my.yaml` 中增大 `save_top_k`（如 10）；如需按 epoch 间隔再存一份，可后续在 Trainer 中增加 `ModelCheckpoint(every_n_epochs=5)` 等。 |

建议：保留当前 `save_top_k: 5` + `save_last=True`，既有一份“最后一轮”的 last，又有 5 个“val/loss 最优”的 ckpt 供后续评测与复现。

### 5.1 训练时长与早停（省算力）

- **粗算**：每 epoch 约 21k batch、约 2 it/s → 约 **2.5–3 h/epoch**；30 epoch 约 **75–90 h（约 3–4 天）** 连续 GPU。
- **早停**：在配置中增加 `early_stopping_patience`（如 5）。当 **val/loss** 连续 5 个 epoch 无提升时自动停止，最优权重已由 ModelCheckpoint 按 val/loss 保存，无需跑满 30 epoch。多数情况下 10–20 epoch 内会触发早停，总时长可降到约 1–2 天。
- 配置示例：`early_stopping_patience: 5`（0 表示不早停）。若算力紧张，可同时将 `max_epochs` 改为 20，再配合早停即可。

---

## 六、训练注意：早停与学习率

若最佳 val/loss 出现在早期 epoch（如 epoch 1），早停后主观感觉指令遵循/任务适应尚未充分，可参考以下原因与改法。

**现象**：最佳 val/loss 出现在 **epoch 1**（即只经过约 2 个 epoch 的有效训练），之后连续若干 epoch 无提升即早停。

**可能原因**：

1. **学习率偏大**：如 `lr: 5e-5` 在有效 batch 较大（如 36/72）时，前期 loss 掉得快，易过早进入平坦区或局部最优，val/loss 不再明显下降；模型还未在“任务格式/指令”上充分收敛即被早停。
2. **早停 patience 偏小**：如 `early_stopping_patience: 5` 时，最佳若出现在 epoch 1，后面 5 个 epoch 不提升就停，可训练轮数偏少。
3. **验证集噪声或规模**：每 epoch 只验证 1 次时，val/loss 可能波动；早期偶然较低易触发早停。可考虑适当增大 `val_check_interval` 或多看几轮再判断，但优先先调 lr 与 patience。
4. **warmup 与衰减**：衰减较慢时，主要矛盾更可能在「初始 lr 偏大」而非衰减形状。

**建议（按优先级）**：

1. **降低学习率重训一版**：在 `config/train_nougat_my.yaml` 中把 `lr` 改为 **3e-5**（或 2e-5），用新版本号避免覆盖：`python train.py --config config/train_nougat_my.yaml --exp_version v2`。预期 loss 下降更平滑，最优可能出现在更晚的 epoch。
2. **增大早停耐心**：配置中设 `early_stopping_patience: 10`，与较低 lr 配合，让训练多跑几轮再停。
3. **可选**：略降 `min_lr`（如从 7.5e-6 改为 5e-6），给后期留一点学习空间；非必须，先改 lr 即可。

**重训 v2 配置变更小结**：

| 配置项 | v1（示例） | v2 建议 | 说明 |
|--------|------------|--------|------|
| lr | 5e-5 | **3e-5** | 减缓前期下降、多训几轮再收敛 |
| early_stopping_patience | 5 | **10** | 避免过早停在“第二个 epoch 就最优” |
| exp_version | v1 | **v2** | 用 `--exp_version v2` 或 yaml 中改，不覆盖 v1 |

其余（batch、accumulate、max_length、数据路径等）可保持不变。若 v2 仍感觉指令遵循不足，可再试 lr=2e-5 或略增 warmup_steps。

---

## 七、具体操作（复现与后续使用）

### 7.1 训练（验证只算 loss、多存 ckpt）

```bash
cd /root/autodl-tmp/nougat
conda activate nougat

python train.py --config config/train_nougat_my.yaml
```

- 确认 `config/train_nougat_my.yaml` 中有：
  - `val_with_generation: false`
  - `save_top_k: 5`
- 训练结束后，在 `result_path/exp_name/exp_version/` 下会有：
  - `last.ckpt`
  - 最多 5 个按 val/loss 最优的 `epoch=XX-val_loss=0.xxxx.ckpt`

### 7.2 用某一 checkpoint 做测试集推理

```bash
export BASE_DIR=/root/autodl-tmp/ocr_data
export EXP_VERSION=v1   # 或你实际使用的 version

# 用「val/loss 最优」的某个 ckpt（需替换为实际文件名）
python test.py \
  --checkpoint /root/autodl-tmp/nougat_exp/nougat_finetune/${EXP_VERSION}/epoch=XX-val_loss=0.xxxx.ckpt \
  --dataset "$BASE_DIR/test.jsonl" \
  --split test \
  --save_path "$BASE_DIR/test_results_finetuned.json" \
  --batch_size 4 \
  --root_name ""
```

（若 test.py 支持从 ckpt 自动加载权重，则上述方式有效；若当前实现只支持“目录形式”的 checkpoint，则需先从 ckpt 中导出权重到目录再指向该目录，可后续再补脚本。）

### 7.3 若希望训练过程中偶尔看生成指标

- 可将 `val_with_generation` 设为 `true`，并适当减小验证量（例如 `val_batches: 0.1`，只验证 10% 的验证集），这样每轮会慢一些但仍有 BLEU/edit_dist；或保持 `val_with_generation: false`，仅在需要时手动跑一次验证集上的 test.py 做生成评测。

---

## 八、文档与配置变更索引

| 文档/配置 | 说明 |
|-----------|------|
| 本文档 | `docs/validation_and_checkpoint_notes.md`：诊断、修改、影响、操作。 |
| `config/train_nougat_my.yaml` | `val_with_generation: false`、`save_top_k: 5`。 |
| `lightning_module.py` | `validation_step` 在 `val_with_generation=false` 时只算 val/loss。 |
| `train.py` | checkpoint 的 monitor/filename 随 `val_with_generation` 切换；`save_top_k` 从 config 读取。 |
| `nougat/model.py` | `inference()` 中 generate：`repetition_penalty=1.15` + `max_new_tokens=1500`，**不使用** `no_repeat_ngram_size`。 |

以上为本次微调验证与 checkpoint 优化的完整复盘与操作说明。

---

## 附录：推理 generate() 最终参数（nougat/model.py）

```python
# 防复读：仅用软惩罚 + 长度硬兜底，不用 no_repeat_ngram_size
max_new_tokens = getattr(self.config, "max_new_tokens", 1500)
decoder_output = self.decoder.model.generate(
    encoder_outputs=encoder_outputs,
    min_length=1,
    max_length=self.config.max_length,
    max_new_tokens=max_new_tokens,
    ...
    repetition_penalty=getattr(self.config, "repetition_penalty", 1.15),
    # 不传 no_repeat_ngram_size，避免破坏 LaTeX 表格/公式中的合法重复
    stopping_criteria=...,
)
```

可通过模型 config 覆盖：`repetition_penalty`、`max_new_tokens`（例如单页更长时可适当调大）。
