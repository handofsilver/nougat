## 一、项目背景与问题设定

- **任务目标**：给定一页学术 PDF 渲染图像，端到端输出**结构化 Markdown（.mmd）**，包含标题、作者、正文、公式、图表、脚注等语义结构，而不仅是「纯文本 OCR」。
- **官方 Nougat 的局限**：
  - 训练标签是相对**扁平的 Markdown**，缺少 `[TITLE]`、`[AUTHOR]`、`[FIGURE]` 等语义标签。
  - 分页仅用启发式切分，图/表/算法（TFA）这种浮动体的页面归属经常错误。
- **本项目的两大工程贡献**：
  1. 在 `docs/data_engineering_design.md` 设计并实现了从 `.tex` 到「按页对齐的结构化 `.mmd` / `.png`」的**完整数据流水线**。
  2. 在此基础上，对 Nougat 进行了**结构化标签+分页友好**的数据增广与 SFT 微调，并在独立测试集上对比 **预训练模型（Baseline） vs 微调模型（Finetuned）** 的表现。

本总结文档聚焦于：

1. **数据与模型侧背景**：实际做了哪些处理、预训练推理和 SFT 推理在输出上的根本区别。
2. **评测方案**：如何在格式约束不同的前提下，科学地比较 Baseline 与 Finetuned。
3. **评测结果与分析**：实际跑出来的核心数字，以及这些数字背后可能的原因。

---

## 二、数据工程与输出格式差异

### 2.1 流水线回顾：从 `.tex` 到页级 `.mmd` / `.png`

根据 `docs/data_engineering_design.md`，主流程可以概括为：

1. **TeX 预处理 + HTML 转换**
   - 清洗 `.tex`（去注释/无用命令）→ 调用 LaTeXML 生成 `.html`。
   - 通过 `fix_bibliography.py`、`fix_citations.py` 修复参考文献和引用链。
2. **HTML → 语义化 `.mmd`**
   - `latexml_parser.py` 将 HTML 解析为**语义元素树**，识别标题、作者、公式、图表、算法等。
   - `markdown.py` 将元素树格式化为带语义标签的 `.mmd`，典型标签包括：
     - `[TITLE]...[END_TITLE]`、`[AUTHOR]...[END_AUTHOR]`
     - `[SUBTITLE]...[END_SUBTITLE]`、`[TEXT]...[END_TEXT]`
     - `[FORMULA]...[END_FORMULA]`
     - `[FIGURE]...[END_FIGURE]`、`[FIGURE_TITLE]...[END_FIGURE_TITLE]`
     - `[TABLE]...[END_TABLE]`、`[ALGORITHM]...[END_ALGORITHM]` 等。
   - `inject_coords_to_mmd.py` 将 pdffigures2 提取的图像坐标注入 `[FIGURE_COORDS](...)[END_FIGURE_COORDS]`。
3. **分页对齐（核心）**
   - `split_md_to_pages.py` + `split_utils/*`（`line_matcher.py`、`markdown_parser.py`、`page_splitter.py` 等）：
     - 把整篇 `.mmd` 与 PDF 各页文本对齐，生成**页级 `.mmd`**。
     - 同时用 `pypdfium2` 渲染页级 `.png`。
4. **首页版面修正（可选）**
   - `layout_correction.py` + `layout_parser`：用 DiT 模型检测首页图像中的「标题/作者」区域，若 `.mmd` 中标签缺失，则对首页 `.mmd` 做标签重标（retag），不改文字内容。

最终产物形态：

- 每篇论文 → 多个页级样本：
  - `01.mmd`, `01.png`, `02.mmd`, `02.png`, ...
  - 通过 `create_index.py` 打包成 `all_pages.jsonl`，再按论文 ID 划分得到 `train/validation/test.jsonl`，其中 `markdown` 字段即页级 `.mmd`。

### 2.2 预训练模型 vs 微调模型：输出本质区别

#### Ground Truth（GT）

无论是 Baseline 还是 Finetuned，评测时使用的 GT 都来自 `test.jsonl` 的 `markdown` 字段：

- 实际内容是**本项目流水线生成的结构化 `.mmd`**，包含 `[TITLE]`、`[FIGURE]` 等标签。
- `nougat/utils/dataset.py` 中 `SciPDFDataset.__getitem__`：
  - `img` ← 对应页的 `.png`。
  - `ground_truth` ← `data.pop("markdown")`，即结构化 `.mmd`。

#### Baseline（预训练 Nougat）

- 官方 `nougat-base` 在原始语料上训练，标签相对扁平，**不含本项目自定义的结构化标签**。
- 推理时（见 [inference_io_and_json_format.md](inference_io_and_json_format.md)）：
  - 同一 `test.jsonl`，按行顺序加载图像。
  - 用 base checkpoint 做自回归生成，decoder 输出经过 tokenizer 解码和 `postprocess`。
  - 生成结果写入 `ocr_data/test_results_base.json` 的 `predictions` 字段。
- **关键点**：Baseline 的输出大多数是「常规 Markdown」（`#` 标题、普通段落），**不会主动产生 `[TITLE]`、`[FIGURE]` 等结构化标签**。

#### Finetuned（SFT 后 Nougat）

- 使用上面构建的结构化 `.mmd` 作为训练标签，对同一体系结构的 Nougat 做 SFT：
  - encoder 仍然接收页级 `.png`。
  - decoder 的监督信号是带标签 `.mmd`，即模型被鼓励学习：
    - 语义结构（标题/作者/正文/图表/脚注等）。
    - 更稳定的分页对齐（每一页应该只生成本页内容）。
- 推理时：
  - 与 Baseline 使用**相同的 test.jsonl、同一张图片序列**。
  - 仅 checkpoint 不同（如 `epoch=02-val_loss=0.0461.ckpt`）。
  - 生成结果写入 `ocr_data/test_results_finetuned_v2.json` 的 `predictions` 字段。

**小结**：

- **GT**：始终是「本项目生成的结构化 `.mmd`」。
- **Baseline 输出**：接近原版 Nougat 的「扁平 Markdown」。
- **Finetuned 输出**：在训练目标的引导下，更趋近于 GT 的结构化形式（虽然不一定完美）。

这也是为什么我们在评测时必须先做**统一清洗与占位符归一化**，再用 NED / 公式匹配等指标比较两者。

---

## 三、评测方案：从脚本到指标

### 3.1 评测数据与脚本

- **评测数据**：`ocr_data/test.jsonl`（共 3491 条左右，实际有效 3490 条），与：
  - Baseline 结果：`ocr_data/test_results_base.json`
  - Finetuned 结果：`ocr_data/test_results_finetuned_v2.json`
- **评测脚本**：`scripts/evaluate.py`
  - 单文件评估：
    ```bash
    python scripts/evaluate.py --results ocr_data/test_results_base.json
    ```
  - Baseline vs Finetuned 双轨对比：
    ```bash
    python scripts/evaluate.py \
      --base ocr_data/test_results_base.json \
      --finetuned ocr_data/test_results_finetuned_v2.json \
      --output ocr_data/eval_summary.json
    ```

脚本会从结果 JSON 中读取：

- `predictions`: 模型生成的文本。
- `ground_truths`: test.jsonl 中的 markdown，经 tokenizer 解码后，与原始 `.mmd` 一致。

然后为两组结果分别计算核心指标，并在 `eval_summary.json` 里给出整体统计与 delta。

### 3.2 统一清洗（normalize_for_eval）

为了让 Baseline 与 Finetuned 的输出在比较前尽量可比，评测脚本在计算任何指标前对 `pred`/`gt` 做统一清洗：

- **去空白 & 合并换行**
  - `strip()` 去掉首尾空白。
  - 用正则 `re.sub(r"\n+", "\n", text)` 将连续换行合并为一个。
- **全角 → 半角**（默认开启，可通过 `--no-clean-fullwidth` 关闭）
  - 将 Unicode 全角空格与全角 ASCII 映射到对应半角字符，减少纯粹标点风格差异带来的编辑距离噪声。
- **图片 / figure 占位符泛化**
  - 将以下模式统一替换为同一个占位符 `[IMAGE_TOKEN]`：
    - `[FIGURE:... ]... [END_FIGURE]` 整块。
    - `[FIGURE_COORDS](...)[END_FIGURE_COORDS]`。
    - 类似 `[IMAGE_xxx]` 的图片占位符。
  - 这样可以避免如 `[FIGURE:S3.F1]` vs `[FIGURE:S3.F7]` 的 ID 差异人为拉大编辑距离，使 NED 更关注文本与结构而非图像 ID。

实现上，清洗逻辑集中在 `normalize_for_eval(text: str)` 中，Baseline 与 Finetuned 共用完全相同的预处理。

### 3.3 指标定义

在清洗后的文本上，脚本计算三类核心指标：

1. **NED（Normalized Edit Distance）**
   - 对每条样本：
     \[
     \text{NED} = 1 - \frac{\text{LevenshteinDistance(pred, gt)}}{\max(|pred|, |gt|)}
     \]
     - 距离使用 `rapidfuzz.distance.Levenshtein.distance`（C++ 实现，语义与 `nltk.edit_distance` 保持一致，仅性能提升）。
     - 仅当 `len(pred) ≥ 4` 且 `len(gt) ≥ 4` 时加入统计，避免极端短文本干扰。
   - 报告的是所有样本 NED 的**均值**。

2. **BLEU**
   - 使用 `nltk.translate.bleu_score.sentence_bleu` + `SmoothingFunction().method1`，对清洗后的 `pred` / `gt` 逐样本计算句子级 BLEU，然后取均值。
   - 与原项目 `nougat/metrics.py` 中 BLEU 的实现一致，只是输入改为清洗后的文本。

3. **公式级 P / R / F1（近似 Formula EM）**
   - 在每条清洗后的文本中，利用正则提取：
     - 行内公式：`$...$` 或 `\(...\)`
     - 独立公式：`$$...$$` 或 `\[...\]`
   - 对于每条样本，得到：
     - `gt_forms`：GT 公式多重集。
     - `pred_forms`：预测公式多重集。
   - 在**全局维度**上做多重集匹配：
     - 对所有样本的 `gt_forms` 计总数 `total_gt`。
     - 对所有样本的 `pred_forms` 计总数 `total_pred`。
     - 对于每一个公式字符串 `f`，统计在 GT 和 Pred 中的出现次数 `c_gt, c_pred`，累计 `min(c_gt, c_pred)` 到 `total_match`。
   - 进而计算：
     \[
     P = \frac{total\_match}{total\_pred},\quad
     R = \frac{total\_match}{total\_gt},\quad
     F1 = \frac{2PR}{P+R}
     \]
   - 它刻画了「在所有公式 token 上，预测字符串与 GT 完全一致的比例」，可视作一种近似的 **公式 Exact Match/F1**。

### 3.4 性能注意点

- 由于每条样本的文本长度较长（上千字符），Levenshtein 距离计算本身不便宜。
- 初版使用 `nltk.edit_distance`，在 3k+ 样本上会非常慢；后来改为 `rapidfuzz.distance.Levenshtein`，在**不改变数学定义的前提下**显著加速。
- 脚本使用 `tqdm` 显示进度条，便于观察评测进展。

---

## 四、核心评测结果（`ocr_data/eval_summary.json`）

以下为在测试集上 Baseline 与 Finetuned 的整体结果（清洗后、全量统计）：

```json
{
  "base": {
    "n_samples": 3490,
    "ned": 0.6592147400101576,
    "bleu": 0.6518432768315856,
    "formula_precision": 0.5931354569308467,
    "formula_recall": 0.6199772918294822,
    "formula_f1": 0.6062594176045217
  },
  "finetuned": {
    "n_samples": 3490,
    "ned": 0.7375897023124373,
    "bleu": 0.6705446041091171,
    "formula_precision": 0.5986121483694299,
    "formula_recall": 0.5673208782458675,
    "formula_f1": 0.5825466162965879
  },
  "delta_finetuned_minus_base": {
    "ned": 0.0783749623022797,
    "bleu": 0.018701327277531488,
    "formula_precision": 0.005476691438583203,
    "formula_recall": -0.052656413583614725,
    "formula_f1": -0.02371280130793385
  }
}
```

可读版本整理如下（保留三位小数）：

| 指标                          | Baseline | Finetuned | Δ(Finetuned − Baseline) |
|-----------------------------|----------|-----------|--------------------------|
| **样本数** (`n_samples`)      | 3490     | 3490      | 0                        |
| **NED（均值）**              | 0.659    | 0.738     | **+0.078**               |
| **BLEU（均值）**             | 0.652    | 0.671     | **+0.019**               |
| **公式 Precision**          | 0.593    | 0.599     | +0.005                   |
| **公式 Recall**             | 0.620    | 0.567     | **−0.053**               |
| **公式 F1**                 | 0.606    | 0.583     | **−0.024**               |

直观解读：

- **整体文本/结构还原（NED）**：Finetuned 相比 Baseline **提升约 0.08**，说明在清洗后的字符级还原上，微调明显更接近结构化 `.mmd` GT。
- **自然语言流畅度（BLEU）**：Finetuned 有**小幅提升**（+0.019），说明对正文类文本的顺序/用词也有一定正向影响。
- **公式整体表现**：
  - Precision 略有提升（+0.005），说明 Finetuned 在公式上**更少“幻觉”或错误公式被识别为 GT 公式**。
  - Recall 有一定下降（−0.053），公式 F1 也略降（−0.024），表明在「覆盖 GT 公式」这一点上，Baseline 更「激进」而 Finetuned 更「保守」。

---

## 五、结果分析与可能原因

### 5.1 为什么整体 NED 提升显著？

**1）数据工程带来的 GT 质量提升 + SFT 对齐**

- 本项目通过 `split_utils/line_matcher.py` + `page_splitter.py` 等组件对分页做了**行级精确对齐 + 字符级边界精化**：
  - 行级：`PageSplitter` 要求 ordered 内容严格递增，unordered 内容独立去重，利用 matched set 判断 gap 是否可以填充。
  - 字符级：`CharacterSplitter` 在跨页行的边界处用字符串匹配决定真正的截断位置。
- 这样得到的 GT `.mmd` 页级文本相对 Meta 原始数据**更干净、更对齐页面**。
- Finetuned 在这样的数据上训练，自然会学到更稳定的分页行为和更好的长文本对齐；在评测时，相比 Baseline，有更高概率「在正确的页只生成这一页该有的东西」。

**2）结构化标签作为显式锚点**

- 微调的目标序列包含大量结构化标签（`[TITLE]`、`[SUBTITLE]`、`[TEXT]`、`[FIGURE]` 等），这些标签在内容上的位置相对稳定。
- 对于字符级 NED 来说，只要标签的位置与 GT 相近，就会显著改善编辑距离：
  - 即使 Baseline 也大致读对了纯文本，但若没有这些标签/分段符，其「结构」会与 GT 有系统性偏差。
  - Finetuned 得益于监督信号，输出形态和 GT 更一致，因此**在同样的 OCR 能力前提下，结构化标签额外贡献了一段 NED 的提升**。

**3）增加了对异常输出的约束**

- 推理配置中通过 `repetition_penalty`、`max_new_tokens` 等手段抑制了严重复读和超长输出。
- 这些约束对 Baseline / Finetuned 都生效，但 Finetuned 在训练阶段已经习惯在较合理的长度内收敛、遵循结构化模板，因而更能受益。

综合来看，「NED 明显提升」更像是**分页对齐 + 结构化标签 + SFT 对这些信号的吸收**共同作用的结果，而不单纯是「纯 OCR 识别字符更准」。

### 5.2 公式指标为何提升有限甚至略降？

在公式维度上，结果是「Precision 略升，Recall/F1 略降」，原因可能包括：

1. **公式数据仍主要来自 LaTeX 源码，几乎无额外「清洗/对齐」监督**：
   - GT 公式基本是从 LaTeX 中抽取的原始数学表达式，经 minimal 正则/normalize 后写入 `.mmd`。
   - 相比正文/结构标签，公式更接近「直接复制 LaTeX」，本项目的数据工程在这个部分的增益有限。

2. **Finetuned 在复杂标签空间上「更保守」**
   - SFT 的目标序列充满 `[FORMULA]...`、`[TEXT]...`、`[FIGURE]...` 等标签，decoder 在有限长度内要同时完成：
     - 结构标签布局；
     - 文本正文；
     - 公式精确生成。
   - 在这种多目标权衡下，Finetuned 可能倾向于**优先「稳妥地」完成结构与正文**，对公式中的部分复杂细节（如稀有 Greek 字母、特殊对齐环境）选择略微「简化」，从而使：
     - 错误公式变少（Precision 略升）。
     - 但有一部分本该生成的复杂公式被「欠生成」或「结构差一点」，导致 Recall/F1 略降。

3. **Baseline 原本的公式能力已经很强**
   - 官方 Nougat 在大规模论文上预训练，本身就把「LaTeX 公式 → 渲染图像」这条映射学得不错。
   - SFT 数据集规模相对有限（几万级页），且主要关注结构化标签、分页与特定领域，而不是全面覆盖所有数学写法。
   - 因此，在公式上，SFT 更像是「在一个已经很不错的基础上进行轻微微调」，稍有 trade-off 属于合理现象。

4. **评测方法的保守性**
   - 我们当前的公式评测采用「精确字符串匹配的多重集 P/R/F1」，**任何空格/换行/等价写法差异都会被视为错误**。
   - 有可能存在「视觉/语义等价但字符串略有不同」的情况，在当前指标下被算作错，从而拉低 Finetuned 的 Recall/F1。

总体判断：**SFT 没有显著提升公式层面的整体 F1，但也保持在与 Baseline 相近的水平，且在 Precision 上略有优势**。这与项目本身把主要精力放在结构化标签与分页对齐上的目标是一致的。

### 5.3 对「OCR 能力 vs 分页精化」的直觉验证

你在问题中提到的直觉大致是：

> 「SFT 不太可能真正提升『OCR 能力』（字符级识别），但分页精化与结构标签可能让整体指标变好。」

结合上述结果，可以给出比较有根据的回答：

- **字符级 OCR 能力**：Baseline 本身已经很强，我们的 SFT 没有针对字符级错误（如 single character mis-recognition）做专门的纠错监督，因此不预期在「逐字符 OCR」上有大幅提升。
- **分页精化与结构化输出**：
  - 数据工程显著提升了 GT 的分页/结构质量，SFT 学到的是「如何在页级范围内，按照结构化 `.mmd` 的约束输出」。
  - NED 与 BLEU 的提升，很大一部分可以理解为「**结构更对、分页更准、异常输出更少**」，而不是简单地「多认对了多少个汉字或英文字符」。
- **公式/表格等结构**：
  - 这些部分目前仍高度依赖原始 LaTeX，处理较为粗糙，因此在 SFT 中没有获得同等强度的「结构+内容双重监督」。
  - 结果反映为：整体 F1 大致持平、略有 trade-off。

换句话说，本轮工作更像是：**在保持原有 OCR/公式能力的前提下，引入更强的结构意识与分页意识**，从而让输出更贴近实际业务需要的格式与结构。

---

## 六、可以写进简历的总结话术示例（可按需改写）

- **项目背景**：基于 Meta Nougat 的端到端学术文档 OCR，针对原版模型训练数据缺乏结构标签、分页对齐粗糙的问题，自建从 LaTeX 源码到「页级 `.mmd`/`.png`」的高质量数据流水线，并在新数据上完成结构化 SFT。

- **数据工程**：实现从 `.tex` → LaTeXML `.html` → 语义元素树 → 带 `[TITLE]`/`[FIGURE]`/`[FORMULA]` 等标签的结构化 `.mmd` 的全流程；设计基于倒排索引 + 双指针分页（`PageSplitter` + 字符级精化），将连续文档精确切分到每页；最终产出约 8 万对高质量页级训练样本。

- **模型与评测**：在官方 `nougat-base` 上进行结构化 SFT，并与预训练 Baseline 在同一测试集上对比。评测脚本对结果统一做文本清洗与图片占位符归一化，采用 **NED（1−归一化编辑距离）、句子级 BLEU 以及公式级 P/R/F1** 三类指标评估。

- **量化效果**（基于 3490 条测试样本的清洗后评测）：
  - 相比预训练 Baseline，微调模型在整体结构化输出任务上的 **NED 均值从 0.659 提升到 0.738（+0.078）**，句子级 **BLEU 从 0.652 提升到 0.671（+0.019）**。
  - 在保留原有数学公式解析能力的前提下，公式级 Precision 略有提升（0.593 → 0.599），Recall/F1 基本持平（轻微下降），整体达到与 Baseline 相近的公式解析质量。
  - 结合分页对齐与结构化标签的指标表现，证明了自建数据流水线与结构化 SFT 能显著提升模型对复杂论文版面的结构理解与页级解析质量。

你可以在此基础上根据目标岗位（偏数据工程 / 模型工程 / 应用落地）做适当裁剪与强调。若需要，我也可以根据具体 JD 再帮你压缩成 3–4 条英文 bullet。 

