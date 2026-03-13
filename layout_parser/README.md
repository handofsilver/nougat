# layout_parser — DiT 版面检测（可选模块）

本目录为**可选**组件，提供基于 DiT + Cascade R-CNN 的页面版面检测能力。当前集成到主流水线的方式：

- `split_htmls_to_pages.py --layout-parser` 启用时，在分页完成后**仅对首页**做版面检测与修正
- `layout_correction.py` 将检测结果与首页 MMD 语义标签比对，对部分标签做自动修正

**不启用 `--layout-parser` 时，主流水线无需安装本模块即可正常运行。**

---

## 确认：只处理首页

**是的，当前逻辑只处理首页。**

- **判断条件**（`split_htmls_to_pages.py`）：`0 in recognized_indices` —— 即分页结果里「第 0 页」（PDF 第 1 页）在「被识别出的页」列表中时，才进入 layout_parser 分支。
- **操作对象**：固定为**分页产出中的第 1 页文件**：
  - **`out/paper_id/01.png`**：由 PDF 第 1 页渲染得到的图像（用于 DiT 检测）
  - **`out/paper_id/01.mmd`**：分页算法分配给第 1 页的 MMD 内容（被修正的目标）
- 不针对整篇 PDF 或整篇 MMD，也不处理第 2 页及以后；首页 = 分页结果里的第 1 页（01.mmd + 01.png）。

---

## 在流水线中的位置

```
阶段 1 - split_htmls_to_pages:
  步骤 1-5: HTML → MMD → 分页 → 渲染（得到 01.mmd, 01.png, 02.mmd, ...）
  步骤 6（可选）: layout_parser 首页修正
    ├── 仅当 0 in recognized_indices 时执行
    ├── 读取 01.png → DiT 检测 → 20 类 bbox + 置信度
    ├── 映射检测标签到 MMD 语义标签（见下）
    └── 与 01.mmd 的标签比对，对指定类别做修正（当前仅 TITLE、AUTHOR）
```

---

## 检测类别（20 类）与代码对应关系

| 来源 | 说明 |
|------|------|
| **定义文件** | `layout_parser/cls_idx2name.json`：key 为 `"1"`～`"20"`，每项 `name` 为中文类名（如 `"文章标题"`、`"作者"`）。 |
| **加载** | `layout_parser_api.py`：读取 `cls_idx2name.json`，构造 `idx2cls = {idx: value['name'] for idx, value in idx2cls.items()}` 传入 `LayoutParser`。 |
| **预测时使用** | `module_layout_parser.py` 的 `parse_batch_pdf_image()`：`class_name = self.idx2cls[str(pred_classes[i].item()+1)]`。模型输出类别为 0-based，+1 后对应 json 的 `"1"`～`"20"`。 |

20 类中文名为：  
文章标题、作者、正文、子标题、表格、表格标题、图片、图片标题、注释、公式、算法、参考文献、片段、表格注释、目录、出处、其他、副标题、日期编号、公式推导。

---

## 参与修正的类别与当前行为

| 位置 | 说明 |
|------|------|
| **视觉标签 → MMD 标签映射** | `nougat/dataset/layout_correction.py` 中的 `VISUAL_LABEL_TO_MMD_TAG` 字典：将 13 种检测中文名映射到 MMD 标签（如 文章标题→TITLE、作者→AUTHOR、副标题→ABSTRACT、正文→TEXT、子标题→SECTION、公式→EQUATION、图片→FIGURE、图片标题/表格标题→CAPTION、表格→TABLE、注释→FOOTNOTE、算法→ALGORITHM、参考文献→BIBLIOGRAPHY）。 |
| **实际执行修正的类别** | **当前仅两类**：**TITLE**、**AUTHOR**（由 `CORRECTABLE_TAGS` 控制）。逻辑在 `correct_first_page()` 中：若视觉检测到 TITLE 而 MMD 没有 [TITLE]，则将首页第一个 [TEXT] 改为 [TITLE]；若视觉检测到 AUTHOR 而 MMD 没有 [AUTHOR]，则在已存在 [TITLE] 的前提下将紧随其后的第一个 [TEXT] 改为 [AUTHOR]。其他在 `VISUAL_LABEL_TO_MMD_TAG` 中的类别只参与视觉序列构造与日志，**不会触发对 MMD 的改写**。 |
| **可配置** | 修正类别由 `layout_correction.CORRECTABLE_TAGS` 控制（默认 `{"TITLE", "AUTHOR"}`）。扩展时可在该集合中加入如 `"ABSTRACT"`，并在 `correct_first_page()` 中已有对应分支。 |

---

## 启用时的日志

当使用 `split_htmls_to_pages.py --layout-parser` 时，**所有** layout_correction 的详细日志**只写入文件，不输出到控制台**。

- **日志目录**：在**数据根目录**下创建 `layout_correction_logs/`（数据根 = `args.out` 的父目录，即与 `out/`、`html/`、`src/` 同级）。
- **日志文件**：每篇论文一个文件，命名为 **`<paper_id>.log`**，例如 `2308.13418.log`。
- **控制台**：仅在该篇首页被实际修正时打印一行提示，例如  
  `[2308.13418] First page corrected by layout_parser (log: /path/to/layout_correction_logs/2308.13418.log)`。

日志文件内容（统一前缀 `[layout_correction]`）：

| 内容 | 说明 |
|------|------|
| **目标文件** | `applied to file: <out/paper_id/01.mmd>`（仅在有修正时出现）。 |
| **修改前/后标签** | `MMD tags before:` / `MMD tags after:` 列出该页语义标签序列。 |
| **每条修改** | `change N (TITLE): [TEXT] -> [TITLE]` 及该行的 **before** / **after** 内容（过长截断到约 80 字符）。 |
| **汇总** | `=== end (N change(s))`。 |

未发生修正时，该论文的 `.log` 文件中仍会有一行视觉标签与 MMD 标签的比对，不会出现 `applied to file` 及后续 before/after。

---

## 依赖与安装

本模块依赖 detectron2 和 DiT，与主仓库的 `pip install -e ".[dataset]"` 环境独立。建议使用单独的 conda 环境或在同一环境中额外安装：

```bash
# detectron2（需与 PyTorch/CUDA 版本匹配）
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# DiT 相关配置已在 sjt_configs/ 和 ditod/ 中，无需额外 pip 安装
# 需自行准备模型权重文件（cascade_dit_base 或 cascade_dit_large）
```

**模型权重路径**：默认使用本目录下的 `model_final.pth`；可通过环境变量 **`LAYOUT_PARSER_MODEL_PATH`** 覆盖为绝对路径或其他路径。

---

## 目录结构

```
layout_parser/
├── module_layout_parser.py      ← 检测主逻辑（DiT + 批量预测 + IoU 去重；类别名来自 idx2cls）
├── post_detect_reshape.py       ← 检测后处理：裁剪、截断行移除
├── post_detect_sort_target.py   ← 按阅读顺序排序检测结果
├── layout_parser_api.py         ← 上层 API；加载 cls_idx2name.json 构造 idx2cls
├── reader_writer.py             ← PDF 渲染与 I/O
├── visualize_label.py           ← 可视化工具
├── cls_idx2name.json            ← 20 类编号 → 中文名（及配色）定义
├── sjt_configs/                 ← Detectron2 / DiT 配置
└── ditod/                       ← DiT backbone 实现
```

---

## 维护方式

作为仓库内普通子目录维护（非 submodule），与 `nougat/`、`docs/` 同仓库提交。如后续独立为单独仓库，可改为 submodule 引用。

更多设计背景见 [docs/data_engineering_design.md](../docs/data_engineering_design.md)。
