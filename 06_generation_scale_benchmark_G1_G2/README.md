# 06 期：代际锚定（G1）与参数规模阶梯（G2）

> 依据：`实验设计/实验设计表.xlsx` 的 G1/G2 两段，以及 `实验设计/run_qwen3_peft_benchmark.py`、
> `实验设计/run_g1_g2_matrix.bash` 两个初稿脚本。本目录是其工程化落地：统一入口、矩阵配置、
> 指纹复用、运行前预检、结果汇总与门禁说明。
>
> 状态：**准备完成，待云端 RTX 4090D 正式运行**（本地 1660 Ti 只做静态检查与汇总）。

---

## 1. 实验定位

G1/G2 回答两个衔接问题：

1. **G1（代际锚定）**：`Qwen2.5-1.5B` → `Qwen3-1.7B` 的代际收益有多少？去掉 4-bit 量化、补齐 MLP、
   调整 rank / alpha 之后，最优 LoRA 配比是否改变？
2. **G2（参数规模阶梯）**：在胜出配比下，`1.7B → 4B → 8B` 的效果-成本曲线如何？4B 能否逼近 8B，
   0.33B 中文双向编码器（chinese-roberta-wwm-ext-large）与 1.7B LoRA 的差距有多大？

结论直接决定 **G3 样本量曲线用哪两个尺寸、哪套 LoRA 配比**，因此 G1/G2 不能跳步。

## 2. 一页速查

| 项 | 值 |
|---|---|
| 矩阵规模 | 9 条 run × 2 个任务 = **18 个单元**；G2-2 复用 G1-4，实际训练 **16 个单元** |
| 任务 | `task1_level1`（W/R/N，train 600 / val 300）、`task2_level2`（B/Y/C/G，train 1200 / val 400） |
| 数据划分 | 金标准 gold 划分（200/类、300/类），与 01–05 期同源同口径 |
| 监督方式 | 单 token：`prompt` 全部 `-100`，仅标签 token 计算 loss；评估为标签 token 受限分类 |
| 截断 | 训练与评估共用 `truncate_head_tail()`：超长保留头 50% + 尾 50%，`max_length=768`（编码器 512） |
| Qwen3 模式 | `enable_thinking=False`，避免 `<think>` 段破坏单 token 监督 |
| 默认超参 | 3 epoch、bs=4、ga=1（有效 4，与 03 期基线对齐）、lr=2e-4、cosine、warmup 5%、dropout 0.05、seed=42 |
| 硬件 | 仅云端 RTX 4090D（本地 1660 Ti 被 `gpu_guard` 拒绝） |
| 产物 | 指标/预测 → `results/<exp>__<task>/`（入库）；adapter/模型 → `assets/06_generation_scale_benchmark_G1_G2/`（不入库） |

## 3. 实验矩阵

| 编号 | 组 | 基座 | 精度 | LoRA 配比 | lr | 对比目的 |
|---|---|---|---|---|---|---|
| G1-1 | G1 | Qwen2.5-1.5B-Instruct | 4bit | r=16, α=32, q/k/v/o | 2e-4 | 基线锚点，复现 03 期 QLoRA 配置 |
| G1-2 | G1 | Qwen3-1.7B | 4bit | r=16, α=32, q/k/v/o | 2e-4 | 同配置换代，纯代际增益 |
| G1-3 | G1 | Qwen3-1.7B | bf16 | r=8, α=16, attn+mlp | 2e-4 | 去量化 + 补 MLP 后 r=8 是否够 |
| G1-4 | G1 | Qwen3-1.7B | bf16 | r=16, α=32, attn+mlp | 2e-4 | r=16 是否仍优于 r=8；G2-2 复用来源 |
| G1-5 | G1 | Qwen3-1.7B | bf16 | r=16, α=16, attn+mlp | 2e-4 | α/r=1 vs 2 的稳定性 |
| G2-1 | G2 | 中文 RoBERTa-wwm-ext-large (0.33B) | bf16 | 全参微调，max_len=512 | 2e-5 | 中文编码器全参对照 |
| G2-2 | G2 | Qwen3-1.7B | bf16 | 同 G1-4（复用，不重训） | 2e-4 | 小尺寸轻量端基线 |
| G2-3 | G2 | Qwen3-4B | bf16 | r=16, α=32, attn+mlp，GC | 2e-4 | 4B 能否逼近 8B；bs=4 × ga=1 |
| G2-4 | G2 | Qwen3-8B | bf16 | r=16, α=32, attn+mlp，GC | 1e-4 | 24G 单卡上限；bs=2 × ga=2，评估 bs=4 |

机器可读版本：`configs/g1_g2_matrix.json`（含 `comparisons` 与 `gates`，汇总脚本按它生成对照表）。

**与设计表的口径说明**

- **G2-2 不重复训练**：设计与 G1-4 完全相同，配置里用 `reuse` 指向 G1-4；汇总时自动生成
  “G2-2（来源 `reuse:G1-4`）”的别名行，避免同一实验跑两遍。
- **G1-1 是“同配置锚点”，不是 03 期结果的逐位复现**：03 期存在 `max_len` 256/768、epoch 1/3
  等多个历史批次；本目录统一 3 epoch、`max_len=768`，用于**同一批次内**的代际对比。
- **G2-1 是新增补齐项**：设计表要求编码器对照，但初稿脚本只覆盖 Qwen。这里补了
  `scripts/train/finetune_encoder.py`，数据划分、评估集、指标口径与 PEFT 组一致。
- **8B 未做量化**：G2-4 用 bf16 LoRA + 梯度检查点，bs=2 × ga=2（有效 4）压到约 19G；
  评估侧另把 `eval_batch_size` 降到 4，避免 logits 撑爆 24G。不要改成 4-bit（会与 G2-3 的精度口径不一致）。
- **4B 也需要梯度检查点**：G2-3 的 bs=4 在 24G 上无 GC 会 OOM，实测开启 GC 后峰值约 13.7G，
  有效 batch 仍为 4，不影响与 1.7B/8B 的对比口径。

**云端实测生效参数（与设计表的差异）**

| 项 | 设计/初值 | 实测生效 | 原因 |
|---|---|---|---|
| 有效 batch（Qwen 组） | bs4 × ga4 = 16 | **bs4 × ga1 = 4** | 16 时 L1 仅 114 步，Qwen2.5 QLoRA 跌到 0.38；回到 4 后与 03 期一致，L1 0.93+ |
| G2-3（4B） | 无 GC | **GC 开启** | bs4 无 GC 在 24G 上 OOM |
| G2-4（8B） | bs2 × ga8 | **bs2 × ga2，eval bs4** | 保持有效 batch 4；评估 logits 转 fp32 会 OOM |
| 模型加载 | HF id | 云端用绝对路径（ModelScope 本地目录） | 实例带宽优先；见 §5 的 `--config` 覆盖方式 |

## 4. 目录结构

```
06_generation_scale_benchmark_G1_G2/
├── README.md
├── configs/g1_g2_matrix.json           # 矩阵、对照项、门禁的唯一来源
├── scripts/
│   ├── benchmark_common.py             # 路径/编码/模型/评估/指纹/环境元信息
│   ├── gpu_guard.py                    # 本地 GPU 禁用守卫
│   ├── run_g1_g2_matrix.py             # 唯一编排入口（plan / run / summarize / all）
│   ├── train/
│   │   ├── run_peft_benchmark.py       # Qwen LoRA / QLoRA 单次训练 + 评估
│   │   └── finetune_encoder.py         # G2-1 中文 RoBERTa-wwm 全参微调
│   ├── evaluate/summarize_results.py   # CSV + Markdown 对比报告（纯标准库）
│   ├── prepare/preflight_check.py      # 数据/标签/模型/GPU/目录预检
│   └── validation/selfcheck_units.py   # 纯逻辑自检（不加载模型）
├── results/                            # summary.json、predictions.csv、对比表（入库）
└── logs/                               # 每次编排的日志与台账（入库）
```

单个单元的运行产物：

```
results/<exp_id>__<task>/summary.json        # 指标 + 效率 + 环境 + 指纹 + 数据快照
results/<exp_id>__<task>/predictions.csv     # gold / pred / correct / text / prob_<label>
assets/06_generation_scale_benchmark_G1_G2/models/<exp_id>__<task>/adapter/   # PEFT
assets/06_generation_scale_benchmark_G1_G2/models/<exp_id>__<task>/model/     # 编码器
```

## 5. 运行流程（云端 4090D）

```bash
# 0) 本地即可做：语法 + 纯逻辑自检（不加载模型、不占 GPU）
python -m py_compile 06_generation_scale_benchmark_G1_G2/scripts/**/*.py
python 06_generation_scale_benchmark_G1_G2/scripts/validation/selfcheck_units.py

# 1) 云端预检：数据规模、标签单 token、模型缓存、GPU 型号、目录可写
python 06_generation_scale_benchmark_G1_G2/scripts/prepare/preflight_check.py --strict

# 2) 计划预览：确认 18 个单元 / 16 次训练 / 2 个复用
python 06_generation_scale_benchmark_G1_G2/scripts/run_g1_g2_matrix.py --mode plan

# 3) 冒烟：每模型 2 条样本、1 步，产物隔离在 assets/06_generation_scale_benchmark_G1_G2/smoke_tests/
python 06_generation_scale_benchmark_G1_G2/scripts/run_g1_g2_matrix.py --mode run --smoke --only G1-4 G2-1

# 4) 正式全量（训练完自动汇总）
python 06_generation_scale_benchmark_G1_G2/scripts/run_g1_g2_matrix.py --mode all

# 5) 断点续跑 / 局部重跑
python 06_generation_scale_benchmark_G1_G2/scripts/run_g1_g2_matrix.py --mode all --state logs/<stamp>/summary_<stamp>.json
python 06_generation_scale_benchmark_G1_G2/scripts/run_g1_g2_matrix.py --mode all --only G2-3 G2-4 --task task1_level1

# 6) 只重算汇总（本地可跑，不碰 GPU）
python 06_generation_scale_benchmark_G1_G2/scripts/evaluate/summarize_results.py
```

PowerShell 下把 `python` 换成 `.\.venv\Scripts\python.exe`，其余参数一致。

**复用与防覆盖**：每个单元写入 `fingerprint`（超参 + 种子 + 冒烟标志 + 核心代码哈希）。
`--skip-existing` 只在指纹一致时跳过；不一致会直接报冲突，需显式 `--overwrite`（旧结果自动改名
`.bak-<时间戳>` 备份），避免“换了参数却复用了旧指标”。

**冒烟隔离**：`--smoke` 时结果固定写到 `assets/06_generation_scale_benchmark_G1_G2/smoke_tests/`，
不会污染 `results/`，也不会覆盖正式 adapter。

**云端模型路径覆盖**：如果实例把模型放在本地目录（例如 ModelScope 下载到
`/root/autodl-tmp/ms/`）而不是 HF 缓存，复制一份只改 `model` 字段的配置，用 `--config` 传入，
仓库里的主配置仍保留可移植的 HF id：

```bash
cp 06_generation_scale_benchmark_G1_G2/configs/g1_g2_matrix.json \
   06_generation_scale_benchmark_G1_G2/configs/g1_g2_matrix.local.json
# 将 9 条 run 的 model 改成实例上的绝对路径后：
python 06_generation_scale_benchmark_G1_G2/scripts/run_g1_g2_matrix.py \
  --mode all --config 06_generation_scale_benchmark_G1_G2/configs/g1_g2_matrix.local.json
```

`.local.json` 属于环境差异，不要提交；`summary.json` 的 `model` 字段会记录当次实际使用的路径。

## 6. 决策出口（门禁）

| 门禁 | 通过条件 | 出口决策 |
|---|---|---|
| G1 完成 | G1-1…G1-5 × L1/L2 均有结果，预检无 FAIL | 选定 **scope（attn / attn+mlp）、rank、alpha**；记录代际增益与量化代价 |
| G2 完成 | G2-1…G2-4 × L1/L2 均有结果（G2-2 可复用） | 选定 **主力尺寸与上限尺寸**（参考帕累托非支配点），确定 G3 的两个尺寸 |
| 进入 G3 | 上述两项完成，且 G1 胜出配比写回 `configs` | 才可构建 G3 样本量曲线（每类 200→1500 / 150→1200）与 G4 多种子 |

汇总报告的“关键对照”表已按 `configs/g1_g2_matrix.json` 的 `comparisons` 自动计算 ΔAcc / ΔMF1，
不需要手工抄数；帕累托表按 `训练峰值显存 ←→ Acc` 标注“非支配 / 被支配”。

## 7. 已知限制与风险

1. **单种子（42）**：G1/G2 只做配置筛选，最终结论的均值±标准差由 G4-1（seed 13/42/2025）补齐。
2. **4-bit vs bf16 的显存不可直接横比**：G1-1/G1-2 是 QLoRA 的显存口径，G1-3…G2-4 是 bf16 LoRA；
   报告里按 `precision` 分列，不把“省显存”当方法优势。
3. **长文本截断**：头尾各 50% 保留是折中方案，仍需在报告里统计 `train_truncated_count` /
   `eval_truncated_count`，若 L2 截断比例过高需单独说明。
4. **8B 单卡余量小**：G2-4 实测训练峰值 18.98G（bs2×ga2+GC），但评估阶段曾因 accelerate 把
   `[B,768,151936]` logits 转 fp32 而 OOM；现已解包模型并把评估 bs 降到 4（见 `fe44a23`）。
5. **编码器对照的 tokenizer 差异**：中文 RoBERTa 用 `text`（去前缀）而非 Qwen 的 chat template，
   两者输入长度与信息量不同，结论只用于“量级对照”，不做逐位公平性声明。
6. **G2-1 首次运行需联网**下载 `hfl/chinese-roberta-wwm-ext-large`（约 1.3G）；离线环境请提前 `huggingface-cli download`。
   该模型是中文全词掩码 BERT-wwm 架构（词表 21128，24 层/1024 隐层 ≈ 0.33B），不是英文 RoBERTa 词表。
7. **本机限制**：本地 1660 Ti 只允许 `py_compile`、`selfcheck_units.py`、`preflight_check.py`
   与 `summarize_results.py`；训练/评估入口有 `gpu_guard` 硬拦截。

## 8. 文件清单与状态

| 文件 | 作用 | 本地已验证 |
|---|---|---|
| `scripts/benchmark_common.py` | 公共工具与评估实现 | `py_compile` ✅ |
| `scripts/train/run_peft_benchmark.py` | G1/G2 Qwen 训练+评估 | `py_compile` ✅、守卫拦截 ✅ |
| `scripts/train/finetune_encoder.py` | G2-1 编码器对照 | `py_compile` ✅ |
| `scripts/run_g1_g2_matrix.py` | 编排 / 台账 / 复用 | plan ✅、复用 ✅、失败台账 ✅ |
| `scripts/evaluate/summarize_results.py` | CSV + Markdown 报告 | 合成数据 ✅ |
| `scripts/prepare/preflight_check.py` | 运行前预检 | 数据/目录/GPU/缓存 ✅ |
| `scripts/validation/selfcheck_units.py` | 纯逻辑自检 | 29/29 ✅ |

云端已完成 16 次训练 / 16 次评估（`results/` 已入库，含逐条预测）。

## 9. 实测结果（2026-11-14，RTX 4090D 24G）

完整表见 `results/g1_g2_comparison_report.md`，原始记录见各 `results/<exp>__<task>/summary.json`。

### 9.1 效果

| 编号 | L1 Acc | L1 Kappa | L2 Acc | L2 Kappa | 训练秒 L1/L2 | 训练峰值 GB |
|---|---|---|---|---|---|---|
| G1-1 Qwen2.5-1.5B 4bit r16 | 0.9333 | 0.900 | 0.8700 | 0.827 | 112 / 204 | 19.4 |
| G1-2 Qwen3-1.7B 4bit r16 | **0.9600** | 0.940 | 0.8725 | 0.830 | 128 / 229 | 19.3 |
| G1-3 Qwen3-1.7B bf16 r8 | 0.9567 | 0.935 | 0.8725 | 0.830 | 122 / 215 | 19.7 |
| G1-4 Qwen3-1.7B bf16 r16 | 0.9500 | 0.925 | 0.8625 | 0.817 | 121 / 217 | 19.8 |
| G1-5 Qwen3-1.7B bf16 r16α16 | 0.9433 | 0.915 | 0.8700 | 0.827 | 121 / 216 | 19.8 |
| G2-1 中文 RoBERTa-large 全参 | 0.9433 | 0.915 | **0.8750** | 0.833 | 31 / 61 | 8.4 |
| G2-3 Qwen3-4B bf16 r16 | **0.9667** | **0.950** | 0.8650 | 0.820 | 345 / 613 | 13.7 |
| G2-4 Qwen3-8B bf16 r16 | 0.9433 | 0.915 | 0.8675 | 0.823 | 481 / 870 | 19.0 |

### 9.2 关键对照（Δ = 处理组 − 基线，L1 / L2）

- 代际（G1-1→G1-2）：**+2.67pp / +0.25pp**，同配置下 Qwen3 只在 L1 有明显收益。
- 去量化 + 补 MLP（G1-2→G1-4）：**−1.00pp / −1.00pp**，4-bit 量化未造成可测损失，方向甚至略偏向 QLoRA。
- 秩（r8→r16）：**−0.67pp / −1.00pp**，r8 不劣于 r16。
- 缩放比（α/r=1→2）：**+0.67pp / −0.75pp**，在噪声内。
- 编码器 vs 1.7B LoRA：**−0.67pp / +1.25pp**，0.33B 中文编码器与 1.7B LoRA 基本持平。
- 尺寸 1.7B→4B：**+1.67pp / +0.25pp**；4B→8B：**−2.34pp / +0.25pp**。
- 帕累托（L1）：编码器（8.4G）与 4B（13.7G）非支配，8B 与 1.7B 被支配；L2 只有编码器非支配。

### 9.3 解读与 G3 选择

1. **L1 的最优性价比是 4B**（0.9667，13.7G，345s）；1.7B r8（0.9567，19.7G）在准确率相近的前提下显存反而更高（bf16 全量权重）。
2. **8B 没有收益**：L1 比 4B 低 2.3pp、L2 仅高 0.25pp，训练时间是 4B 的 1.4–1.6 倍，帕累托被支配。
3. **L2 存在明显平台**：0.8625–0.875 横跨 0.33B–8B 六个数量级的容量差异，差异 ≤1.25pp（400 条里约 5 条），
   单种子下不能解读为真实容量差异。L2 的瓶颈更可能在**标注一致性/样本量**，这正是 G3 样本曲线要回答的。
4. **配比建议**：G3 用 `bf16 + attn_mlp + r8/α16`（r8 不劣、显存与时间更低）；4-bit 只在需要跑 8B 时才有价值。
5. **G3 的两个尺寸**：建议 **1.7B 与 4B**（8B 不进入）；L1 关注 200→1500/类 的饱和拐点，L2 关注 150→1200/类
   在 0.865 平台上是否有抬升。

### 9.4 本轮相对设计稿的修正（已合入代码）

| 修正 | 原因 | 提交 |
|---|---|---|
| 评估改右填充 + 每条真实末位取 logits | 左填充未传 position_ids，RoPE 位置整体偏移；修正前训练集也只有 0.42，修正后 0.9333 | `a937bb4` |
| 训练后 `unwrap_model` + `empty_cache`，8B 评估 bs=4 | Trainer 的 accelerate 包装把全量 logits 转 fp32（3.7G），8B 评估 OOM | `fe44a23` |
| 有效 batch 统一为 4（ga=1；8B 为 bs2×ga2） | 有效 16 时 L1 仅 114 步，Qwen2.5 QLoRA 跌到 0.38 | config 固化 |
| G2-3 开梯度检查点 | 4B bs4 无 GC 在 24G 上 OOM | config 固化 |
| `find_snapshot()` 支持 HF_HOME/HF_HUB_CACHE | 实例若搬迁 HF 缓存会找不到数据 | `9ab32dd` |

> 注意：配置固化后，上一轮（ga=4 等）的旧结果指纹会失配；若在别处重跑，`--skip-existing`
> 会报冲突，需 `--overwrite` 或换目录，属预期行为。
