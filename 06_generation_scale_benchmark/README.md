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
   0.33B 双向编码器（RoBERTa-large）与 1.7B LoRA 的差距有多大？

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
| 默认超参 | 3 epoch、bs=4、ga=4（有效 16）、lr=2e-4、cosine、warmup 5%、dropout 0.05、seed=42 |
| 硬件 | 仅云端 RTX 4090D（本地 1660 Ti 被 `gpu_guard` 拒绝） |
| 产物 | 指标/预测 → `results/<exp>__<task>/`（入库）；adapter/模型 → `assets/06_generation_scale_benchmark/`（不入库） |

## 3. 实验矩阵

| 编号 | 组 | 基座 | 精度 | LoRA 配比 | lr | 对比目的 |
|---|---|---|---|---|---|---|
| G1-1 | G1 | Qwen2.5-1.5B-Instruct | 4bit | r=16, α=32, q/k/v/o | 2e-4 | 基线锚点，复现 03 期 QLoRA 配置 |
| G1-2 | G1 | Qwen3-1.7B | 4bit | r=16, α=32, q/k/v/o | 2e-4 | 同配置换代，纯代际增益 |
| G1-3 | G1 | Qwen3-1.7B | bf16 | r=8, α=16, attn+mlp | 2e-4 | 去量化 + 补 MLP 后 r=8 是否够 |
| G1-4 | G1 | Qwen3-1.7B | bf16 | r=16, α=32, attn+mlp | 2e-4 | r=16 是否仍优于 r=8；G2-2 复用来源 |
| G1-5 | G1 | Qwen3-1.7B | bf16 | r=16, α=16, attn+mlp | 2e-4 | α/r=1 vs 2 的稳定性 |
| G2-1 | G2 | RoBERTa-large (0.33B) | bf16 | 全参微调，max_len=512 | 2e-5 | 编码器全参对照 |
| G2-2 | G2 | Qwen3-1.7B | bf16 | 同 G1-4（复用，不重训） | 2e-4 | 小尺寸轻量端基线 |
| G2-3 | G2 | Qwen3-4B | bf16 | r=16, α=32, attn+mlp | 2e-4 | 4B 能否逼近 8B |
| G2-4 | G2 | Qwen3-8B | bf16 | r=16, α=32, attn+mlp，GC | 1e-4 | 24G 单卡上限；bs=2 × ga=8 |

机器可读版本：`configs/g1_g2_matrix.json`（含 `comparisons` 与 `gates`，汇总脚本按它生成对照表）。

**与设计表的口径说明**

- **G2-2 不重复训练**：设计与 G1-4 完全相同，配置里用 `reuse` 指向 G1-4；汇总时自动生成
  “G2-2（来源 `reuse:G1-4`）”的别名行，避免同一实验跑两遍。
- **G1-1 是“同配置锚点”，不是 03 期结果的逐位复现**：03 期存在 `max_len` 256/768、epoch 1/3
  等多个历史批次；本目录统一 3 epoch、`max_len=768`，用于**同一批次内**的代际对比。
- **G2-1 是新增补齐项**：设计表要求编码器对照，但初稿脚本只覆盖 Qwen。这里补了
  `scripts/train/finetune_encoder.py`，数据划分、评估集、指标口径与 PEFT 组一致。
- **8B 未做量化**：G2-4 用 bf16 LoRA + 梯度检查点，靠 bs=2/ga=8 压到 24G；若云端仍 OOM，
  优先降到 `batch_size=1, grad_accum=16`，不要改成 4-bit（会与 G2-3 的精度口径不一致）。

## 4. 目录结构

```
06_generation_scale_benchmark/
├── README.md
├── configs/g1_g2_matrix.json           # 矩阵、对照项、门禁的唯一来源
├── scripts/
│   ├── benchmark_common.py             # 路径/编码/模型/评估/指纹/环境元信息
│   ├── gpu_guard.py                    # 本地 GPU 禁用守卫
│   ├── run_g1_g2_matrix.py             # 唯一编排入口（plan / run / summarize / all）
│   ├── train/
│   │   ├── run_peft_benchmark.py       # Qwen LoRA / QLoRA 单次训练 + 评估
│   │   └── finetune_encoder.py         # G2-1 RoBERTa-large 全参微调
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
assets/06_generation_scale_benchmark/models/<exp_id>__<task>/adapter/   # PEFT
assets/06_generation_scale_benchmark/models/<exp_id>__<task>/model/     # 编码器
```

## 5. 运行流程（云端 4090D）

```bash
# 0) 本地即可做：语法 + 纯逻辑自检（不加载模型、不占 GPU）
python -m py_compile 06_generation_scale_benchmark/scripts/**/*.py
python 06_generation_scale_benchmark/scripts/validation/selfcheck_units.py

# 1) 云端预检：数据规模、标签单 token、模型缓存、GPU 型号、目录可写
python 06_generation_scale_benchmark/scripts/prepare/preflight_check.py --strict

# 2) 计划预览：确认 18 个单元 / 16 次训练 / 2 个复用
python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode plan

# 3) 冒烟：每模型 2 条样本、1 步，产物隔离在 assets/06_generation_scale_benchmark/smoke_tests/
python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode run --smoke --only G1-4 G2-1

# 4) 正式全量（训练完自动汇总）
python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all

# 5) 断点续跑 / 局部重跑
python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all --state logs/<stamp>/summary_<stamp>.json
python 06_generation_scale_benchmark/scripts/run_g1_g2_matrix.py --mode all --only G2-3 G2-4 --task task1_level1

# 6) 只重算汇总（本地可跑，不碰 GPU）
python 06_generation_scale_benchmark/scripts/evaluate/summarize_results.py
```

PowerShell 下把 `python` 换成 `.\.venv\Scripts\python.exe`，其余参数一致。

**复用与防覆盖**：每个单元写入 `fingerprint`（超参 + 种子 + 冒烟标志 + 核心代码哈希）。
`--skip-existing` 只在指纹一致时跳过；不一致会直接报冲突，需显式 `--overwrite`（旧结果自动改名
`.bak-<时间戳>` 备份），避免“换了参数却复用了旧指标”。

**冒烟隔离**：`--smoke` 时结果固定写到 `assets/06_generation_scale_benchmark/smoke_tests/`，
不会污染 `results/`，也不会覆盖正式 adapter。

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
4. **8B 单卡余量小**：G2-4 峰值接近 24G，OOM 时按 §3 的降批方案处理，并记录实际生效参数。
5. **编码器对照的 tokenizer 差异**：RoBERTa 用 `text`（去前缀）而非 Qwen 的 chat template，
   两者输入长度与信息量不同，结论只用于“量级对照”，不做逐位公平性声明。
6. **G2-1 首次运行需联网**下载 `FacebookAI/roberta-large`（约 1.4G）；离线环境请提前 `huggingface-cli download`。
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

尚未执行：任何真实训练与评估（按 §7 与项目 GPU 规范，全部留待云端 4090D）。
正式跑完后需回填：各单元耗时/峰值显存、G1 胜出配比、G2 帕累托结论、G3 两个尺寸的选择依据。
