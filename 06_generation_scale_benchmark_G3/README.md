# 06 G3：样本量 Scaling（纯金标准曲线 + R1/R2 银标扩增曲线）

> 依据：`实验设计/g3_matrix.json`、`实验设计/build_g3_datasets.py`、`实验设计/run_g3_scaling.py`，
> 以及 G1/G2 的实测结论（`../06_generation_scale_benchmark_G1_G2/results/g1_g2_comparison_report.md`）。
>
> 状态：代码与数据构建脚本已就绪，**G3 尚未开跑**。

## 1. 为什么拆成两段

G1/G2 的复盘给出三个约束：

1. **Level-2 的平台来自 G 类**：B/Y/C 的 F1 在 0.85–0.94，G 类只有 0.778–0.823；4B 用 r16 时
   G-F1 反而掉到 0.7919，说明 300 条/类上高容量 LoRA 会过拟合。
2. **4B 在 L1 已超过回打语料的 Teacher**（GPT-4o-mini：L1 91.0% / L2 87.5%），继续只加 R_1/R_2
   伪标签可能被 9%–12% 的噪声拖低上限，因此必须先有**零噪声的纯金标准曲线**作参照。
3. **编码器与 1.7B/4B 的差距极小**，G3 需要三者同台，才能判断“加数据”还是“加容量”更值得。

因此 G3 把样本量拆成两段，分别拟合：

| 段 | Level-1（W/R/N） | Level-2（B/Y/C/G） | 数据来源 |
|---|---|---|---|
| `pure_gold` | 50 / 100 / 150 / **200** 每类 | 75 / 150 / 225 / **300** 每类 | 金标准 train，嵌套抽样 |
| `gold_plus_silver` | 400 / 600 / 1000 每类 | 500 / 800 / 1200 每类 | 金标准全量 + 全量语料 `R_1` / `R_2` |

**边界规则（由 `build_g3_datasets.py::validate_config()` 强制校验）**

- Level-1：每类 **≤ 200** 只取金标准（共 600 条）；**> 200** 才允许从 `CN_DOCS_FINAL_complete.parquet`
  的 `R_1` 列补充银标。
- Level-2：每类 **≤ 300** 只取金标准（共 1200 条）；**> 300** 才允许从 `R_2` 列补充银标。
- 纯金标准档位若混入任何银标，构建直接 `AssertionError`；银标档位必须包含全部金标准。
- 相邻档位严格嵌套（50 ⊂ 100 ⊂ 150 ⊂ 200 ⊂ 400 ⊂ …），保证曲线只反映“新增数据”的边际效应。

## 2. 模型与配置

| 模型 id | 架构 | 精度 | 复用来源 |
|---|---|---|---|
| `chinese_roberta_large` | 中文 RoBERTa-wwm-ext-large（0.33B，全参） | bf16，lr 2e-5，max_len 512 | L1 200 / L2 300 复用 G2-1 |
| `qwen3_1.7b_bf16_r8` | Qwen3-1.7B + LoRA r8/α16，attn+mlp | bf16，lr 2e-4 | L1 200 / L2 300 复用 G1-3 |
| `qwen3_4b_bf16_r8` | Qwen3-4B + LoRA r8/α16，attn+mlp，GC | bf16，lr 2e-4 | 无（G2-3 是 r16，需重训） |

统一：3 epoch、有效 batch 4（编码器 8）、cosine + 5% warmup、seed 42。

**单元数**：3 模型 × 14 档 = **42** 个单元；其中 4 个金标准全量点复用，实际训练 **38** 次。

## 3. 数据构建

```bash
python 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py
# 或显式指定全量语料
python 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py \
  --corpus-parquet assets/shared/datasets/raw/CAPC-CG_V1.0/data/CN_DOCS_FINAL_complete.parquet
```

零泄漏措施：

- 银标池排除**所有**金标准 train + validation 文本，并整篇文档排除命中金标准的文档；
- 单文档最多 4 段（`max_para_per_doc`），避免长文档主导分布；
- 文本长度限制 20–768 字，去空白后去重；
- 构建后逐档断言与验证集零重叠；
- `g3_manifest.json` 记录每档 `total / gold_count / silver_count / sha256 / file`，
  训练时的 `data_sha` 会进 fingerprint，数据一改旧结果自动失效。

产物：`06_generation_scale_benchmark_G3/data/g3/<task>/train_per_class_<N>.jsonl` + `g3_manifest.json`（约 8 MB，入库）。

无全量 parquet 时的降级：L1 从 05 期 T/D 训练集取银标；L2 只构建纯金标准段（银标档位跳过并打印缺口）。

## 4. 运行流程

```bash
# 1) 构建数据 + 自检（本地可跑，不占 GPU）
python 06_generation_scale_benchmark_G3/scripts/prepare/build_g3_datasets.py
python 06_generation_scale_benchmark_G3/scripts/validation/selfcheck_g3.py

# 2) 计划预览（42 单元 / 38 训练 / 4 复用）
python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode plan

# 3) 冒烟：只挑 4B 的金标准最小档，产物隔离在 assets/06_generation_scale_benchmark_G3/smoke_tests/
python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode run --smoke \
  --only qwen3_4b --tier-mode gold_only --task task1_level1

# 4) 先跑纯金标准段（约 1 小时，回答“少样本谁学得最快”）
python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode all --tier-mode gold_only

# 5) 再跑银标扩增段（约 1.5–2 小时，回答“伪标签还能不能推高上限”）
python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode all --tier-mode silver_only

# 单模型 / 单任务调试
python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode all --only qwen3_4b --task task2_level2
```

后台运行：

```bash
nohup python 06_generation_scale_benchmark_G3/scripts/run_g3_scaling.py --mode all \
  > 06_generation_scale_benchmark_G3/logs/run_all_$(date +%m%d_%H%M).log 2>&1 &
```

**默认不保存每个档位的模型**（编码器全参会占 18 GB）。需要逐一留存时加 `--save-model`。

## 5. 复用规则

- 仅复用「金标准全量档位」（L1 200、L2 300），来源为 `06_generation_scale_benchmark_G1_G2/results/<exp>__<task>/summary.json`；
- 复用前逐项校验 `precision / lora_r / lora_alpha / target_scope / train_samples` 与 G3 配置一致，
  不一致则打印 `[REUSE-MISS]` 并改为重训；
- 复用结果写入 G3 的 `summary.json`，标记 `source=reuse:<exp>`，`predictions_csv` 仍指向 G1/G2 的逐条预测；
- 4B 不复用：G2-3 是 r16/α32，G3 统一 r8/α16。

## 6. 产物与判读

```
06_generation_scale_benchmark_G3/
├── configs/g3_matrix.json
├── data/g3/                      # 14 个训练档位 + manifest（入库）
├── scripts/                      # g3_common / build / run / selfcheck
├── results/                      # summary.json、predictions.csv、g3_scaling_report.md（入库）
└── logs/                         # 编排日志（入库）
assets/06_generation_scale_benchmark_G3/models/              # 仅 --save-model 时产生（不入库）
```

`g3_scaling_report.md` 按任务分两张表，同模型相邻档位给出 **ΔAcc** 与 **Δ瓶颈类 F1**
（L1 看 W-F1，L2 看 G-F1），可直接定位饱和拐点。

判读口径：

- 纯金标准段看**斜率**：谁在 50–100/类 就接近自身 200/类 的天花板，谁就是少样本首选；
- 银标段看**是否越过纯金标准上界**：若 1000/类 的 L1 仍低于 200/类 的 4B，说明 R_1 噪声已经压过数据红利；
- L2 重点看 **G-F1** 是否随样本量上升：G1/G2 的 0.79–0.82 若在 800–1200/类 仍不抬升，则瓶颈在标注边界而非样本量；
- 三种子方差未覆盖（G3 固定 seed 42），单点差异 <1.5pp 时不要过度解读。

## 7. 与 G4 的衔接

G3 完成后，把「L1 最优 (模型, 每类样本量)」与「L2 最优 (模型, 每类样本量)」写入
`06_generation_scale_benchmark_G4/configs/g4_matrix.json` 的 `best` 段，G4 才启动：

- G4-1：seed 13 / 42 / 2025，报告均值±标准差；
- G4-2：`enable_thinking=True` + 强模型生成的「受众/语气」短理由，检验能否突破 92%。

## 8. 风险与限制

1. **银标噪声**：R_1/R_2 是 Teacher 回打，L1 Teacher 91.0%、L2 87.5%，与 4B 的 L1 96.67% 同量级，
   因此银标段只用于“能否进一步推高”，不用于定义上限。
2. **文档级排除依赖文本匹配**：全量语料没有金标准文档 ID，只能通过命中文本反查文档再整篇剔除；
   若金标准段落被改写（非去空白差异），可能漏排除。manifest 记录排除文档数，便于审计。
3. **R 类样本稀缺**：全库 R_1=R 仅 12.3 万，L1 银标 1000/类（R 需 1000 条）仍充足，但更长曲线会先撞到 R 类上限。
4. **4B 显存**：沿用 G2 的 `bs4 + GC`，峰值约 13.7 GB；若换卡或换 batch，需重新确认 OOM 边界。
5. **不保存模型**：默认只留结果，复现需按 `data_sha + fingerprint` 重跑；需要部署时再挑档位 `--save-model`。
