# assets/ —— 大文件集中区

本目录存放**不进入版本库**的大文件（约 14 GB），由 `.gitignore` 整体排除，
仅 `README.md` 本身入库。代码、使用数据、日志、结果表都在各实验目录内。

## 目录结构

```
assets/
├── 01_ml_baselines/artifacts/          早期基线模型（2.3 GB）
├── 03_qlora_finetuning/artifacts/      QLoRA 适配器（404 MB）
├── 04_error_analysis/artifacts/        错误分析模型与数据（235 MB）
├── 05_incremental_learning_time/
│   ├── models/
│   │   ├── stage1/                     T1 基线（sequential_ft）
│   │   ├── stage2/ … stage5/           各阶段：sequential_ft / ewc / replay / adapter
│   │   └── joint/                      Joint 上界与数据量对照（cumulative_*_jointfull|joint600）
│   └── continual_state/                EWC Fisher/anchor 快照（469 MB）
├── 05_incremental_learning_domain/
│   ├── models/stage1 … stage6, joint/  同上（6 个领域）
│   └── intermediate/
│       ├── domain/                     标题抽取与领域标注中间产物（79 MB）
│       └── domain_v14/                 v14 版候选池（73 MB）
├── 06_generation_scale_benchmark/
│   ├── models/<exp_id>__<task>/        G1/G2 适配器或编码器全参模型（不入库）
│   └── smoke_tests/                    冒烟产物（results / models 隔离，不入库）
├── shared/datasets/                    全量语料（raw 498 MB + processed_readable 6.8 GB）
├── archives/                           打包备份（6 个 tar.gz，725 MB）
└── README.md                           ← 唯一入库的文件
```

## 主库中保留了什么

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `05_*/data/temporal/` | T1–T5 训练/测试 jsonl（10 个文件） | ✅ |
| `05_*/data/domain_incremental/` | D1–D6 训练/测试 jsonl（12 个文件） | ✅ |
| `05_*/data/joint/` | Joint 合并训练集 + 抽样清单 + sha256 基准 | ✅ |
| `05_*/scripts/` | 全部代码 | ✅ |
| `05_*/results/` | 汇总 xlsx、逐条预测 csv、结果 json、图表 png | ✅ |
| `05_*/logs/` | 训练与编排日志 | ✅ |

## 代码如何定位

路径常量集中在各实验 `scripts/incremental_common.py`：

```python
ASSETS_ROOT = PROJECT_ROOT / "assets" / "05_incremental_learning_time"  # 或 _domain
DATA_ROOT   = INCREMENTAL_ROOT / "data" / "temporal"                    # 使用数据在主库
MODEL_ROOT  = ASSETS_ROOT / "models"                                    # 模型在 assets
SMOKE_ROOT  = ASSETS_ROOT / "smoke_tests"
```

模型按阶段分组，命名规则固定，由 `model_group()` 解析：

```python
model_group(method, task_number)      # cumulative → "joint"，其余 → f"stage{task_number}"
method_model_dir("ewc", 2)            # → assets/.../models/stage2/ewc_task2
adapter_dir(1)                        # → assets/.../models/stage1/qwen15b_*_task1_...
```

## 恢复方式

1. **只要代码与结果**：直接 clone，无需此目录。
2. **复现训练**：从 `archives/` 解压模型包，或按 `README.md` 的数据构建流程重新生成 `data/`
   （需要 `assets/shared/datasets/raw/` 与 HF 基座 `Qwen/Qwen2.5-1.5B-Instruct`）。
3. **只取某次产物**：从 `archives/*.tar.gz` 解出对应目录，包内已排除 `checkpoint-*`。
