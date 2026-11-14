# 06 G4：稳定性复现与推理增强（占位）

> 状态：**未开始设计，暂不放代码**。等 G3 跑完、确定「最优 (模型, 每类样本量)」并收到
> G4 的设计参考后，再补 `configs/` 与 `scripts/`，避免半成品。

## 目录命名约定

与 `06_generation_scale_benchmark_G1_G2`、`06_generation_scale_benchmark_G3` 保持一致：

```
06_generation_scale_benchmark_G1_G2/   # 已完成
06_generation_scale_benchmark_G3/      # 代码与数据已就绪，待训练
06_generation_scale_benchmark_G4/      # 本目录，仅占位
```

未来补齐时沿用同一结构：

```
06_generation_scale_benchmark_G4/
├── README.md
├── configs/g4_matrix.json
├── data/                 # 理由增强数据（G4-2）
├── scripts/              # 复用 G1_G2 的公共实现
├── results/
└── logs/
assets/06_generation_scale_benchmark_G4/   # 模型/中间产物（不入库）
```

## 预留的实验设计（来自实验设计表）

| 编号 | 内容 | 依赖 |
|---|---|---|
| G4-1 | 对 G3 最优的 (模型, 每类样本量) 跑 seed 13 / 42 / 2025，报均值±标准差 | G3 结果与最优档位 |
| G4-2 | `thinking=True` + 强模型生成的「受众/语气」短理由，检验能否突破 92% | 理由生成方案 + 生成式评估改造 |

G4-2 的训练/评估与现有单 token 打分不同：需要在标签前生成理由，并在评估时解析生成的标签，
属于生成式流程，需单独设计（不能直接复用 G1–G3 的 `evaluate_label_scores`）。

## 启动前需要确认的事

1. G3 报告（`06_generation_scale_benchmark_G3/results/g3_scaling_report.md`）给出的 L1/L2 最优档位；
2. G4-2 的理由来源（强模型名称、API、成本上限）与标注格式；
3. 是否接受生成式评估带来的额外推理成本与时延。
