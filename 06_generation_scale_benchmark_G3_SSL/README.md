# 06 G3-SSL：Level-2 半监督去噪 Scaling 实验

用于剥离原 G3 阶段 `R_2` 弱 Teacher 银标（准确率 87.5%）带来的噪声干扰，在 **不依赖 `R_2` 伪标签** 的前提下，验证 Level-2（`B/Y/C/G`）的真实样本量 Scaling 曲线上限。

## 1. 核心机制

1. **零泄漏无标注候选池**：仅从 `CN_DOCS_FINAL_complete.parquet` 抽取 `R_1 == 'W'`（肯定型政策指令）且未命中任何金标准文档的段落（16,000 条），**不使用 `R_2` 标签**。
2. **异构双模型交叉打分**：用 300/类纯金标准分别训练 `chinese-roberta-wwm-ext-large`（0.33B 双向编码器）与 `Qwen3-4B (bf16 r=8)`（生成式大模型），对候选池联合打分，仅保留**双模型预测类别一致**且满足置信度门槛（单模型 $\ge 0.65$、均值 $\ge 0.75$）的样本。
3. **难度分层抽样（Anti-Matthew）**：按 **70% 高置信度（$\ge 0.90$）+ 30% 中置信度甜区（$0.75\text{–}0.90$）** 严格嵌套构建 `500 / 800 / 1200` 每类训练集。
4. **双策略单次连续评测**：
   - `ssl_co_mix`：300 金（weight=1.0）+ 伪标签（weight=0.5）加权训练 3 epochs；
   - `ssl_co_polish`：紧接在纯金标准上以 `0.2×lr` 精调 1 epoch 收尾，防止伪标签带偏金标准决策边界。

## 2. 云端运行指令（AutoDL `/root/CAPC-CG-Project-06` 根目录）

```bash
# 1) 静态自检
python 06_generation_scale_benchmark_G3_SSL/scripts/validation/selfcheck_g3_ssl.py

# 2) 构建半监督数据集（自动训练/缓存双 Teacher 并对 16,000 条无标注池打分，约 16 分钟）
nohup python 06_generation_scale_benchmark_G3_SSL/scripts/prepare/build_ssl_datasets.py \
  > 06_generation_scale_benchmark_G3_SSL/logs/build_ssl.log 2>&1 &

# 3) 数据构建完成后运行全量自检 + 计划预览
python 06_generation_scale_benchmark_G3_SSL/scripts/validation/selfcheck_g3_ssl.py
python 06_generation_scale_benchmark_G3_SSL/scripts/run_g3_ssl.py --mode plan

# 4) 运行半监督 Scaling 训练与自动汇总（RoBERTa + Qwen3-4B 各 3 档，约 1.5 小时）
nohup python 06_generation_scale_benchmark_G3_SSL/scripts/run_g3_ssl.py --mode all \
  > 06_generation_scale_benchmark_G3_SSL/logs/run_ssl.log 2>&1 &
```

完成后将 `06_generation_scale_benchmark_G3_SSL/results/` 打包即可（内含与 G3 原生 `R_2` 银标横向对比的 `g3_ssl_report.md`）。
