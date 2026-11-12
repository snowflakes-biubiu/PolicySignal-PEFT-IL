# 时间增量三种方法的训练计划

本文只覆盖 `05_incremental_learning_time` 的时间任务（T1–T5）。对比方法固定为 **EWC、Replay、Adapter**。不继续把顺序微调训到 Task 3–5，也不做 λ、记忆库大小或 LoRA rank 的搜索。

## 当前 Task 2 是什么

`scripts/train/train_task2.py` 只调用 `incremental_common.train_main(2)`。Task 2 及以后的模型加载是：

```python
model = PeftModel.from_pretrained(base_model, previous_adapter, is_trainable=True)
```

`is_trainable=True` 表示 Task 1 那一套 LoRA 权重会在 Task 2 的 600 条数据上继续被更新。保存出来的仍是**一个** `peft_type: LORA`、`r: 16` 的 adapter，目录名是 `qwen15b_temporal_task2_e2_r16_a32_lr2e-4`。`training_metadata.json` 里的 `parent_adapter` 只记录它从哪个目录接着训，不是新增了一个冻结模块。

它容易被看成 Adapter，是因为参数文件本身就是 PEFT LoRA adapter。持续学习里的 Adapter 指另一件事：每个任务新建一套 LoRA，旧的冻结，测试时把见过的各套加在一起。当前代码里没有 `add_adapter`，没有冻结旧 LoRA，损失里也没有 Fisher 惩罚或回放样本。

因此已完成的两次训练是顺序微调：

| 阶段 | 状态 | 目录 |
|---|---|---|
| Task 1 | 已完成，300 step | `artifacts/models/qwen15b_temporal_task1_e2_r16_a32_lr2e-4` |
| Task 2 | 已完成，300 step，约 10641 秒 | `artifacts/models/qwen15b_temporal_task2_e2_r16_a32_lr2e-4` |
| Task 3–5 | 未开始 | 无 |

不要再运行 `scripts/train/train_task3.py`。那个脚本会把顺序微调继续做下去。已有的 Task 1、Task 2 权重保留，Task 2 的权重不能当作 EWC、Replay 或 Adapter 的 Task 2。

Task 1 的累计测试已经完成：T1 Accuracy 0.743，结果在 `results/temporal/temporal_task1_cumulative_test.json`。Task 2 还没有累计测试。

## 脚本现状与执行门禁

时间数据校验与标签监督保持不变：每期 train 600 / test 300，N/R/W 均衡，年份落在该期内，文档和文本不跨任务、不跨 train/test；监督只在最后一个标签 token 上计算损失。超参为 Epoch=2、Rank=16、Alpha=32、学习率 2e-4，目标模块为 `q_proj`、`k_proj`、`v_proj`、`o_proj`。

旧入口 `train_task1.py`–`train_task5.py` 仍然是顺序微调。Task1、Task2 已完成；不要再运行 `train_task3.py`–`train_task5.py`。新增方法入口为 `train_ewc.py`、`train_replay.py`、`train_adapter_cl.py`，通过 `--task 2` 开始独立方法实验，输出到 `artifacts/models/ewc_taskN`、`replay_taskN`、`adapter_taskN`，不改动 `qwen15b_temporal_task*`。

评测入口仍为 `evaluate_task1_cumulative.py`–`evaluate_task5_cumulative.py`。每个阶段模型都在 T1 到当前阶段的冻结测试集上评估，输出 ACC、每期 Accuracy/Macro-F1/Kappa、N/R/W 各类指标、混淆矩阵、BWT 和 Forgetting。Task1 的旧结果已复制为 `temporal_ewc_task1_cumulative_test.json`、`temporal_replay_task1_cumulative_test.json`、`temporal_adapter_task1_cumulative_test.json`。Adapter bundle 会在测试时同时激活清单中的全部 LoRA，不按测试期选择 adapter。

执行门禁：先对现有顺序微调 Task2 运行 `evaluate_task2_cumulative.py`，取得其 T1 准确率作为遗忘参照；随后运行三种新方法的 Task2 训练和累计测试。只有三种方法的 T1 准确率都高于顺序微调 Task2 的 T1 准确率，且 Task2 自身学起来，才进入新方法的 Task3。不要把已有顺序微调 Task2 adapter 作为任一新方法的 Task2 结果。

## 三种方法

三种方法都从已经训好的 Task 1 adapter 开始。Task 1 对它们是同一次训练：还没有旧参数需要保护，EWC 惩罚为 0，Replay 还没有记忆库，Adapter 也只有一套新建的 LoRA。不重训 Task 1。

**EWC。** 继续更新同一套 LoRA。每期训练结束后，只在 LoRA 参数上用当期训练集估计一次对角 Fisher，并保存参数快照。下一期的损失是交叉熵加上 `(λ/2) Σ F (θ − θ*)²`。λ 固定为 5000。产物是单个 adapter，现有评测可以直接加载。

**Replay。** 继续更新同一套 LoRA。记忆库按每个已完成任务、每个类别 20 条抽取，每期 60 条，抽出后固定。各期训练样本量是：

| 任务 | 新数据 | 记忆 | 合计 |
|---|---:|---:|---:|
| Task 2 | 600 | 60 | 660 |
| Task 3 | 600 | 120 | 720 |
| Task 4 | 600 | 180 | 780 |
| Task 5 | 600 | 240 | 840 |

不回放全部历史。产物同样是单个 adapter。

**Adapter。** Task 2 起每个新任务 `add_adapter` 一套新的 LoRA，rank、alpha、dropout 和目标模块与 Task 1 相同。旧 LoRA 的 `requires_grad` 设为 False，只在当期 600 条上训练新 LoRA。测试不提供年份或任务编号，将该阶段全部 LoRA 同时启用、增量相加。按测试期挑选对应 LoRA 会使遗忘接近 0，不能和 EWC、Replay 比较。未合并的多 adapter 目录要保留，供下一期继续冻结旧模块；评测读取的也是这个目录。

## 训练次数

正式新训练是 Task 2–5 共 4 个阶段、3 种方法，**12 次**。按 Task 2 顺序微调的 10641 秒估算，EWC 和 Adapter 每次约 3 小时，8 次约 24 小时。Replay 的步数约为顺序微调的 1.1–1.4 倍，4 次约 15 小时。合计约 39 小时，另加每期一次 Fisher 估计，该估计远短于一次训练。

不包含：重训 Task 1、顺序微调 Task 3–5、超参搜索、全历史回放。

## 执行顺序

1. **已完成：**将 Task1 测试结果复制为 EWC、Replay、Adapter 各自的 Task1 历史文件。校验过三份文件与原结果 SHA256 一致；评测只读取准确率，不重测 Task1。
2. **待执行：**运行 `evaluate_task2_cumulative.py` 默认的 `sequential_ft`，取得旧顺序微调 Task2 在 T1 上的参照准确率。该步骤只推理，不会训练或覆盖模型。
3. **已完成：**新增 `train_ewc.py`、`train_replay.py`、`train_adapter_cl.py`；累计评测支持指定方法和模型目录，并能同时激活 Adapter bundle 中的全部 LoRA。新方法输出分别写入 `artifacts/models/ewc_taskN`、`replay_taskN`、`adapter_taskN`，不改动原 `qwen15b_temporal_task*`。
4. **下一阶段：**依次训练 EWC Task2、Replay Task2、Adapter Task2，每种方法训练后立即运行对应的 Task2 累计评测。不要并行启动长训练。通过标准：每个方法的 T2 能学起来，且 T1 准确率高于顺序微调 Task2 的 T1 参照。若不满足，先检查 Fisher 是否进入损失、记忆样本是否参与 Replay、旧 LoRA 是否冻结，不推进 Task3。
5. **门禁通过后：**再按 EWC、Replay、Adapter 各自的 Task3、Task4、Task5 顺序推进。每个阶段训练后立即累计测试；Fisher 和记忆库在相应阶段训练结束时保存。不要运行旧顺序微调 `train_task3.py`–`train_task5.py`。

BWT 和 Forgetting 依赖每一期刚训完时的测试结果。中间阶段不能留到最后一起测。
