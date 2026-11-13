#!/usr/bin/env bash
# Joint Training 上界实验 · 时间增量 + 领域增量
#
#   bash run_joint_all.sh check   # 环境自检 + 生成合并数据 + sha256 跨环境校验（不训练）
#   bash run_joint_all.sh run     # 4 个训练 + 4 次评估 + 打包结果
#
# 重要：本脚本只新增文件，不覆盖 incremental_common.py / continual_training.py /
#       incremental_evaluation.py / gpu_guard.py / run_full_matrix.py，
#       因此云端历史 E1–E4 产物的 code_version 指纹保持不变。
set -euo pipefail
cd "$(dirname "$0")"

T=05_incremental_learning_time
D=05_incremental_learning_domain
STEP="${1:-check}"

echo "===== 0. 环境自检 ====="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; import sys; print(' CUDA OK:', torch.cuda.get_device_name(0))" || exit 1
for X in "$T" "$D"; do
    n=$(ls "$X"/data/*/*_train.jsonl 2>/dev/null | wc -l)
    t=$(ls "$X"/data/*/*_test.jsonl 2>/dev/null | wc -l)
    echo "  $X: train=$n test=$t"
    test -f "$X/scripts/train/train_joint.py" || { echo "  [错误] 缺少 train_joint.py"; exit 1; }
    test "$n" -ge 5 || { echo "  [错误] 训练数据文件不足"; exit 1; }
done
NOW=$(python -c "import sys;sys.path.insert(0,'$D/scripts');import incremental_common as i;print(i.code_version())" 2>/dev/null || echo "?")
echo "  领域 code_version=$NOW（全新实例无历史产物，仅记录）"

echo "===== 0b. 基座模型（按 HF_HOME 定位）====="
HFHUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_DIR="$HFHUB/models--Qwen--Qwen2.5-1.5B-Instruct"
echo "  HF_HOME=${HF_HOME:-<默认>}  查找: $MODEL_DIR"
if [ ! -d "$MODEL_DIR/snapshots" ]; then
    echo "  [错误] 未找到 HF 缓存结构。若用 ModelScope 下到了普通目录，"
    echo "         需先转成 models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<hash>/ + refs/main"
    exit 1
fi
python - <<'PY'
from transformers import AutoTokenizer, AutoConfig
n = "Qwen/Qwen2.5-1.5B-Instruct"
t = AutoTokenizer.from_pretrained(n, local_files_only=True)
c = AutoConfig.from_pretrained(n, local_files_only=True)
print("  tokenizer/config OK  vocab=", len(t), " layers=", c.num_hidden_layers)
PY

echo
echo "===== 1. 生成合并训练集并校验 sha256 ====="
for X in "$T" "$D"; do
    python "$X/scripts/train/train_joint.py" --size 600 --prepare-only
    python "$X/scripts/train/train_joint.py" --size full --prepare-only
    for TAG in joint600 jointfull; do
        GOT=$(sha256sum "$X/data/joint/joint_${TAG}.jsonl" | awk '{print $1}')
        WANT=$(cat "$X/data/joint/${TAG}.sha256")
        if [ "$GOT" != "$WANT" ]; then
            echo "  [错误] $X $TAG 与本地基准不一致"
            echo "    got =$GOT"
            echo "    want=$WANT"
            echo "  数据文件有分叉，请先排查，不要开始训练。"
            exit 2
        fi
        echo "  OK  $X $TAG  $GOT"
    done
done

if [ "$STEP" = "check" ]; then
    echo
    echo "校验通过。确认上面的输出后执行： bash run_joint_all.sh run"
    exit 0
fi

echo
echo "===== 2. 训练（4 个串行，约 60–90 分钟）====="
mkdir -p "$T/logs" "$D/logs"
for SPEC in "time:full:jointfull" "time:600:joint600" "domain:full:jointfull" "domain:600:joint600"; do
    IFS=: read -r EXP SIZE TAG <<< "$SPEC"
    X="05_incremental_learning_${EXP}"
    echo "  [$(date +%H:%M:%S)] 训练 $EXP $SIZE"
    python -u "$X/scripts/train/train_joint.py" --size "$SIZE" --tag "$TAG" \
        > "$X/logs/joint_${TAG}.log" 2>&1 || { tail -n 20 "$X/logs/joint_${TAG}.log"; exit 3; }
    tail -n 2 "$X/logs/joint_${TAG}.log"
done

echo
echo "===== 3. 评估（测试全部窗口/领域 test 集）====="
mkdir -p "$T/results/jointfull" "$T/results/joint600" "$D/results/domain_incremental"
python "$T/scripts/evaluate/evaluate_task5_cumulative.py" --method cumulative \
    --model-path "assets/$T/models/joint/cumulative_task5_jointfull" \
    --output-dir "$T/results/jointfull"
python "$T/scripts/evaluate/evaluate_task5_cumulative.py" --method cumulative \
    --model-path "assets/$T/models/joint/cumulative_task5_joint600" \
    --output-dir "$T/results/joint600"
python "$D/scripts/evaluate/evaluate_task6_cumulative.py" --method cumulative \
    --model-path "assets/$D/models/joint/cumulative_task6_jointfull" --run-tag jointfull
python "$D/scripts/evaluate/evaluate_task6_cumulative.py" --method cumulative \
    --model-path "assets/$D/models/joint/cumulative_task6_joint600" --run-tag joint600

echo
echo "===== 4. 核对结果文件 ====="
ls -l "$T/results/jointfull"/*.json "$T/results/joint600"/*.json
ls -l "$D/results/domain_incremental/"*cumulative_task6_joint*.json

echo
echo "===== 5. 打包（结果 + 日志 + 合并数据 + 模型元数据，排除 checkpoint）====="
OUT="joint_results_$(date +%m%d_%H%M).tar.gz"
tar -czf "$OUT" \
    --exclude='checkpoint-*' \
    "$T/results/jointfull" "$T/results/joint600" \
    "$D/results/domain_incremental" \
    "$T/logs/joint_jointfull.log" "$T/logs/joint_joint600.log" \
    "$D/logs/joint_jointfull.log" "$D/logs/joint_joint600.log" \
    "$T/data/joint" "$D/data/joint" \
    "assets/$T/models/joint/cumulative_task5_jointfull" \
    "assets/$T/models/joint/cumulative_task5_joint600" \
    "assets/$D/models/joint/cumulative_task6_jointfull" \
    "assets/$D/models/joint/cumulative_task6_joint600"
ls -lh "$OUT"
sha256sum "$OUT"
echo "完成。把 $OUT 拉回本地即可。"
