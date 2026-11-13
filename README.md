# CAPC-CG Experiments

This project contains experiments for Chinese policy-signal classification:
Level 1 (N/R/W) and Level 2 (B/C/G/Y). Run scripts from the project root
because most scripts use paths relative to it.

## Project layout

| Directory | Purpose |
|---|---|
| `01_ml_baselines/` | Traditional ML and BERT training, evaluation, models, and results |
| `02_llm_inference/` | LLM API inference scripts, smoke tests, logs, and model-specific results |
| `03_qlora_finetuning/` | QLoRA training/evaluation scripts, configs, adapters, and results |
| `04_error_analysis/` | Error-analysis scripts, artifacts, analysis data, and results |
| `05_incremental_learning_time/` | Temporal incremental learning: code, T1–T5 data, results, logs |
| `05_incremental_learning_domain/` | Domain incremental learning: code, D1–D6 data, results, logs |
| `06_generation_scale_benchmark/` | G1/G2 generation anchor + parameter-scale ladder: matrix config, code, results, logs |
| `05_*/data/temporal/` | T1–T5 train/test jsonl (10 files, in git) |
| `05_*/data/domain_incremental/` | D1–D6 train/test jsonl (12 files, in git) |
| `05_*/data/joint/` | Joint-training merged sets, sampling manifests, sha256 baselines (in git) |
| `05_*/results/` | Summary xlsx, per-sample prediction csv, result json, figures (in git) |
| `05_*/logs/` | Training and orchestration logs (in git) |
| `assets/05_*/models/stage{N}/` | Per-stage models: sequential_ft / ewc / replay / adapter (not in git) |
| `assets/05_*/models/joint/` | Joint upper-bound and data-volume baselines (not in git) |
| `assets/05_*/continual_state/` | EWC Fisher / anchor snapshots (not in git) |
| `assets/05_*/intermediate/` | Domain-labeling intermediates: `domain/`, `domain_v14/` (not in git) |
| `assets/shared/datasets/` | Full corpora: raw and processed_readable (not in git) |
| `assets/{01,03,04}_*/artifacts/` | Earlier-phase models and analysis artifacts (not in git) |
| `assets/archives/` | Packaged backups (`*.tar.gz`, not in git) |
| `shared/common/` | Shared preprocessing and metric implementations |
| `shared/dependencies/` | Shared Python dependency manifest |
| `shared/reports/` | Project-wide reports and summaries |
| `shared/references/` | Research reference materials |
| `.venv/` | Local Python environment (not in git) |

**Split of responsibilities.** Source code, in-use datasets (`T1–T5`, `D1–D6`),
logs, and result tables stay inside each experiment folder and are versioned.
Everything large or regenerable — models, corpora, continual-learning state,
labeling intermediates, archives — lives under `assets/`, which `.gitignore`
excludes. The repository therefore clones to a few tens of MB.
See `assets/README.md` for the full tree and restore instructions.

Models are grouped by training stage so each stage is self-contained:

```
assets/05_incremental_learning_time/models/
├── stage1/   qwen15b_temporal_task1_...
├── stage2/   adapter_task2  ewc_task2  replay_task2  qwen15b_temporal_task2_...
├── stage3/ … stage5/
└── joint/    cumulative_task5_jointfull  cumulative_task5_joint600
```

Path resolution goes through `incremental_common.model_group()` /
`continual_training.method_model_dir()`, so no script hard-codes a stage path.

Within each experiment phase, scripts, results, and logs are grouped by role.
Use the existing subdirectories (`scripts/train/`, `scripts/evaluate/`,
`results/`, `logs/`) rather than creating new files in the project root.

## Error-analysis experiments

E1, E2-A, and E2-B training/evaluation scripts are under
`04_error_analysis/scripts/`. E2 adapters, checkpoints, and run metadata are
under `assets/04_error_analysis/e2_class_weighted/`; the shared `split_v2`
train/validation data is under `04_error_analysis/data/split_v2/`. E1/E2
comparison results are stored under `04_error_analysis/results/`.

## Temporal incremental learning

The five temporal tasks use the self-built JSONL files under
`05_incremental_learning_time/data/temporal/` (`text` and `label` fields), not the
original dataset's `messages` parquet format. The training scripts use the same
Level-1 N/R/W policy-signal prompt as the gold dataset and the best QLoRA
settings from the previous experiments: Epoch=2, Rank=16, Alpha=32, and
learning rate=2e-4.

Before a formal run, check all five datasets and tokenizer inputs, then run a
one-optimizer-step smoke test. The smoke-test adapter is saved in a temporary
directory under `assets/05_incremental_learning_time/smoke_tests/` and removed
afterward; it does not create or overwrite a formal task adapter.

```powershell
.\.venv\Scripts\python.exe 05_incremental_learning_time\scripts\tests\smoke_test_incremental.py --check-only
.\.venv\Scripts\python.exe 05_incremental_learning_time\scripts\tests\smoke_test_incremental.py --task 1
```

The original `train_task1.py`–`train_task5.py` files implement sequential
fine-tuning and are retained for reproducibility. Task1 and Task2 are already
complete. Do not run their legacy Task3–Task5 entries for the current EWC /
Replay / Adapter comparison.

| Legacy sequential stage | Train | Cumulative evaluation | Adapter directory |
|---|---|---|---|
| 1 | `scripts/train/train_task1.py` | `scripts/evaluate/evaluate_task1_cumulative.py` | `qwen15b_temporal_task1_e2_r16_a32_lr2e-4` |
| 2 | `scripts/train/train_task2.py` | `scripts/evaluate/evaluate_task2_cumulative.py` | `qwen15b_temporal_task2_e2_r16_a32_lr2e-4` |
| 3 | `scripts/train/train_task3.py` | `scripts/evaluate/evaluate_task3_cumulative.py` | `qwen15b_temporal_task3_e2_r16_a32_lr2e-4` |
| 4 | `scripts/train/train_task4.py` | `scripts/evaluate/evaluate_task4_cumulative.py` | `qwen15b_temporal_task4_e2_r16_a32_lr2e-4` |
| 5 | `scripts/train/train_task5.py` | `scripts/evaluate/evaluate_task5_cumulative.py` | `qwen15b_temporal_task5_e2_r16_a32_lr2e-4` |

Adapters and Trainer checkpoints are stored under
`assets/05_incremental_learning_time/models/`; per-step training metrics are
written to `05_incremental_learning_time/logs/temporal/<adapter>/training_metrics.jsonl`;
cumulative JSON metrics and
prediction CSVs under `05_incremental_learning_time/results/temporal/`. Result
files use the method-specific names described below, such as
`temporal_sequential_ft_taskN_cumulative_test.json` and
`temporal_sequential_ft_taskN_cumulative_test_predictions.csv`.

Each cumulative evaluation runs the selected stage model against every seen
period's frozen test set: Task2 evaluates both T1 and T2; Task3 evaluates T1,
T2, and T3, and so on. The JSON/terminal output includes per-period accuracy,
Macro-F1, Cohen's Kappa, N/R/W precision/recall/F1, and confusion matrices.
It also reports `ACC` (mean accuracy over all seen test periods), `BWT`
(mean current accuracy minus accuracy immediately after first learning each
earlier period), and `Forgetting` (mean best recorded pre-final accuracy minus
current accuracy for earlier periods). BWT and Forgetting are `N/A` until the
same method has completed evaluations at all required earlier stages; Task1
has no prior task, so those two metrics are not defined there.

The evaluator supports `sequential_ft`, `ewc`, `lwf`, `replay`, `packnet`, and
`adapter`. Dedicated Task2-Task5 training scripts are available for EWC,
Replay, and task-specific Adapter:

```powershell
# First run the existing sequential Task2 model as the forgetting reference.
python 05_incremental_learning_time\scripts\evaluate\evaluate_task2_cumulative.py

# Then launch each new method independently; do not run train_task3.py for this plan.
python 05_incremental_learning_time\scripts\train\train_ewc.py --task 2
python 05_incremental_learning_time\scripts\train\train_replay.py --task 2
python 05_incremental_learning_time\scripts\train\train_adapter_cl.py --task 2

# Evaluate each resulting Task2 model on both frozen T1 and T2 test sets.
python 05_incremental_learning_time\scripts\evaluate\evaluate_task2_cumulative.py `
	--method ewc `
	--model-path assets\05_incremental_learning_time\models\ewc_task2
python 05_incremental_learning_time\scripts\evaluate\evaluate_task2_cumulative.py `
	--method replay `
	--model-path assets\05_incremental_learning_time\models\replay_task2
python 05_incremental_learning_time\scripts\evaluate\evaluate_task2_cumulative.py `
	--method adapter `
	--model-path assets\05_incremental_learning_time\models\adapter_task2
```

Run the Task2 method trainings one at a time. Do not start Task3 for any method
until all three Task2 cumulative results pass the forgetting gate against the
sequential Task2 reference.

The Task1 test result has already been registered as the same-method initial
baseline for EWC, Replay, and Adapter. The EWC/Replay/Adapter Task2 training
scripts start from the completed Task1 adapter; they do not consume the old
sequential Task2 adapter. The sequential Task2 evaluation is a separate
reference only. Each method gets independent history/result files named
`temporal_<method>_taskN_cumulative_test.json` and
`temporal_<method>_taskN_cumulative_test_predictions.csv`; the old sequential
Task1 result is accepted as history when evaluating sequential Task2.

EWC, Replay, and Adapter results must each pass the Task2 gate before their
Task3 runs. The legacy sequential `train_task3.py`-`train_task5.py` scripts are
not part of this comparison plan. `lwf` and `packnet` are accepted by the
evaluator only; no training implementations are currently included for them.

The original Hugging Face snapshot
(`Baron-Sun/CAPC-CG_V1.0`, revision
`d05965f169e6f7da33a25f667edf39add0339e2b`) is available locally at
`assets/shared/datasets/raw/CAPC-CG_V1.0/`. The Hugging Face cache copy has been left
intact.

## Generation anchor and scale benchmark (06)

`06_generation_scale_benchmark/` implements the G1 (Qwen2.5→Qwen3, QLoRA→bf16 LoRA,
rank/alpha ablation) and G2 (1.7B/4B/8B ladder plus RoBERTa-large encoder control)
designs. The matrix lives in `configs/g1_g2_matrix.json`; one entry point covers plan,
smoke, full run and summarization:

```powershell
.\.venv\Scripts\python.exe 06_generation_scale_benchmark\scripts\validation\selfcheck_units.py
.\.venv\Scripts\python.exe 06_generation_scale_benchmark\scripts\prepare\preflight_check.py --offline
.\.venv\Scripts\python.exe 06_generation_scale_benchmark\scripts\run_g1_g2_matrix.py --mode plan
```

Training and evaluation must run on the cloud RTX 4090D; `scripts/gpu_guard.py` rejects
local GPUs. G2-2 reuses G1-4 (same configuration) and each unit carries a configuration
fingerprint so `--skip-existing` cannot silently reuse stale results. See the folder
README for the decision gates into G3.

## Running experiments

Run commands in PowerShell from the project root. Examples:

```powershell
.\.venv\Scripts\python.exe 01_ml_baselines\scripts\train\baseline_ml.py
.\.venv\Scripts\python.exe 01_ml_baselines\scripts\train\finetune_bert.py --task task1_level1
.\.venv\Scripts\python.exe 03_qlora_finetuning\scripts\train\train_level1_qlora.py
.\.venv\Scripts\python.exe 04_error_analysis\scripts\train\train_e2_b.py
```

LLM inference entry points and utilities are under
`02_llm_inference/scripts/`; they may call external model APIs and incur usage
costs. Many files named `test_*.py` are experiment runners rather than pytest
unit tests. Inspect scripts under `smoke_tests/` or `tests/` before running:
some may call an API, load a model, or perform a short training smoke test.

## Dependencies

The shared dependency manifest is
`shared/dependencies/requirements.txt`. It records the versions from the
project's current environment. The manifest selects the CUDA 12.4 PyTorch
wheel; installing the full environment requires a compatible NVIDIA driver and
GPU. For a CPU-only environment, replace the PyTorch entry with the appropriate
CPU wheel before installing.
