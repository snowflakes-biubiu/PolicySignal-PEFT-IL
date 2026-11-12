"""CAPC-CG 基线2：BERT-base-Chinese 监督微调 (开源模型标杆).

对照论文 Appendix F 中 BERT-base-Chinese 的开源结果：
  Level-1 (R/W/N): Kappa=0.817  M-F1=0.846
  Level-2 (B/Y/C/G): Kappa=0.800  Acc=0.850

注意：
  - 在【发布的】 2500 条划分上训练+验证；论文 exact 数字基于其未公开的内部
    6000/1901 划分，因此这里为"同源方法的可复现近似"，用于验证方法成立、
    并确认 BERT >> 传统 ML 的排序。
  - GTX 1660 Ti (6GB)：用 fp16 + batch=16 + max_len=128 完全够用。
  - 运行：<venv>/python finetune_bert.py --task task1_level1
          <venv>/python finetune_bert.py --task task2_level2
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "shared" / "common"))

import preprocess
import metrics as M

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
)
from sklearn.preprocessing import LabelEncoder

RESULTS_MD = PROJECT_ROOT / "01_ml_baselines" / "results" / "results.md"


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def compute_metrics(labels):
    def _fn(eval_pred):
        logits, y_true = eval_pred
        y_pred = logits.argmax(axis=-1)
        rep = M.evaluate([labels[i] for i in y_true.tolist()],
                         [labels[i] for i in y_pred.tolist()], labels)
        return {
            "accuracy": rep["accuracy"],
            "cohen_kappa": rep["cohen_kappa"],
            "macro_f1": rep["macro_f1"],
        }
    return _fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(preprocess.TASK_LABELS), default="task1_level1")
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None,
                    help="模型/checkpoint 输出目录（默认 01_ml_baselines/artifacts/models/bert_<task>）")
    args = ap.parse_args()

    labels = preprocess.TASK_LABELS[args.task]
    train = preprocess.load_split(args.task, "train")
    val = preprocess.load_split(args.task, "validation")
    le = LabelEncoder().fit([r["label"] for r in train])

    X_tr = [r["text"] for r in train]
    y_tr = le.transform([r["label"] for r in train])
    X_va = [r["text"] for r in val]
    y_va = le.transform([r["label"] for r in val])

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels)
    )

    ds_tr = TextDataset(X_tr, y_tr, tok, args.max_len)
    ds_va = TextDataset(X_va, y_va, tok, args.max_len)

    out = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "01_ml_baselines" / "artifacts" / "models" / f"bert_{args.task}"
    )
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        processing_class=tok,
        compute_metrics=compute_metrics(labels),
    )

    print(f"\n[训练] task={args.task} model={args.model} device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    trainer.train()

    print("[验证]")
    res = trainer.evaluate()
    print("  ", {k: round(v, 4) for k, v in res.items() if k.startswith("eval_")})

    # 用 metrics.py 计算论文口径指标
    preds = trainer.predict(ds_va).predictions.argmax(axis=-1)
    rep = M.evaluate([labels[i] for i in y_va.tolist()],
                     [labels[i] for i in preds.tolist()], labels)
    print(f"\n[论文指标] task={args.task}")
    print(f"  Accuracy   : {rep['accuracy']:.3f}")
    print(f"  Cohen Kappa: {rep['cohen_kappa']:.3f}")
    print(f"  Macro-F1   : {rep['macro_f1']:.3f}")

    # 保存模型
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    print(f"模型已保存: {out}")

    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write(f"\n| {args.task} | BERT-base-Chinese (微调) | {rep['accuracy']:.3f} | "
                f"{rep['cohen_kappa']:.3f} | {rep['macro_f1']:.3f} | "
                f"论文(Table 7/8 内部1901测试集): L1 Kappa=0.817/Acc=0.897/M-F1=0.846; L2 Kappa=0.800/Acc=0.850/M-F1=0.849 |\n")


if __name__ == "__main__":
    main()
