

## 基线1：传统机器学习 (SVM / XGBoost + TF-IDF)
> 发布划分(2500条)上的可复现近似；论文 exact 数字基于未公开的内部6000/1901。
> A=论文忠实配置(对照 Table 3)；B=增强配置(仅对照分析，非复现)。

| Task | Config | Acc | CohenKappa | MacroF1 | 备注 |
|---|---|---:|---:|---:|---|
| task1_level1 | SVM|paper | 0.853 | 0.780 | 0.853 | | 论文参考(Acc/Kappa/MF1)=0.508/0.175/0.442
| task1_level1 | XGBoost|paper | 0.820 | 0.730 | 0.819 | | 论文参考(Acc/Kappa/MF1)=0.487/0.153/0.42
| task2_level2 | SVM|paper | 0.845 | 0.793 | 0.843 |
| task2_level2 | XGBoost|paper | 0.848 | 0.797 | 0.847 |
| task1_level1 | SVM|enhanced | 0.813 | 0.720 | 0.812 |
| task1_level1 | XGBoost|enhanced | 0.813 | 0.720 | 0.814 |
| task2_level2 | SVM|enhanced | 0.815 | 0.753 | 0.814 |
| task2_level2 | XGBoost|enhanced | 0.828 | 0.770 | 0.827 |

| task1_level1 | BERT-base-Chinese (微调) | 0.867 | 0.800 | 0.866 | 论文(Table 7/8, 内部1901测试集): L1 Kappa=0.817/Acc=0.897/M-F1=0.846; L2 Kappa=0.800/Acc=0.850/M-F1=0.849 |
