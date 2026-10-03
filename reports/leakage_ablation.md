# Leakage Ablation

Same model (Logistic Regression), same 5-fold StratifiedKFold split, two different preprocessing orders.

| Version | Accuracy | ROC-AUC |
|---|---|---|
| Leaky (global median + global quantile bins, fit before split) | 0.8339 ± 0.0167 | 0.8714 ± 0.0188 |
| Safe (this project's Pipeline, refit per fold) | 0.8126 ± 0.0261 | 0.8667 ± 0.0244 |

**Gap:** +0.0213 accuracy, +0.0047 ROC-AUC (leaky minus safe).

A positive gap means the leaky version reports a better score than the model would actually achieve on genuinely unseen data — the inflation comes entirely from letting validation-fold rows influence the median and quantile-bin statistics used to preprocess the training rows in the same fold.

Note: at this dataset size (891 rows, 5 folds) the numeric gap can be small or noisy — the point of this experiment is demonstrating that the mechanism exists and reporting the measured gap honestly, not manufacturing a large effect.