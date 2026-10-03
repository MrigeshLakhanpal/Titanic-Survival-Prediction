# Baseline vs Model Comparison

5-fold stratified cross-validation, mean ± standard deviation
across folds.

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression| 0.8126 ± 0.0261| 0.8667 ± 0.0244 |
| Decision Tree| 0.8170 ± 0.0117| 0.8534 ± 0.0137 |
| Random Forests| 0.8159 ± 0.0131| 0.8704 ± 0.0245 |
| Gradient Boosting| 0.8294 ± 0.0223| 0.8663 ± 0.0240 |
| K-Nearest Neighbors| 0.7946 ± 0.0246| 0.8550 ± 0.0204 |
| Support Vector Machine| 0.8103 ± 0.0190| 0.8553 ± 0.0287 |
| Naive Bayes| 0.7542 ± 0.0174| 0.8181 ± 0.0339 |
Best model by ROC-AUC: Random Forests
ROC-AUC = 0.8704 ± 0.0245

Any model scoring below the sex-only rule's accuracy has not learned anything beyond what a single column already reveals.