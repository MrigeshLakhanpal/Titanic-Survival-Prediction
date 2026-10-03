# Titanic Survival Prediction
Binary classification on Kaggle's "Titanic: Machine Learning from Disaster" dataset — built not to chase leaderboard accuracy on an already-saturated dataset, but as a deliberate exercise in the engineering discipline most introductory solutions skip: leak-safe preprocessing, honest baseline comparison, and an empirical ablation experiment that *measures* the pipeline's core safety claim instead of just asserting it.

**This is my first complete, end-to-end machine learning project.** I chose a simple, well-understood dataset on purpose, so the focus could stay entirely on methodology — the kind of rigor meant to transfer directly to real, higher-stakes tabular problems, not just to this one.

## The core engineering claim: this pipeline is leak-safe, and that's proven, not assumed

Most public Titanic solutions compute imputation medians, bin edges, and encodings on the *entire* dataset before ever splitting it into train/validation — silently letting validation-fold information leak into the statistics used to preprocess the training fold. This project avoids that by construction: every preprocessing step lives inside a single `scikit-learn` `Pipeline`, so when it's passed to `cross_validate()`, **every statistic is refit from scratch on each fold's training data alone, never on the full dataset.**

That claim isn't just described here — it's demonstrated two separate ways:

- **[`reports/leakage_ablation.md`](reports/leakage_ablation.md)** — the same model, same folds, run once leak-safe and once with the common "preprocess before split" anti-pattern, with the measured score inflation reported honestly: **+0.0213 accuracy, +0.0047 ROC-AUC.**
- **`tests/test_pipeline.py::test_imputer_statistics_differ_across_subsets`** — an automated test asserting leak-safety as a property of the *code*, not just a design intention.

## Skills & concepts demonstrated

| Category | Specifics |
|---|---|
| Data leakage prevention | Fold-scoped preprocessing via `Pipeline` + `ColumnTransformer`; empirical ablation methodology |
| Feature engineering | Regex-based title extraction, group-wise median imputation (custom `BaseEstimator`/`TransformerMixin` transformer), fixed-edge vs. quantile-edge binning distinction |
| Model evaluation | Stratified 5-fold cross-validation; Accuracy + ROC-AUC reported as mean ± standard deviation; majority-class and single-feature floor baselines |
| Model comparison | 7 classifiers benchmarked under one leak-safe harness — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, SVM (via `CalibratedClassifierCV`), Naive Bayes |
| Subgroup auditing | Out-of-fold predictions (`cross_val_predict`) broken down by Sex, Pclass, and Age bracket |
| Software engineering | Modular `src/` package, `pytest` test suite, pinned `requirements.txt`, `joblib` artifact packaging, CLI via `argparse` |

## Repository structure

```
├── notebooks/
│   └── EDA.ipynb                 # Exploratory analysis — never imported elsewhere
├── src/
│   ├── feature.py                # Row-wise feature engineering (leak-free by construction)
│   ├── pipeline.py               # ColumnTransformer + leak-safe preprocessing + build_pipeline()
│   └── train.py                  # Model comparison, cross-validation, final artifact
├── reports/
│   ├── generate_baseline_report.py / baseline_vs_model.md
│   ├── leakage_ablation.py / leakage_ablation.md
│   └── subgroup_analysis.py / subgroup_analysis.md
├── tests/
│   └── test_pipeline.py          # Automated leak-safety and correctness checks
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/titanic) into `data/` (not included in this repo — see Data & Licensing below).

## Reproducing the results

```bash
cd src
python train.py --train_path ../data/train.csv --out ../model.joblib

cd ../reports
python generate_baseline_report.py
python leakage_ablation.py
python subgroup_analysis.py

cd ..
pytest tests/ -v
```

## Results

5-fold stratified cross-validation, mean ± standard deviation across folds.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Majority-class baseline | 0.6162 ± 0.0023 | 0.5000 ± 0.0000 |
| Sex-only rule ("predict survived iff female") | 0.7868 ± 0.0188 | — |
| Logistic Regression | 0.8126 ± 0.0261 | 0.8667 ± 0.0244 |
| Decision Tree | 0.8170 ± 0.0117 | 0.8534 ± 0.0137 |
| **Random Forest** | 0.8159 ± 0.0131 | **0.8704 ± 0.0245** |
| Gradient Boosting | 0.8294 ± 0.0223 | 0.8663 ± 0.0240 |
| K-Nearest Neighbors | 0.7946 ± 0.0246 | 0.8550 ± 0.0204 |
| Support Vector Machine | 0.8103 ± 0.0190 | 0.8553 ± 0.0287 |
| Naive Bayes | 0.7542 ± 0.0174 | 0.8181 ± 0.0339 |

**Best model by ROC-AUC: Random Forest (0.8704 ± 0.0245).** Every model clears both baselines, confirming each is learning genuine signal beyond what sex or majority-class alone reveal. Full table: [`reports/baseline_vs_model.md`](reports/baseline_vs_model.md).

## Leakage ablation

Same model (Logistic Regression), same folds, two preprocessing orders:

| Version | Accuracy | ROC-AUC |
|---|---|---|
| Leaky (global median + global quantile bins, fit before split) | 0.8339 ± 0.0167 | 0.8714 ± 0.0188 |
| Safe (this project's pipeline, refit per fold) | 0.8126 ± 0.0261 | 0.8667 ± 0.0244 |

**Gap: +0.0213 accuracy, +0.0047 ROC-AUC (leaky minus safe).** The leaky version reports a better score than the model would actually achieve on genuinely unseen data — the inflation comes entirely from letting validation-fold rows influence the preprocessing statistics applied to the training fold in the same split. At this dataset size (891 rows) the gap is modest, but the mechanism is real and measured here, not asserted. Full writeup: [`reports/leakage_ablation.md`](reports/leakage_ablation.md).

## Subgroup analysis

Accuracy from out-of-fold predictions (every passenger scored by a model that never trained on them):

- **By Sex:** male 0.8215, female 0.7962 — notably the *inverse* of raw survival rate, where women survived at far higher rates. The sex-based survival signal is strong enough that deviations from it (a woman who didn't survive, a man who did) are what the model struggles to catch.
- **By Pclass:** 2nd class highest at 0.9239; 1st class lowest at 0.7685 — wealthier passengers' outcomes were less predictable from the available features.
- **By Age bracket:** children (<16) are the weakest subgroup in the entire analysis at 0.7349, below every other breakdown in either table (n=83, so partly a small-sample effect, but a genuine and honestly-reported weak spot).

Full tables: [`reports/subgroup_analysis.md`](reports/subgroup_analysis.md).

## Design decisions worth highlighting

A few choices in `src/` that go beyond scikit-learn defaults, and the reasoning behind each:

- **`GroupMedianAgeImputer` (custom transformer, `pipeline.py`):** fills missing `Age` using the median for each (Pclass, Sex) group rather than one dataset-wide median — a materially better estimate — while still being fit only inside each cross-validation fold, so the smarter imputation doesn't reintroduce the leak it's meant to avoid.
- **Fixed-edge age bins vs. quantile fare bins:** `AgeGroup` uses literature-based cutoffs (0/12/18/35/60/80) that don't depend on the data at all, so computing them before or after a split is provably safe — this logic lives in `feature.py`. `Fare` is binned with `KBinsDiscretizer` using *quantile* edges instead, which **are** a statistic of the data — so that step lives inside the `Pipeline` in `pipeline.py`, not alongside the age bins. Two similar-looking features, two different leak-safety requirements, reflected deliberately in where each one's code lives.
- **`CalibratedClassifierCV` wrapping the SVM:** `SVC(probability=True)`'s internal probability estimation is deprecated as of scikit-learn 1.9 and scheduled for removal in 1.11; wrapping it explicitly keeps the model producing well-defined probabilities without depending on a mechanism already flagged for removal.

## Data & licensing

Raw data is not included in this repository. Kaggle's Titanic competition data is distributed under Kaggle's competition rules, not an open license — download it yourself from the [competition page](https://www.kaggle.com/competitions/titanic) for personal/educational use.

## Scope and limitations

This is a historical, closed dataset — there is no live deployment target or real-world stakeholder for a 1912 shipwreck model. The packaged pipeline artifact and reproducibility tooling here exist to demonstrate production-adjacent engineering discipline (leak-safety, automated testing, honest baseline comparison), not to claim real-world applicability. Given the dataset's small size (891 labeled rows), all cross-validated metrics should be read with their reported standard deviation, not as point estimates.