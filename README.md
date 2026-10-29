# PU Learning for Fraud Detection: when some fraud is never found

In fraud and AML data, a "1" means **confirmed** fraud. A "0" does **not** mean clean: some fraud is never detected, so it hides among the 0s.
A normal model treats these hidden frauds as normal, and learns to score fraud-like cases lower.

This is a **Positive-Unlabeled (PU) learning** problem: we have labeled positives, and unlabeled data that mixes negatives with hidden positives.

## Approach

1. Take a fraud dataset where all labels are known, and **hide half of the fraud labels** to simulate undetected fraud.
   Because the truth is still known, we can measure how much each method recovers.
2. Compare three models, all evaluated on the **true** labels:
   - **Oracle**: trained on the true labels. Not possible in real life; it shows the best case.
   - **Naive**: every unlabeled row counts as non-fraud.
   - **PU** (Elkan & Noto, 2008): unlabeled rows that look like fraud get a lower weight as non-fraud.
3. Use **EBM** to see *where* PU changes what the model learns.

### PU learning in four steps

`y` = true label (not seen in real life), `s` = what we see (1 = confirmed fraud, 0 = unlabeled)

| Step | What | How |
|---|---|---|
| 1 | Estimate `g(x) = P(s = 1 \| x)` | Train a model on `s`. Each row is scored by a model that did not see it (cross-fitting). |
| 2 | Estimate `c = P(s = 1 \| y = 1)`, the share of fraud that is found | Average `g(x)` of the labeled fraud (often too low when classes overlap), a business estimate of the fraud rate, or best: a **random audit** of unlabeled cases. |
| 3 | Estimate `w`, the chance that an unlabeled row is hidden fraud | `w = (1 − c) / c × g / (1 − g)`. Most rows get `w ≈ 0`. |
| 4 | Train the final model | Labeled fraud = fraud (weight 1). Each unlabeled row is used twice: as fraud with weight `w`, and as non-fraud with weight `1 − w`. |

**Assumption (SCAR)**: fraud is found at random, so every fraud has the same chance `c` of being found. In real life, some types of fraud are easier to find.

## Data and setup

- **Data**: [Bank Account Fraud Dataset Suite (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022), `Base.csv`: 1M bank account applications over 8 months, about 1% fraud.
- **Out-of-time split**: train on months 0–5, test on months 6–7.
- **Simulation**: half of the fraud cases keep their label (`c = 0.5`); the rest become unlabeled.
- **Model**: LightGBM for the main comparison; EBM to compare the learned shapes.

## Results (test months, true labels, c = 0.5)

| Model | AUC | PR-AUC | Hidden fraud found in top 1% | Mean predicted probability |
|---|---|---|---|---|
| Oracle (true labels) | 0.8929 | 0.1903 | 24.3% | 1.19% |
| Naive | 0.8875 | 0.1799 | 22.6% | 0.60% |
| **PU** | **0.8929** | **0.1874** | **23.8%** | **1.19%** |

True fraud rate in the test months: 1.40%.

- PU closed the gap between naive and oracle: **all of it for AUC**, and about **70% for PR-AUC** and for hidden fraud found in the top 1%.
- The gain is **small** here, because half of the labels were kept, so the naive model was not far from the best case. The fewer frauds are labeled, the more the naive model is distorted.
- The naive model predicts **half the true level** (it learns the chance of being *confirmed*, which is `c` × the chance of fraud). PU brings it back to the oracle level. This part is expected once `c` is known.
- With EBM, the naive model **weakened the effects in the most fraud-like ranges**, and PU brought them back closer to the oracle.

## What I learned

- The effect of PU depends on **how much fraud is missed**. So in practice, I would first estimate that, for example with a **random audit**, and then decide if PU is worth it.
- **Estimating `c` is the hardest part.** The standard estimate assumes fraud and normal cases are perfectly separable; when they overlap, it is too low.
- The method assumes labels are found **at random**. Real investigations find some fraud types more easily, which breaks this assumption.

## Repository structure

```
pu-learning-fraud/
├── src/
│   ├── pu_learning_fraud.ipynb   # the analysis
│   └── pu_utils.py               # PU weights, cross-fitting, c sensitivity (reusable)
└── data/                         # raw data (not in git)
```

## How to run

```bash
pip install -r requirements.txt
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw/bank_account_fraud/
```

Unzip `Base.csv` into `data/raw/bank_account_fraud/`, then run `src/pu_learning_fraud.ipynb` from top to bottom.

## Background

At KPMG Japan, I used PU learning for financial statement fraud, where the label (a regulator's penalty) is very strict and many frauds are never labeled. On that real data, the gain from PU was much bigger than in this simulation.
