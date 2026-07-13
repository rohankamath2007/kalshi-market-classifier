# Kalshi Prediction Market Classifier

Binary classification model to predict whether Kalshi prediction market contracts resolve "yes" or "no" using market signals. Built to surface high-confidence trading opportunities by minimizing missed winning trades.

---

## Overview

Kalshi is a regulated prediction market platform where users trade binary contracts on real-world outcomes. This project builds a classification pipeline on 5,000 contracts, using market features to predict contract resolution.

**The core product decision:** In a prediction market context, a false negative (predicting "no" on a contract that resolves "yes") carries higher user cost than a false positive. The model is designed with this in mind — class weighting is applied to prioritize recall over raw accuracy.

---

## Dataset

- **Source:** Kalshi prediction market contract data
- **Size:** 5,000 binary contracts
- **Class distribution:** ~67% "no" / ~33% "yes" (imbalanced — addressed via class weighting)
- **Target variable:** `result` — whether the contract resolved "yes" or "no"

---

## Features

Final feature set selected through iterative analysis and visualization (KDE plots, pairplots, log transformations):

| Feature | Description |
|---|---|
| `last_price` | Last traded price of the contract |
| `yes_ask` | Ask price for the "yes" side |
| `volume_24h` | 24-hour trading volume |
| `open_interest` | Number of open contracts |

---

## Models

### Random Forest
- `n_estimators=100`, `max_depth=15`, `class_weight='balanced'`
- Train accuracy: 80.9% · Test accuracy: 80.9%
- Recall: 0.97 · F1: 0.79 · ROC-AUC: 0.87

### XGBoost
- `scale_pos_weight=2` to up-weight minority class
- Train accuracy: 83.0% · Test accuracy: 83.6%
- Recall: 0.97 · F1: 0.84 · ROC-AUC: 0.93

**XGBoost selected** for its superior ability to distinguish winning from losing contracts (0.93 vs 0.87 ROC-AUC) while maintaining the same high recall.

---

## Key Results

- **0.97 recall** — minimizes missed winning predictions
- **0.93 ROC-AUC** (XGBoost) vs **0.87** (Random Forest)
- **0.84 average precision** on imbalanced dataset

---

## Tech Stack

- Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn
- Deepnote (notebook environment)

---

## Files

| File | Description |
|---|---|
| `kalshi_classifier.ipynb` | Full notebook — EDA, feature engineering, model training, evaluation |
