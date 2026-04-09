# fraud-benford

Credit card fraud detection on the ULB dataset. Compares classical and neural models, and adds a Benford's Law feature inspired by forensic accounting. Framing is inspired by Stripe's Payments Foundation Model, which uses sequence structure to catch card-testing attacks that per-transaction models miss.

**Course:** COMP 5630 / 6630, Machine Learning, Auburn Spring 2026.

## Run it

**Colab (one click):** open `notebook/fraud_classifier.ipynb` in Colab and Run All. The notebook clones this repo, installs deps, and downloads the dataset from a public URL.

**Local (uv):**
```
uv sync
uv run jupyter lab
```

## Dataset

ULB credit card fraud dataset, 284,807 transactions over ~48 hours, 492 labeled as fraud (~0.17%). Features V1–V28 are PCA-anonymized; Time and Amount are in the clear. Downloaded at runtime from `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`.

## Layout

```
notebook/   the Run-All deliverable
src/        importable modules used by the notebook
  data.py         download + time-ordered split
  benford.py      leading-digit features and chi-square vs. Benford
  features.py     time-window aggregate features
  models.py       model wrappers (LR, XGBoost, LightGBM, MLP, autoencoder)
  evaluation.py   imbalanced metrics and plots
report/     NeurIPS-format report source
```

## Model progression

1. Logistic regression baseline
2. + class weighting / SMOTE
3. + Benford leading-digit features
4. + time-window aggregates
5. Gradient-boosted trees
6. Small MLP
7. Autoencoder anomaly score (stretch)

## Cite

Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., and Bontempi, G. *Credit card fraud detection: a realistic modeling and a novel learning strategy.* IEEE Transactions on Neural Networks and Learning Systems, 2018.
