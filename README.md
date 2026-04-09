# fraud-benford

Credit card fraud detection on the public ULB dataset. Compares seven supervised classifiers plus one unsupervised anomaly model, adds a Benford's Law feature inspired by forensic accounting, and tests a time-window aggregate feature set inspired by Stripe's published description of their Payments Foundation Model.

**Course:** COMP 5630 / 6630, Machine Learning, Auburn University, Spring 2026.

**Deadline:** April 21, 2026.

## Run it

**Colab (one click):** open `notebook/fraud_classifier.ipynb` in Colab and Run All. The notebook clones this repo, installs dependencies, and downloads the dataset from a public Google Storage URL. No Kaggle auth or manual data download needed.

Colab quick-link:
`https://colab.research.google.com/github/dutch-casa/fraud-benford/blob/main/notebook/fraud_classifier.ipynb`

**Local:**
```
pip install -r requirements.txt
jupyter lab notebook/fraud_classifier.ipynb
```

Expected total runtime on a Colab CPU runtime: 4 to 8 minutes. The autoencoder is the slowest step (about 1 to 2 minutes).

## Dataset

ULB credit card fraud dataset, 284,807 transactions over roughly 48 hours in 2013, with 492 labeled fraud (0.173 percent positive rate). Features `V1` through `V28` are PCA-anonymized by the original authors; `Time` and `Amount` are in the clear. The notebook fetches the dataset at runtime from TensorFlow's public mirror:

```
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Citation: Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., and Bontempi, G. *Credit card fraud detection: a realistic modeling and a novel learning strategy.* IEEE Transactions on Neural Networks and Learning Systems, 2018.

## Repository layout

```
notebook/
  fraud_classifier.ipynb          the Run-All deliverable for Colab
  fraud_classifier_postrun.ipynb  snapshot of a completed run with outputs
src/
  data.py         load_raw, time_ordered_split, time_ordered_three_way_split
  benford.py      leading-digit extraction, chi-square vs. Benford, one-hot features
  features.py     rolling time-window aggregate features (count, sum, z-score, time-since-last)
  evaluation.py   AUPRC, ROC curves, Benford histogram plot, model comparison table
  models.py       uniform fit/predict_proba wrappers for LR, XGBoost, LightGBM, MLP, autoencoder
report/
  report.tex      NeurIPS-format report source (main deliverable alongside the notebook)
pyproject.toml    dependency declarations for local development
requirements.txt  pip-installable mirror of the dependencies for Colab
```

Every module in `src/` has typed function signatures and contracts encoded as runtime assertions where they are non-trivial. The code is organized around a data-oriented and invariant-first philosophy: data structures define the shape of each pipeline stage, functions transform one stage into the next, shared state is avoided, and models expose a uniform `fit(X, y)` / `predict_proba(X)` interface.

## Model progression

1. Logistic regression on base features, unweighted
2. Logistic regression on base features with `class_weight="balanced"`
3. Logistic regression with Benford leading-digit features
4. Logistic regression with Benford and time-window aggregates (full feature set)
5. XGBoost on the full feature set, untuned
6. LightGBM on the full feature set, untuned
7. Two-layer scaled MLP on the full feature set
8. Autoencoder anomaly detector trained only on legitimate rows
9. LightGBM tuned via a 16-configuration grid on the validation slice
10. XGBoost tuned via an 18-configuration grid on the validation slice
11. LightGBM ablation on base features only (feature-importance check)
12. LightGBM ablation on base plus Benford features only (feature-importance check)

## Headline results

See `report/report.tex` for the full comparison table and the discussion. Top line: the scaled MLP reaches test AUPRC 0.8161 and precision 0.833 at 80 percent recall, the best on both metrics. The tuned LightGBM with Benford features only reaches AUPRC 0.808. Benford features give a small consistent lift across linear and tree models; the time-window features hurt both, which we attribute to the dataset lacking per-card identifiers.

Fraud transaction amounts are 6.9 times more distant from Benford's predicted leading-digit distribution than legitimate amounts, a fingerprint that holds up independent of any classifier.

## Reproducing the report numbers

1. `git clone https://github.com/dutch-casa/fraud-benford.git`
2. Open `notebook/fraud_classifier.ipynb` in Colab
3. Runtime &rarr; Run All
4. Compare the output of the final comparison table to Table 1 in the report

The random seed is fixed at 42 throughout, so the numbers should reproduce bit-for-bit on the same Colab runtime version.

## Author

Dutch Casa, solo submission. See the Collaboration and Tools section of the report for the full tools list and attribution.
