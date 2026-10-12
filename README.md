# Does Explainability Survive Data Drift?

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2610.05379-B31B1B)](https://arxiv.org/abs/2610.05379)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

Reproduction code and data for the paper **"Does Explainability Survive Data Drift?"** by Samuel Ozechi.

---

## Overview

This repository contains the experimental pipeline for evaluating whether local explanations produced by an Isolation Forest anomaly detector remain faithful to the model's decisions as the input distribution drifts over time.

The study addresses a gap in the anomaly detection and explainability literature: detection performance is monitored routinely, but explanation fidelity is not, even in regulated domains where explanations are required for compliance. The paper investigates:

1. Whether the dataset exhibits covariate shift **without** concept drift.
2. Whether explanation structure remains stable under drift.
3. Whether decision-level explanation fidelity degrades across future windows.
4. Whether explanations produced before drift remain faithful to a model that has been updated on the drifted distribution.

**Key finding:** Explanation fidelity declines under data drift even when the feature–target relationship is stable. Prior explanations remain informative but are consistently less faithful to the current model's decisions than freshly generated ones. The transferability gap is stable across windows rather than compounding.

---

## Dataset

The study uses the **IEEE-CIS Fraud Detection dataset** from Kaggle.

### Download

1. Create a Kaggle account and accept the competition rules at [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection).
2. Download `train_transaction.csv` and optionally `train_identity.csv`.
3. Place the transaction file at `data/raw/ieee_cis/train_transaction.csv`.

### Dataset characteristics

- Approximately 590,540 labelled transactions.
- Approximately 3.5% fraud rate.
- 376 numeric features selected after preprocessing.
- Chronological ordering available through the `TransactionDT` column.

The dataset is not redistributed with this repository and remains subject to Kaggle's terms of use.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/explanation-fidelity-drift.git
cd explanation-fidelity-drift
```

Replace `<your-username>` with the appropriate GitHub username or organization.

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Dependencies

See `requirements.txt` for the project dependencies. The intended dependency versions include:

- NumPy ≥ 1.26
- pandas ≥ 2.2
- scikit-learn ≥ 1.4
- SciPy ≥ 1.11
- Matplotlib ≥ 3.8
- Seaborn ≥ 0.13

---

## Reproducing the Experiments

The following commands describe the experimental workflow. Run them from the repository root with the virtual environment activated and the dataset available at the expected path.

### 1. Dataset audit (optional)

```bash
python3 scripts/audit_ieee_cis.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output results/audit
```

### 2. Drift characterisation

#### Performance-drop test for concept drift

```bash
python3 scripts/confirm_concept_drift.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output-dir results/concept_drift
```

#### Conditional drift test

Classifier two-sample test using the joint information in \(X\) and \(Y\):

```bash
python3 scripts/test_conditional_drift.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output-dir results/conditional_drift
```

#### Covariate shift between consecutive windows

```bash
python3 scripts/compute_consecutive_window_drift.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output results/concept_drift/consecutive_window_drift.csv
```

### 3. Detection benchmark

#### Static Isolation Forest baseline

```bash
python3 scripts/run_static_if.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output results/tables/static_if_v1.csv
```

#### Full 5 × 4 detection grid

```bash
python3 scripts/evaluate_detection_full_grid.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output-dir results/retrained/full_grid
```

### 4. Explanation generation and structure

#### Generate ExIFFI explanations across W1–W4

```bash
python3 scripts/generate_exiffi_temporal_explanations.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --sample-strategy stratified
```

#### Evaluate explanation structure

```bash
python3 scripts/evaluate_exiffi_temporal_explanations.py
```

### 5. Fidelity analysis

#### Metric sufficiency: ExIFFI versus random control

```bash
python3 scripts/evaluate_exiffi_fidelity_metric_sufficiency.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --windows 1,2,3,4
```

#### Temporal fidelity trajectories

```bash
python3 scripts/evaluate_exiffi_temporal_fidelity.py \
    --transactions data/raw/ieee-cis/train_transaction.csv \
    --sample-strategy stratified
```

#### Full-grid fidelity analysis

```bash
python3 scripts/analyze_full_grid_fidelity.py \
    --results-dir results/retrained \
    --output-dir results/retrained/full_grid \
    --primary-k 5
```

### 6. Cross-model transferability

#### Cumulative retraining experiment

```bash
python3 scripts/run_cumulative_retraining.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output-dir results/cumulative_retraining
```

#### Static explanation transfer fidelity against current models

```bash
python3 scripts/evaluate_static_explanation_transfer_fidelity.py \
    --transactions data/raw/ieee_cis/train_transaction.csv \
    --output-dir results/static_transfer_fidelity
```

### 7. Generate figures

```bash
python3 scripts/create_figures_detection_separate.py
python3 scripts/create_figures_explanation_structure_separate.py
python3 scripts/create_figure_5_5_2_static_transfer.py
```

**Note:** Verify script names, command-line arguments, and output paths against the current repository before running the pipeline. The commands above reflect the documented workflow and may require adjustment if scripts have been renamed or removed.

---

## Key Outputs

| Path | Description |
|---|---|
| `results/concept_drift/concept_drift_results.csv` | Performance-drop test results |
| `results/conditional_drift/conditional_drift_results.csv` | Conditional drift test results |
| `results/concept_drift/covariate_shift_summary.csv` | Per-window covariate shift |
| `results/tables/static_if_v1.csv` | Static model detection F1 and PR-AUC across W1–W4 |
| `results/retrained/full_grid/full_grid_detection.csv` | Full detection grid |
| `results/retrained/full_grid/full_grid_cell_summary.csv` | Fidelity by model, window, and \(k\) |
| `results/static_transfer_fidelity/static_transfer_fidelity_comparison.csv` | Static transfer versus current-model baseline |
| `results/cumulative_retraining/cumulative_comparison.csv` | Cumulative retraining comparison |
| `results/figures/` | Paper figures in PNG and PDF formats |

---

## Experimental Design

### Chronological split

- **Training period:** Earliest 60% of transactions (354,324 observations).
- **Evaluation windows:** Remaining 40%, divided into four equal, chronologically ordered windows, W1–W4 (59,054 observations each).
- **Preprocessing:** Fitted on the training period only and applied unchanged to all future windows.

### Model conditions

| Condition | Description |
|---|---|
| Static | One Isolation Forest trained on the initial training period and applied unchanged to all future windows. |
| Retrained | One Isolation Forest per evaluation window, trained on that window's data only. |
| Cumulative | One Isolation Forest per evaluation window, trained on the initial training data plus all windows up to and including that evaluation window. |

All models use the same documented hyperparameters: 300 trees, `max_samples="auto"`, `contamination="auto"`, and axis-aligned splits.

### Explanation method

Local explanations are generated using ExIFFI, a model-specific feature-importance method that aggregates path weights along an observation's decision paths through an Isolation Forest.

Feature contributions are aggregated across the relevant paths and trees, producing a feature-importance vector for each observation.

### Fidelity metric

Decision-level fidelity is measured using a controlled feature intervention:

1. Identify the top-\(k\) features in the explanation.
2. Replace those features with their training-period medians while leaving all other features unchanged.
3. Re-score the model on the perturbed observation.
4. Calculate the relative score change.

The metric is:

\[
\text{Relative Score Change}
=
\frac{|\text{score}(x')-\text{score}(x)|}
{|\text{score}(x)|+\epsilon}
\]

where \(x\) is the original observation, \(x'\) is the perturbed observation, and \(\epsilon\) is a small constant to avoid division by zero.

The primary reported value is \(k=5\).

### Statistical framework

- Bootstrap: 2,000 iterations with 95% confidence intervals.
- Permutation tests: 200–1,000 permutations per comparison.
- Multiple-comparison correction: Benjamini–Hochberg false discovery rate (FDR), \(\alpha=0.05\).
- Fixed random seed: 42 across the pipeline.

---

## Reproducibility Notes

### Random seeds

Scripts use `random_state=42` by default. Retrained models use `42 + window_index`.

### Sampling

The documented sampling strategy uses 100 observations per window, with stratified sampling of 50% fraud and 50% non-fraud by default, using seed `42 + window_index`.

### Determinism

Given a fixed random seed and the same version of the IEEE-CIS dataset, results should match the paper within numerical precision, subject to differences in software versions and execution environments.

### Runtime

The full pipeline is estimated to take approximately 3–4 hours on a modern laptop. Cumulative retraining and transfer-fidelity evaluations are the most time-consuming components.

### Hardware

No GPU is required. The experiments are designed to run on a CPU.

---

## Results from the Paper

The following tables summarise the principal findings reported in the paper.

### Drift characterisation

| Window | Fraud rate | Performance drop | BH-FDR adjusted p-value | Concept drift detected |
|---|---:|---:|---:|---|
| W1 | 4.32% | −0.0272 | 1.000 | No |
| W2 | 3.49% | −0.0128 | 1.000 | No |
| W3 | 3.13% | +0.0041 | 0.911 | No |
| W4 | 3.75% | −0.0094 | 1.000 | No |

Covariate shift: 53.7%–70.2% of features show statistically significant distributional change per window.

### Detection performance

| Window | Static F1 | Best retrained F1 | Relative F1 improvement |
|---|---:|---:|---:|
| W1 | 0.27 | 0.30 | +9.5% |
| W2 | 0.24 | 0.26 | +7.9% |
| W3 | 0.22 | 0.23 | +6.4% |
| W4 | 0.20 | 0.22 | +8.2% |

### Explanation fidelity: main result

| Window | Current-model baseline | Static explanation transfer | Random control | Transfer/baseline |
|---|---:|---:|---:|---:|
| W1 | 0.0360 | 0.0270 | 0.0019 | 0.75 |
| W2 | 0.0407 | 0.0291 | 0.0019 | 0.72 |
| W3 | 0.0394 | 0.0303 | 0.0019 | 0.77 |
| W4 | 0.0407 | 0.0291 | 0.0021 | 0.72 |

Prior explanations remain substantially more informative than the random control, with static-to-random ratios of approximately 13.7×–16.3×. However, they are consistently around 25%–30% less faithful than freshly generated explanations. The transferability gap remains stable across the four evaluation windows rather than progressively widening.

---

## Citation

If you use this code or reproduce these experiments, please cite the paper:

```bibtex
@article{ozechi2026explainability,
  title   = {Does Explainability Survive Data Drift?},
  author  = {Ozechi, Samuel},
  journal = {arXiv preprint arXiv:2610.05379},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.05379}
}
```

---

## Contact

**Samuel Ozechi**

- Email: ozechisamuel@gmail.com
- Affiliation: University of East London; Interswitch Group

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

The IEEE-CIS Fraud Detection dataset is subject to Kaggle's terms of use and is not redistributed with this repository.

---

## Acknowledgements

- The IEEE Computational Intelligence Society for the IEEE-CIS Fraud Detection dataset.
- The authors of DIFFI and ExIFFI for the model-specific explanation methods on which this study builds.
- The open-source community behind scikit-learn, pandas, and the broader Python scientific stack.

---

## Known Issues and Future Work

### Single-dataset scope

All results are based on the IEEE-CIS dataset, Isolation Forest, and ExIFFI. Generalisation to other datasets, detector families, and explanation methods remains future work.

### Perturbation protocol

Fidelity is measured using training-period median substitution. Alternative perturbation schemes, including mean substitution, out-of-distribution sampling, and feature masking, may produce different absolute magnitudes.

### Finite temporal windows

Four evaluation windows establish directional trends but do not characterise every possible drift regime.