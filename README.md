# Code for "Conformal Prediction Under Distribution Shift"

This repository contains the Python code used to reproduce the numerical population study in Chapter 4 of the bachelor thesis

**Conformal Prediction Under Distribution Shift: Linking DRIG and Robust Validation**.

No empirical data are used. The code implements deterministic population calculations for the fixed-B Gaussian SCM settings studied in the thesis.

## Setup

Install the required packages with

```bash
pip install -r requirements.txt
```

## Reproducing the Chapter 4 results

Run

```bash
python run_chapter4.py
```

The script writes outputs to

```text
outputs/<setting>/latest/
outputs/_summary/latest/
```

The combined summary table is written to

```text
outputs/_summary/latest/chapter4_setting_summary.csv
outputs/_summary/latest/chapter4_setting_summary.tex
```

The main setting-specific outputs are

```text
outputs/mean_only_p2/latest/
outputs/cov_only_p2/latest/
outputs/mixed_second_moment_p2/latest/
```

Each setting folder contains tables, plots, metadata, and a short run summary.

The default run uses `gamma_step=0.01`, `tv_grid_n=1400`, and `opt_grid_n=151`, matching the final thesis-oriented plotting run.

## Main files

| Thesis quantity / step | Code location |
|---|---|
| fixed-reference coefficient path | `src/scm_core.py`, `solve_b_gamma` |
| residual laws under the fixed-B SCM | `src/scm_core.py`, `observed_residual_laws` |
| folded-normal score distributions | `src/score_tools.py` |
| total-variation robust-validation quantities | `src/score_tools.py`, `src/analysis.py` |
| mean-only and covariance-only uncertainty sets | `src/uncertainty_sets.py` |
| gamma-grid analysis for Chapter 4 | `src/analysis.py` |
| thesis-oriented figures | `src/plotting.py` |
| complete reproduction script | `run_chapter4.py` |

## Notes

This repository is not intended as a general-purpose implementation of DRIG or robust validation. It is a reproduction script for the deterministic population calculations reported in Chapter 4 of the thesis.
