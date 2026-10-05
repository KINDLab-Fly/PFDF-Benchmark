# PFDF Explainable Benchmark

Code and processed inputs for *Towards Explainable Benchmarking for Data-driven
Post-Wildfire Debris Flow Prediction*. This release comes from the canonical
`WildFire-DebrisFlow2/03_Code` research code and the paper's saved results.

There are four distinct reproduction levels: reconstruct the archived paper
artifacts, independently evaluate historical held-out predictions and policies,
retrain models with saved hyperparameters, or repeat the full searches. Copying archived figures does **not** constitute model
retraining. The commands below make this distinction explicit.

## Install

Use Python 3.10 or 3.11. Python 3.9 is unsupported because the research code uses
Python 3.10 type annotations. The release was tested in a fresh Python 3.10
environment on macOS/Apple Silicon with CPU training.

```bash
git clone https://github.com/KINDLab-Fly/PFDF-Benchmark.git
cd PFDF-Benchmark
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

If you only want the saved tables and figures, install `requirements-quick.txt`
instead. The release includes the processed workbook and cached terrain,
soil-moisture and structured weather features. These commands need no API keys,
external data downloads or connections to the original research workspace.

## Reconstruct saved paper artifacts

```bash
python scripts/reproduce_all.py
```

This generates Table 1's weather example, the 52 benchmark cells in Table 2,
the 40 regional cells in Table 3, and copies all three archived figure PDFs into
`outputs/`. It verifies immutable input checksums and compares Table 2 with the
numeric reference in `validation/benchmark-reference.csv`. The inputs and their provenance are recorded in
`release-manifest.json`. The same command also independently recomputes all
52 benchmark cells, 40 regional cells, and 230 feature-importance mean/std rows
from portable exports of 640 historical model checkpoints and 50 policy
checkpoints:

```bash
python scripts/verify_published_predictions.py
```

These NPZ files contain numeric held-out predictions and policy probabilities,
not executable pickles. Their checkpoint provenance and hashes are included.
They were exported by evaluating the original trained research checkpoints;
this is an evaluation check rather than a claim of new training.

Figure numbering in the current paper is:

| Artifact | Meaning | Authoritative input |
|---|---|---|
| Figure 1 | Benchmark workflow | `figures/Framework.pdf` |
| Table 1 | AFD to 18 structured indicators | `scripts/make_table1_weather_prompt.py` |
| Table 2 | Original and selected-augmented benchmark | `results/baseline_*_clean_table.csv` |
| Figure 2 | Global and regional top-10 feature importance | `results/rl_feature_selection_augmented_70-15-15_feature_mean.csv` |
| Table 3 | Regional NN minus fixed-USGS Threat Score | `results/region_performance/*absolute_improvement_long_with_counts.csv` |
| Figure 3 | OSD, SWCO and MT diagnostics | Processed workbook and augmented feature cache |

## Run the actual models

First run a bounded integration check. This trains six classical models and one
neural model for each feature configuration at 70/15/15, split seed 42 and neural
initialization seed 42. It retains the original 400-epoch maximum and
validation-based early stopping, but does not repeat hyperparameter searches.

```bash
python -m unittest discover -s tests -v
python scripts/run_benchmark.py --mode smoke
```

Then refit all saved best configurations over all four split ratios, five split
seeds and ten neural initialization seeds:

```bash
python scripts/run_benchmark.py --mode refit
python scripts/verify_refits.py
```

The refit mode uses the included completed-campaign JSON files. It saves fitted
models and the exact train/validation/test indices under `results/checkpoints/`,
plus summary tables and checkpoint records. The verification command compares
means and population standard deviations against the saved four-decimal values
in every Table 2 cell, including a fresh computation of the fixed USGS model.
Neural scores are first averaged over ten initialization seeds within each split;
the reported mean and standard deviation are then computed over five split seeds.

To repeat validation-selected searches from scratch, including **both** original
and selected-augmented neural models, run:

```bash
python scripts/run_benchmark.py --mode search
```

This mode is much more expensive. The neural grid has 200 configurations,
five initialization seeds for hyperparameter selection, and ten evaluation
seeds. Fresh searches write separately tagged outputs and do not overwrite the
archived paper CSVs. Numerical agreement across different hardware and library
versions is not guaranteed; the pinned dependencies and saved split seeds
provide the tested reference environment.

## Verified reproduction limits

The original campaigns used Linux, Python 3.11.15, scikit-learn 1.3.2,
PyTorch 2.4.1+cu124 and an RTX 4090. The two main runtime versions were
recovered from checkpoint metadata and run logs. The release pins scikit-learn
1.3.2 and PyTorch 2.4.1, with NumPy 1.26.4 for compatibility. The complete
locally tested CPU dependency lock is `requirements-lock-cpu.txt`.

The full macOS CPU refit matches all **44 non-neural mean/std cells** at the
saved four-decimal precision. The eight neural cells differ from the original
GPU training, with a maximum absolute mean-TS difference of **0.009440**.
`verify_refits.py` deliberately exits nonzero when four-decimal agreement fails;
its detailed comparison makes these differences visible. The portable historical
prediction/policy verification matches the entire published numerical evidence.
See `validation/` for the actual local reports.

A CUDA machine can install the original PyTorch build before the remaining
requirements:

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

A new CUDA training run was not performed in this local audit. PyTorch does not
guarantee identical CPU/GPU training results; see its [reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
The neural training algorithm and stored paper values have not been changed to
force agreement with this machine.

The reported `Logistic regression` row follows the original table-generation
rule: choose the larger aggregated test mean of FullLR and PenalizedLR for each
ratio. Both families select their hyperparameters on validation folds first.
This final family comparison uses test results and is a limitation of the
original reporting protocol, rather than an additional nested model selection.

## Feature selection and regional diagnostics

After the augmented refits, repeat the policy training and aggregation:

```bash
python scripts/run_benchmark.py --mode rl
```

This uses the augmented 70/15/15 checkpoints for split seeds 42–46, six classical
predictors plus one neural predictor, 1,000 policy steps and 32 masks per step.
The policy uses training samples only. Its objective is predictive separation
collapse relative to the unmasked ensemble, with a sparsity penalty; it does
not establish causal effects.

After the **complete original refit**, regenerate regional NN-versus-USGS
statistics from the saved neural checkpoints:

```bash
PYTHONPATH=src python -m experiments.eval_original_nn_usgs_region_improvement
```

These newly evaluated outputs go to `outputs/refit-region-performance/`, keeping
the archived regional CSVs unchanged.

To redraw the numerical feature-importance and diagnostic figures:

```bash
python scripts/regenerate_plots.py
```

This writes candidate plots to `outputs/regenerated/`; the exact archived
paper PDFs remain unchanged in `figures/`. Fonts and rendering can differ from
the archived PDFs. The regenerated plots use embedded TrueType fonts.

## Data and interpretation notes

- The loader retains 1,243 events and 316 positives from 32 fire IDs. Complete-case
  filtering yields 1,131 events, 292 positives and 31 fire IDs.
- The original model matrix has 14 raw columns: 11 continuous features, two cyclic
  month features and one ordinal zone. Neural models embed the zone into eight
  dimensions. The selected-augmented matrix has 23 raw columns, adding slope,
  a soil-moisture proxy and seven cached structured-weather features.
- The feature key `soilm_30d` retains its historical name but denotes the mean of
  the preceding **three calendar months**; `soilm_7d` denotes the preceding month.
- Structured weather uses cached AFD extraction, covering 1,058/1,131 complete-case
  events. The other forecast products discussed in the paper are not inputs to
  the selected-augmented model in this release.
- Continuous features are normalized using training-fold statistics. Missing
  augmented/weather values become zero after scaling. Splits are stratified by
  zone and label, and the classification threshold is fixed at 0.5.
- The split protocol can share fires, watershed segments and weather documents
  across folds. Its scores measure retrospective interpolation; they are not an
  evaluation on entirely unseen fires or a prospective deployment guarantee.
- Selected augmented features were chosen in exploratory studies rather than
  nested feature selection inside every evaluation fold.
- Table 3 in the manuscript prints `+0.12` for NAZ at 60/20/20. The underlying
  saved value is `0.1145959596`, which rounds to `+0.11`; regenerated tables use
  the numeric source. Near-zero negative values can print as `-0.00` instead of
  the manuscript's `+0.00`. Rare-region metrics may be undefined and may average
  fewer than five available split seeds. The release does not edit the manuscript.

The workbook originates from USGS Open-File Report 2016-1106:
<https://doi.org/10.3133/ofr20161106>. Terrain and soil-moisture values are cached
derived products; rerunning external collection is unnecessary for this release.
Cached LLM outputs are inputs to this benchmark; this release does not promise
bit-identical regeneration of those outputs from a changing hosted model.

## Repository layout

```text
data/        Processed workbook and cached derived features
figures/     Exact archived paper figure PDFs
results/     Curated results and completed hyperparameter searches
scripts/     Artifact, training, verification and plotting entrypoints
src/         Canonical research modules adapted to this standalone layout
tests/       Dataset, split, feature-scaling and metric contract checks
validation/  Numeric benchmark reference and verification reports
outputs/     Generated artifacts (ignored by Git)
```

The release intentionally excludes manuscript sources, API credentials, unrelated experimental
models, large raw caches, trained model weights and local absolute-path symlinks.
Compact historical prediction/policy exports remain included as verification evidence.
GitHub Actions reconstructs the saved artifacts and exercises a real model smoke
run; a complete new hyperparameter search is not a CI check.
