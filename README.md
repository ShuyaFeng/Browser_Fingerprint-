# Browser Fingerprint Entropy Attribution

Shapley-value attribution of browser fingerprint entropy. We cast a fingerprint as
a cooperative game where the value of a feature set is its joint entropy,
`v(S) = H(F_S)`, and use Shapley values to decompose the joint entropy into
per-feature contributions that provably sum to the whole, correctly discounting the
redundancy that per-feature marginal entropies double-count.

## Repository layout

```
src/                      core engine
  entropy_fast.py         fast joint-entropy backend (numpy void-view + mixed-radix
                          packing); Miller-Madow, Chao-Shen, and Grassberger
                          estimators; 1/N entropy extrapolation
  shapley_fast.py         exact (2^n) and Monte-Carlo Shapley values, pairwise
                          interaction index, bootstrap CIs, efficiency check
  entropy.py, shapley.py  pandas reference implementations (test baselines)
  data_loader.py          dataset loading helpers

scripts/                  experiments (each writes to results/)
  run_experiment.py       main per-feature attribution and interactions
  estimator_validation.py estimator ground-truth and independence-null validation
  temporal_analysis.py    cross-session stability and the linking game (FPStalker)
  who_remains_exposed.py  anonymity-set distribution and the rare-hardware tail
  optimal_defense.py      defense selection (marginal/Shapley/greedy/optimal) and
                          cluster-level Shapley
  defense_analysis.py     residual-entropy audit of real defenses
  entropy_migration.py    counterfactual entropy migration across a defense sequence
  reidentification.py     uniqueness and k-anonymity bridge
  sensitivity_analysis.py feature-selection robustness
  cross_dataset.py        replication on a second corpus
  interaction_ci.py       bootstrap confidence intervals for interactions
  make_figures.py         render figures from results/
  download_data.sh        fetch the datasets
  preprocess_fpstalker.py parse the FPStalker SQL dump to a feature CSV

tests/                    unit tests (pytest)
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

The datasets are not redistributed here. Fetch and preprocess them with:

```bash
bash scripts/download_data.sh
python scripts/preprocess_fpstalker.py
```

This populates `data/` with the Li and Cao corpus and the FPStalker dataset.

## Running

```bash
python scripts/run_experiment.py          # main attribution (300K sample)
python scripts/estimator_validation.py    # estimator validation
python scripts/temporal_analysis.py       # temporal attribution and linking game
python scripts/optimal_defense.py         # defense selection and cluster Shapley
python scripts/make_figures.py            # render all figures
```

Results and figures are written to `results/` (git-ignored).

## Tests

```bash
pytest tests/
```
