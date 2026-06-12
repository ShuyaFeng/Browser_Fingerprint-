# Where Does Browser Fingerprint Uniqueness Come From? — Research Artifact

Anonymized artifact for the submission *"Where Does Browser Fingerprint
Uniqueness Come From? A Shapley-Value Attribution of Entropy and Its
Implications for Defense."*

It contains the full analysis pipeline: bias-corrected entropy estimators
(Miller–Madow and coverage-adjusted Chao–Shen), exact and Monte-Carlo Shapley
solvers over the fingerprint entropy game, the Shapley interaction index, the
defense audit by residual joint entropy, the re-identification bridge, the
temporal/linking analyses, and every figure-generation script.

## Layout

```
src/                  Core library
  entropy.py            Reference entropy estimators
  entropy_fast.py       Fast estimators (mixed-radix packing, Chao-Shen, Miller-Madow)
  shapley.py            Reference Shapley implementation
  shapley_fast.py       Exact (2^n enumeration) and permutation Monte-Carlo solvers,
                        pairwise Shapley interaction index
  data_loader.py        Dataset loading and feature derivation
scripts/              Analysis and figure scripts (see mapping below)
tests/                Unit tests (estimators, solvers, efficiency axiom)
results/              Precomputed outputs (CSV/JSON) from the paper's runs (300K sample)
results_scale1M/      Outputs of the 1M-sample robustness run
data/                 Dataset download target (see "Datasets" below)
```

## Setup

Python 3.9+:

```
pip install -r requirements.txt
```

## Quick start (no dataset download required)

The repository ships the precomputed result tables, so you can verify the
pipeline and regenerate every data-backed figure of the paper in seconds:

```
python -m pytest tests/            # estimator + solver unit tests
python scripts/replot_figures.py   # rebuilds figures from results/*.csv
```

## Datasets

Both corpora are public; neither is redistributed here.

- **Li & Cao (IMC 2020)** — primary corpus, 7.2M fingerprints.
  `bash scripts/download_data.sh` downloads it from Zenodo
  (record 7743719, ~3.7 GB compressed) into `data/raw/li_cao_imc2020/`.
- **FPStalker (Vastel et al., S&P 2018)** — replication and temporal corpus.
  Obtain `fingerprints.csv` from the authors' public release
  (github.com/Spirals-Team/FPStalker) and place it in `data/raw/fpstalker/`,
  then run `python scripts/preprocess_fpstalker.py`.

## Full reproduction: script-to-output mapping

| Script | Produces |
|---|---|
| `run_experiment.py` | Shapley attribution + interaction matrix + device split (`shapley_attribution.csv`, `pairwise_interactions.csv`, `desktop_vs_mobile.csv`; figures `fig1`–`fig4` via `make_figures.py`) |
| `defense_analysis.py` | defense and point-defense residual entropy (`defense_effectiveness.csv`, `point_defenses.csv`; `fig5`, `fig6`) |
| `entropy_migration.py` | counterfactual defense-deployment sequence (`entropy_migration.csv`; `fig7`) |
| `reidentification.py` | uniqueness and anonymity sets per defense, entropy-vs-uniqueness sweep (`reidentification.csv`, `entropy_uniqueness.csv`; `fig8`, `fig9`) |
| `sensitivity_analysis.py` | random-subset and 25-feature sensitivity (`sensitivity_*.csv`; `fig10`) |
| `cross_dataset.py` | Li & Cao vs. FPStalker replication (`cross_dataset.csv`; `fig11`) |
| `estimator_validation.py` | ground-truth + shuffled-null estimator controls (`estimator_validation.json`; `fig12`) |
| `temporal_analysis.py` | stability vs. one-shot Shapley, linking game (`temporal_stability.csv`, `linking_game.csv`; `fig13`) |
| `optimal_defense.py` | k-feature selection rules, cluster-level Shapley (`optimal_defense.csv`, `cluster_shapley.csv`; `fig14`) |
| `who_remains_exposed.py` | anonymity-set CDF, exposure by GPU rarity (`exposure.csv`; `fig15`) |
| `interaction_ci.py` | bootstrap CIs for all pairwise interactions (`interaction_ci.json`) |
| `make_overview.py` | framework overview figure (`fig0`) |

Notes:
- All sampling is seeded; re-running reproduces the paper's numbers exactly.
- The exact solver enumerates all 2^18 feature subsets in minutes on a laptop;
  `run_experiment.py --skip-interactions` skips the interaction matrix.
- The Monte-Carlo solver is validated against the exact solver to within
  1e-9 bits; the efficiency axiom (sum of Shapley values = joint entropy) is
  checked at the end of every run.

## License

MIT (see LICENSE). Released anonymously for double-blind review.
