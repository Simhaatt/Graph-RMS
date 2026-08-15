# Graph-RMS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21876447.svg)](https://doi.org/10.5281/zenodo.21876447)

Reproducibility package for **Graph-RMS: Training-Free and Class-Count-Free
Spectral-Spatial Region Discovery in Hyperspectral Remote-Sensing Images**.

Graph-RMS constructs a sparse reciprocal spectral-spatial graph, applies
damped graph diffusion, extracts fine representation-space modes, performs
spatial cleanup, and consolidates mode prototypes using reference-free
partition safeguards. Under a fixed configuration, the implementation is
deterministic. The primary method does not receive labelled training pixels or
the semantic class count during per-scene processing.

This repository is a curated release copy assembled from the frozen evidence
used by the manuscript. It includes source code, exact primary and automatic
selector settings, nine per-scene partitions, classical and no-class-count
comparison tables, diagnostic studies, provenance records, and manuscript
figure assets. Original hyperspectral cubes are not redistributed.

## Scientific scope

- **RMS means Robust Mean Shift.** It does not mean root mean square. The
  separate prototype-dispersion statistic is a root-mean-square Euclidean
  radius in standardized diffusion-PCA space.
- The output is a structural spectral-spatial region partition, not necessarily
  a one-region-per-semantic-class map.
- Reference labels are introduced only after the selected partition is frozen,
  except during the explicitly declared global calibration of the later
  automatic-selector experiment on eight development scenes.
- Trento is the independent holdout only for the primary scene-specific
  procedure. Because its primary outcome was known before automatic-v2 was
  designed, the automatic Trento result is a retrospective transfer audit.
- Leiden on the same weighted graph is competitive. The evidence does not
  support a universal-superiority claim.

## Repository contents

```text
Graph-RMS/
  graphrms/               Core graph, diffusion, mode, consolidation, and selection code
  configs/                Frozen primary and automatic-selector protocols
  scripts/                Rerun, baseline, analysis, export, and QA entry points
  data/                    Acquisition instructions only; raw cubes are excluded
  results/
    main_tables/          Manuscript comparison and statistical tables
    supporting_results/   Ablation, sensitivity, robustness, profiling, and controls
    per_dataset/          Frozen summaries, endpoint curves, and selected partitions
    automatic_selector/   Eight-scene calibration and retrospective Trento audit
    provenance/           Environment, checksums, manifests, and registry
  figures/                Curated manuscript-linked figure files
  manuscript_support/     Supplied manuscript source and availability-statement fragment
  docs/                   Dataset, experiment, FAIR, audit, release, and result maps
  tests/                  Lightweight structural and numerical QA
```

## Installation

The recorded validation environment used Python 3.12.13, NumPy 2.0.2,
scikit-learn 1.6.1, SciPy 1.16.3, PyTorch 2.11.0+cu128, and an NVIDIA
A100-SXM4-80GB. The `+cu128` PyTorch build is recorded in
`results/provenance/pip_freeze.txt`; install the PyTorch wheel appropriate to
your CUDA driver before installing the remaining requirements if the default
wheel is unsuitable.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

For editable installation:

```bash
python -m pip install -e .
```

## Data preparation

Read [data/README.md](data/README.md) and [docs/DATASETS.md](docs/DATASETS.md).
The loader can retrieve the exact public mirror filenames used by the study,
but users remain responsible for checking and accepting the original dataset
terms. The repository does not grant rights to any third-party cube or
reference map.

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dataset salinas_a --data-dir data
```

## Reproducing the primary runs

The exact frozen settings are in `configs/primary.yaml` and the machine-readable
registry is in `results/provenance/frozen_results_registry.json`.

```bash
# One dataset, one complete frozen execution
python scripts/run_main_experiments.py --datasets salinas_a --data-dir data --output-dir outputs/primary --save-labels

# All nine complete scenes; requires substantial RAM/GPU memory
python scripts/run_main_experiments.py --all --data-dir data --output-dir outputs/primary --save-labels
```

These commands compute new executions. They do not overwrite the curated
`results/` evidence unless a user explicitly chooses that output location.

## Reproducing baselines and diagnostics

```bash
# Four true-K classical baselines, five declared seeds
python scripts/run_baselines.py classical --dataset salinas_a --data-dir data --output-dir outputs/baselines

# Class-count-free HDBSCAN and same-graph Leiden
python scripts/run_baselines.py no-k --datasets salinas_a --data-dir data --output-dir outputs/no_k

# Pavia University component ablation
python scripts/run_ablations.py --data-dir data --output-dir outputs/ablation

# Rebuild result tables/figures from curated CSVs; no model rerun
python scripts/export_tables.py
python scripts/generate_figures.py
```

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the information-access rules,
seed policies, evaluation mapping, automatic selector, and computational
scope. See [docs/RESULTS_MAP.md](docs/RESULTS_MAP.md) for the source of every
manuscript table and figure.

## Main reported evidence

The primary nine-scene Graph-RMS means are OA 0.6358, BA 0.5319, NMI 0.7245,
and ARI 0.6086. In the five-method classical protocol, Graph-RMS has the
highest NMI on all nine scenes. On the independent primary Trento holdout it
obtains OA 0.8533, BA 0.6774, NMI 0.8495, and ARI 0.9447. These are preserved
results, not values regenerated while assembling this repository.

The separate no-class-count comparison is more nuanced. Graph-RMS exceeds the
declared fixed-default HDBSCAN configuration on all 36 scene-metric
comparisons, while Leiden is competitive and has nearly the same nine-scene
mean NMI. Report the absolute tables and limitations rather than only win
counts.

## Reproducibility levels

1. **Inspect immediately:** all curated CSV, JSON, NPY, PNG, configuration,
   and provenance files are included.
2. **Rebuild analysis products:** tables, statistics, and supported figures can
   be regenerated from curated outputs without downloading raw cubes.
3. **Rerun experiments:** requires the public datasets and appropriate compute.
   The largest profiled scene used about 14.9 GB peak allocated GPU memory.

The publication workflow figure is provided as the vector asset
`figures/graph.pdf`; the earlier PNG is retained as a preview. The
author-supplied bibliography is preserved as
`manuscript_support/references.bib` (with a root copy for direct LaTeX builds).
The audit found complete citation-key coverage, no duplicate BibTeX keys, and a
valid vector PDF with embedded fonts.

## Citation, licence, and archival status

Citation metadata are provided in `CITATION.cff` and `.zenodo.json` for
`https://github.com/Simhaatt/Graph-RMS`. The software is released under the MIT
License. Release `v1.0.1` is the current corrective reproducibility release;
the stable all-versions concept DOI is
`https://doi.org/10.5281/zenodo.21876447`. Release `v1.0.0` remains preserved at
`https://doi.org/10.5281/zenodo.21876448`. See
[docs/RELEASE.md](docs/RELEASE.md).
