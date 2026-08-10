# Reproducibility guide

## Quick integrity check

```bash
python scripts/check_environment.py
python scripts/audit_results.py
pytest -q
```

The audit verifies repository structure, nine registered datasets, 63 rows in
the seven-method historical evidence table, 36 bootstrap rows, eight ablation
rows, 51 sensitivity rows, 27 profile rows, three cross-device rows, no-K
coverage, exact primary metrics, and selected manuscript values.

## Rebuild tables and supported figures without raw cubes

```bash
python scripts/export_tables.py
python scripts/generate_figures.py
```

Outputs are written under `outputs/derived/` by default, leaving curated
evidence untouched. The figure command assembles all six preserved manuscript
figures, including the author-supplied workflow diagram; it does not redesign
them.

## Rerun a frozen primary scene

```bash
python scripts/run_main_experiments.py \
  --datasets salinas_a \
  --data-dir data \
  --output-dir outputs/primary \
  --save-labels
```

Use `--cpu` to force CPU execution. To run all scenes, replace the dataset
argument with `--all`. A complete rerun downloads third-party data and can
require more than 15 GB of allocated GPU memory.

## Rerun classical baselines

```bash
python scripts/run_baselines.py classical \
  --dataset salinas_a \
  --methods pca_kmeans minibatch_kmeans slic_kmeans fcm \
  --seeds 0 1 2 3 4 \
  --data-dir data \
  --output-dir outputs/classical
```

The true class count is supplied to these baseline methods, matching the
manuscript protocol.

## Rerun class-count-free baselines

```bash
python scripts/run_baselines.py no-k \
  --datasets salinas_a \
  --methods leiden hdbscan \
  --data-dir data \
  --output-dir outputs/no_k
```

The no-K run is resumable and checkpoints expensive units. On large scenes,
HDBSCAN and Leiden can be substantially slower than Graph-RMS.

## Automatic selector

The automatic selector deliberately separates label-free selection from
reference evaluation. Do not combine these steps in a single custom script.

```bash
python scripts/run_automatic_selector.py select \
  --dataset salinas_a --data-dir data --output-root outputs/automatic

python scripts/run_automatic_selector.py evaluate \
  --dataset salinas_a --data-dir data --output-root outputs/automatic
```

Development locks must be completed before freezing a new protocol. The
included Trento audit is retrospective; rerunning it does not create a new
untouched holdout.

## Numerical repeatability

Graph-RMS is deterministic under a fixed computational path, but sparse
floating-point operations can alter small-region merge decisions across
devices. The stored CPU/GPU comparison gives all-pixel ARI 0.9960 for KSC,
0.9114 for HanChuan, and 0.9921 for Botswana. Expect partition-level
repeatability, not bitwise identity or identical region counts.

## Provenance

- `results/provenance/environment.json`: final profiling environment.
- `results/provenance/pip_freeze.txt`: full captured package list.
- `results/provenance/frozen_results_registry.json`: portable frozen registry.
- `results/provenance/package_file_checksums.csv`: source evidence checksums.
- `results/provenance/*manifest.json`: study-specific execution records.

Some preserved historical manifests contain their original Colab or Windows
paths. These fields are documentary and are not consumed by release scripts.
Portable scripts resolve paths relative to the repository root.
