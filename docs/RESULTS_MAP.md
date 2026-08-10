# Manuscript results map

This map links every numbered manuscript table and figure in the supplied
LaTeX source to the curated evidence. `Verified` means the rounded manuscript
values were checked against the stored CSV/JSON records.

## Tables

| Manuscript label | Content | Primary repository source | Status and notes |
|---|---|---|---|
| `tab:dataset_overview` | Nine scene dimensions, bands, classes, labelled support, role | `results/main_tables/dataset_registry.csv` | Verified. Trento is the primary holdout. |
| `tab:frozen_pipeline` | Shared and scene-specific primary settings | `configs/primary.yaml`; `results/provenance/frozen_results_registry.json` | Verified for all nine scenes. |
| `tab:primary_results` | Full/support region counts and Graph-RMS OA/BA/NMI/ARI | `results/main_tables/master_results_tidy.csv`; `results/supporting_results/oversegmentation.csv` | Verified. Nine-scene means 0.6358/0.5319/0.7245/0.6086. |
| `tab:classical_complete` | Graph-RMS plus four true-K classical baselines | `results/main_tables/master_results_tidy.csv` | Verified. Baseline entries are five-seed means. |
| `tab:classical_statistics` | Average ranks, Friedman, Holm-adjusted Wilcoxon | `results/main_tables/classical_average_ranks.csv`; `friedman.csv`; `wilcoxon.csv` | Verified. Inferential family excludes no-K and adapted modern methods. |
| `tab:no_k_complete` | Graph-RMS, Leiden, and fixed-default HDBSCAN | `results/main_tables/no_k_aggregate.csv`; Graph-RMS rows in `master_results_tidy.csv` | Verified. Leiden has five evaluation seeds; HDBSCAN one declared run. |
| `tab:no_k_statistics` | Win counts and exact paired Wilcoxon summaries | `results/supporting_results/no_k_per_run.csv`; `results/main_tables/no_k_aggregate.csv` | Verified against manuscript values. |
| `tab:trento_configuration` | Frozen primary Trento settings and region counts | `results/per_dataset/trento/summary.json`; `configs/primary.yaml` | Verified: T100, radius 0.20, tau 0.75, beta 0.5, 213 fine, 68 full, 41 on support. |
| `tab:trento_comparison` | Trento metrics, bootstrap CI, SLIC-KMeans, Leiden, HDBSCAN | `master_results_tidy.csv`; `bootstrap_ci.csv`; `no_k_aggregate.csv` | Verified. Bootstrap block size 16, 300 replicates. |
| `tab:trento_diagnostics` | AMI, homogeneity, completeness, purity, class coverage | `results/supporting_results/graph_rms_information_metrics.csv`; `oversegmentation.csv` | Verified. |
| `tab:region_granularity` | K, full/support region counts and ratios | `results/supporting_results/oversegmentation.csv` | Verified. |
| `tab:granularity_controls` | Trento post-hoc K=6 and count-matched MiniBatch-KMeans | `results/supporting_results/oracle_k.csv`; `count_matched_minibatch_aggregate.csv`; `graph_rms_information_metrics.csv` | Verified. Count-matched dashes reflect fields omitted from the stored summary. |
| `tab:ablation` | Eight Pavia component variants | `results/supporting_results/ablation_summary.csv`; `ablation_runs_raw.csv` | Verified, including 189,178 regions without fine grouping. |
| `tab:sensitivity_summary` | Selected findings from 51 runs | `results/supporting_results/sensitivity_runs_raw.csv`; `sensitivity_summary.csv` | Verified. The full raw grid is retained. |
| `tab:selector_audit` | Mean oracle gaps and rule-ablation consequences | `results/supporting_results/selector_oracle_gaps.csv`; `selector_rule_ablation.csv` | Verified. |
| `tab:auto_selector_summary` | Eight-development-scene primary versus automatic means | `results/automatic_selector/development_master_table.csv`; `development_rule_search.csv`; `development_lock.json` | Verified. Global calibration used development-scene NMI/ARI; deployment did not use target labels/K. |
| `tab:perturbation_results` | Salinas-A, Indian Pines, and adaptive Pavia perturbations | `results/supporting_results/robustness_perturbations.csv`; `perturbation_runs_frozen_threshold.csv`; `pavia_adaptive_robustness.csv` | Verified. Pavia pixel-replacement trials are abstentions. |
| `tab:pavia_partition_agreement` | Pavia perturbation ARI versus frozen partition | `results/supporting_results/pavia_adaptive_robustness.csv` | Verified. |
| `tab:cross_device_repeatability` | CPU/GPU cluster counts and partition ARI | `results/supporting_results/cross_device_repeatability.csv` | Verified. |
| `tab:runtime_memory_complete` | Three-run runtime and peak process/GPU memory | `results/supporting_results/runtime_memory.csv`; `profile_runs_raw.csv`; `results/provenance/environment.json` | Verified. Hardware A100-SXM4-80GB. |

## Figures

| Manuscript filename | Curated file | Numerical/data source | Status |
|---|---|---|---|
| `figures/graph.pdf` | `figures/graph.pdf` | Author-supplied workflow artwork; method equations and `configs/primary.yaml` | Present; one-page vector PDF with embedded fonts. |
| `figures/fig2_representative_cluster_maps.png` | `figures/fig2_representative_cluster_maps.png` | Frozen NPY partitions for KSC, HongHu, Trento; reference maps required only for regeneration | Present; copied without visual redesign from the supplied evidence figure. |
| `figures/fig3_fragmentation.png` | `figures/fig3_fragmentation.png` | `results/supporting_results/oversegmentation.csv` | Present. |
| `figures/fig4_sensitivity.png` | `figures/fig4_sensitivity.png` | `results/supporting_results/sensitivity_runs_raw.csv` | Present. |
| `figures/repeatability_and_perturbation_matched_style.png` | same path under `figures/` | `cross_device_repeatability.csv`; `pavia_adaptive_robustness.csv` | Present. |
| `figures/fig6_runtime_memory.png` | `figures/fig6_runtime_memory.png` | `runtime_memory.csv` | Present. |

## Per-scene selected partitions

Every primary scene has:

```text
results/per_dataset/<dataset>/summary.json
results/per_dataset/<dataset>/endpoint_curve.csv
results/per_dataset/<dataset>/selected_labels.npy
```

NPY arrays contain the complete scene partition in the original H x W shape.
The portable registry points to these summaries. The original evidence source
paths and checksums remain documented in the provenance files.

## Interpretation boundaries

- `master_results_tidy.csv` also preserves adapted DLSS/S2DL outputs. Those
  rows are historical diagnostics and are not used in the manuscript's main
  superiority or inferential claims.
- The no-K table must remain separate from the true-K classical table.
- The automatic Trento audit must remain separate from the independent primary
  Trento holdout result.
- Full-scene region counts must not be replaced with labelled-support counts.
