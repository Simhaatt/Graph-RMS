# Graph-RMS v1.0.0 — reproducibility package for training-free and class-count-free HSI region discovery

Initial reproducibility release supporting the Graph-RMS manuscript.

## Included

- core sparse spectral-spatial graph and damped diffusion implementation;
- fine-mode extraction, spatial cleanup, and reciprocal prototype consolidation;
- frozen primary and automatic-selector configurations;
- nine curated full-scene partitions and endpoint curves;
- four classical true-K baseline summaries;
- fixed-default HDBSCAN and same-graph Leiden no-K results;
- ablation, sensitivity, selector, granularity, robustness, bootstrap,
  cross-device, runtime, and memory evidence;
- manuscript-linked figures and a complete table/figure results map;
- environment, checksums, manifests, FAIR documentation, and automated audits.

## Not included

- original third-party hyperspectral cubes or reference maps;
- superseded exploratory runs and notebook caches;
- adapted modern-method outputs as primary superiority evidence.

## Important interpretation notes

Graph-RMS discovers structural spectral-spatial regions and does not estimate
the semantic class count. Trento is the independent holdout for the primary
procedure. The automatic Trento evaluation is retrospective because the
primary Trento result was already known. Leiden is a competitive same-graph
alternative and the results do not support universal superiority claims.

## Licence and archive

The software is released under the MIT License. The canonical source
repository is `https://github.com/Simhaatt/Graph-RMS`; the version DOI will be
assigned when GitHub release `v1.0.0` is archived by Zenodo.
