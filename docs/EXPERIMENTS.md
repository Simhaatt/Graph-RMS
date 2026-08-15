# Experimental protocol

## 1. Primary Graph-RMS procedure

For a hyperspectral cube with (N) pixels and (B) bands, Graph-RMS applies:

1. per-band full-scene z-score standardization;
2. PCA-20 only for ranking local neighbours;
3. reciprocal local graph construction inside a radius-7 spatial window,
   retaining at most 20 candidates and repairing degree below four;
4. Gaussian spectral-spatial edge affinities using median edge-distance
   bandwidths;
5. row-normalized damped diffusion
   (Y^{(t+1)}=0.5Y^{(t)}+0.5WY^{(t)});
6. PCA to at most ten components in the checkpoint representation followed by
   population-standardization of each retained component;
7. capped 50-neighbour radius connectivity to obtain fine modes;
8. radius-3 spatial majority cleanup, retaining the centre label on ties;
9. reciprocal prototype consolidation over at most 30 candidate neighbours;
10. reference-free endpoint guards for stability, largest-region fraction,
    normalized entropy, and compression.

The mode dispersion is

\[
s_a^{\rm raw}=\sqrt{\frac{1}{n_a}\sum_{i\in a}\lVert z_i-\mu_a\rVert_2^2}.
\]

This is an RMS Euclidean radius in standardized diffusion-PCA space. It is not
the expansion of the method acronym. The method acronym denotes **Robust Mean
Shift**.

The floor is calculated once from initial fine modes whose raw dispersion is
above (10^{-6}) and size is at least five. It is the 25th percentile of that
set, or (10^{-3}) if the set is empty. Subsequent pooled updates use the
already floored stored dispersions, matching the frozen implementation.

All exact shared and scene-specific values are in `configs/primary.yaml`.

## 2. Endpoint selection

The primary candidate sequence includes the frozen threshold grid from zero
through 3.0 and both size exponents 0 and 0.5. A candidate endpoint is
admissible when the stored partition satisfies:

- adjacent-threshold ARI at least 0.95;
- largest-region fraction at most 0.40;
- normalized partition entropy at least 0.35; and
- compression of the cleaned fine partition at least 0.20.

The primary procedure selects an admissible size-aware endpoint when
supported; otherwise it uses the conservative beta-zero branch. LongKou is the
only primary scene using that conservative fallback. The selected partition is
frozen before loading its reference map.

## 3. Automatic operating-point experiment

Automatic-v2 separates global calibration from deployment:

- Global rule calibration used reference-based NMI and ARI on eight
  development scenes to choose one rule from a predeclared rule grid.
- Per-scene deployment of the locked rule uses the cube and reference-free
  partition statistics only. It receives neither target labels nor target
  (K).
- The local scale is the median positive finite tenth non-self-neighbour
  distance in standardized diffusion-PCA space.
- The full candidate grid and deterministic hierarchy are frozen in
  `configs/automatic_selector.yaml` and preserved in the development lock.
- `scripts/run_automatic_selector.py` executes the recovered automatic-v2
  component selector; `_run_automatic_selection.py` is retained only as the
  automatic-scale-v1 endpoint-surface generator.
- Seven development scenes selected the conservative branch; KSC selected the
  size-aware branch. No development scene abstained.
- Trento was already known when automatic-v2 was designed. Its automatic
  result is retrospective and must not be presented as a second independent
  holdout.

Eight-scene means changed from primary OA/BA/NMI/ARI
0.6086/0.5137/0.7089/0.5666 to automatic
0.5744/0.5384/0.7239/0.5813. The retrospective Trento automatic result used
(T=100), gamma 1.25, realized radius 0.1360, tau 0.75, and beta 0.5,
returning 67 regions and OA/BA/NMI/ARI
0.6148/0.5122/0.7656/0.7164.

## 4. Comparison groups and fairness

### True-class-count classical group

PCA-KMeans, MiniBatch-KMeans, fuzzy c-means, and SLIC-KMeans receive the
reference class count as the requested number of clusters. They use full-scene
band standardization and PCA-20 and are summarized using five fixed seeds.
These methods form the predefined five-method Friedman/Holm family together
with Graph-RMS.

### Class-count-free group

- HDBSCAN uses the declared scikit-learn defaults on standardized PCA-20.
  Noise remains label -1 during evaluation.
- Leiden operates on the exact frozen weighted Graph-RMS graph. Its resolution
  is chosen from a fixed grid using cross-seed and adjacent-resolution
  agreement plus largest-community and entropy guards. It uses five evaluation
  seeds.

HDBSCAN and Leiden receive neither true (K) nor labels for fitting or
selection. Their paired tests are reported separately from the true-K family.
Leiden is a strong same-graph control and is not uniformly inferior to
Graph-RMS.

### Excluded modern adaptation diagnostics

Stored DLSS and S2DL adaptation outputs are preserved in supporting evidence
for transparency but were excluded from superiority claims and inferential
tests because an author-native sanity anchor was not established. A fair
modern self-supervised comparison would require a separately standardized
training, patch, augmentation, and output-count protocol.

## 5. Evaluation

All methods partition every scene pixel. Metrics use only nonzero reference
pixels. OA and BA use one-to-one Hungarian assignment; unmatched predicted
regions count as errors. NMI and ARI are permutation-invariant and are the
principal structural-partition measures. Supplementary AMI, homogeneity,
completeness, V-measure, purity, and fragmentation diagnostics are also stored.

The full-scene region count and the count intersecting the labelled support are
kept separate. This distinction is essential for KSC and Botswana, whose
reference maps cover only 1.66% and 0.86% of their scenes.

## 6. Diagnostic studies

- **Ablation:** eight Pavia University configurations covering graph gate,
  spatial affinity, diffusion, cleanup, fine-mode grouping, consolidation, and
  size weighting.
- **Sensitivity:** 51 one-factor runs on Salinas-A, Indian Pines, and Pavia
  University over window radius, search PCA dimension, checkpoint, and fine
  radius.
- **Selector audit:** oracle gaps and removal of persistence, largest-region,
  entropy, and compression guards.
- **Perturbation:** two seeds each for 1% additive noise and 5% pixel
  replacement on three representative scenes; Pavia reselects its endpoint.
- **Cross-device:** frozen CPU versus new GPU partitions for KSC, HanChuan,
  and Botswana.
- **Uncertainty:** 300 spatial block-bootstrap replicates per scene and metric.
- **Granularity controls:** post hoc Trento consolidation to six groups and
  count-matched MiniBatch-KMeans using full-scene and labelled-support budgets.
- **Profiling:** three end-to-end A100 runs for each scene, recording runtime,
  process RSS, peak allocated GPU memory, and peak reserved GPU memory.

## 7. Compute expectations

Curated analysis products are immediately inspectable on CPU. Full scene reruns
require much more compute. The recorded A100 mean runtime ranges from 1.21 s
to 42.34 s, but preparation, downloads, environment setup, and repeated
diagnostic grids add substantial wall time. Peak allocated GPU memory reaches
14.9 GB on HongHu. CPU-only execution is supported but was not the hardware
profile reported in the main runtime table.
