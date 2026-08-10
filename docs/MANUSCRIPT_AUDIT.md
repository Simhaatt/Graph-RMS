# Manuscript and artifact audit

Audit date: 10 August 2026  
Supplied manuscript: `manuscript_support/manuscript_supplied.tex`  
Target named by the authors: *GIScience & Remote Sensing*

## Overall finding

The stored evidence supports the manuscript's principal numerical tables.
Primary, classical, no-K, holdout, granularity, ablation, sensitivity,
selector, robustness, cross-device, bootstrap, and runtime values are present
and consistent at the manuscript's displayed precision. No scientific result
was regenerated or altered while assembling this repository.

## Verified scientific records

- Nine complete scenes and their dimensions/support counts.
- Exact primary T, fine radius, tau, and beta settings for each scene.
- Nine frozen full-scene partitions and endpoint curves.
- Four classical true-K baselines with five-seed means.
- HDBSCAN and Leiden class-count-free comparisons across all nine scenes.
- Primary Trento holdout configuration, metrics, and 300-replicate block CIs.
- Pavia eight-variant ablation and 51-run three-scene sensitivity study.
- Selector oracle gaps and rule ablations.
- Perturbation and cross-device repeatability records.
- Three-run A100 runtime and memory profiles.
- Eight-scene automatic-v2 calibration records and retrospective Trento audit.

## Resolved supplied-asset checks

- The author-supplied `figures/graph.png` is present and decodes as a valid
  1693 x 929 RGB PNG.
- The author-supplied bibliography contains 35 unique entries. All 33 citation
  keys used by `manuscript_supplied.tex` are present; no duplicate keys or
  unbalanced braces were detected.

## Publication-identifier status

With author approval, the duplicate Data Availability section was removed and
the fictitious `zenodo.XXXXXXXX` DOI was deleted. The single remaining
statement names the public GitHub repository and the verified version-specific
Zenodo DOI `10.5281/zenodo.21876448` for release `v1.0.0`.
A replacement template is retained in
`manuscript_support/DATA_CODE_AVAILABILITY.tex`.

## Text-quality and consistency findings

- The scientific acronym is used correctly as Robust Mean Shift. Continue to
  distinguish it from the RMS Euclidean prototype radius.
- The automatic selector is correctly described as globally calibrated using
  eight development scenes and label-free at target-scene deployment. Its
  Trento result is correctly called retrospective.
- The manuscript contains duplicated comment headers around several sections;
  these do not change compiled content but can be cleaned before submission.
- A sentence in the fine-mode subsection appears duplicated/malformed in the
  supplied source around the explanation of the reduced representation. It
  should be copyedited against the authors' intended wording.
- The modern-method discussion is conservative and consistent with the stored
  validity audit: adapted DLSS/S2DL outputs are not used for superiority claims.

## Portability audit

Portable release scripts resolve paths from the repository root. A small
number of preserved historical provenance manifests contain their original
`/content/...`, Google Drive, or `D:\cvip\...` paths. They are retained as
execution evidence and are not consumed by the portable run scripts. The
nonportable historical evidence-builder script was excluded from this release.

## Large or excluded artifacts

Raw cubes, original download ZIPs, caches, notebook build products, external
baseline repositories, and old exploratory outputs are excluded. See
`docs/LARGE_FILES.md`. Curated CSV, JSON, NPY, and PNG evidence remains.

## Numerical caveats to preserve

- Do not claim Graph-RMS wins every metric or every scene.
- State that Leiden is competitive on the identical weighted graph.
- Qualify HDBSCAN conclusions as specific to the declared fixed-default run.
- Explain that OA/BA penalize structural fragmentation under one-to-one
  Hungarian matching.
- Do not describe the number of Graph-RMS regions as an estimate of semantic
  class count.
- Do not treat automatic Trento as a second untouched holdout.
- Do not interpret cross-device results as bitwise identity.
