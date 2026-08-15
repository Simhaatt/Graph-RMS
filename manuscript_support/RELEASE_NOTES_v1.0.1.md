# Graph-RMS v1.0.1

This corrective reproducibility release preserves all v1.0.0 primary results
while repairing the public automatic-selector path and incorporating the
authors' current manuscript assets.

## Corrections and additions

- Restores the exact automatic-v2 component selector recovered from the frozen
  development rule and makes it the public automatic-selector implementation.
- Adds deterministic archive replay over all nine stored endpoint caches. The
  replay reproduces every selected endpoint exactly and obtains partition ARI
  1.0 for all eight development scenes with archived label arrays.
- Adds an independent development-calibration replay that exactly recovers the
  frozen global thresholds and all recorded aggregate metrics.
- Adds regression tests and strict audit checks for the rule hash, selection
  metadata, archived partitions, calibration result, and public wrapper.
- Corrects automatic-selector configuration metadata and clarifies that the
  earlier runner is retained only as the archived v1 candidate generator.
- Adds the current manuscript source, bibliography, vector workflow figure,
  and supplied robustness/repeatability figure assets.
- Corrects implementation documentation for graph diffusion, deterministic
  affinity bandwidth estimation, evaluation, and licensing metadata.

## Scientific scope

No primary Graph-RMS partitions or headline scene-level scores were changed.
Trento remains the independent holdout for the primary procedure. Its
automatic-v2 result is still identified as retrospective because earlier
Trento scores were already known before that selector audit.

The MIT License applies to the software. Original third-party hyperspectral
datasets are not redistributed.
