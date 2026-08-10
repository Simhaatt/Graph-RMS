# Excluded and large-file inventory

The clean release intentionally excludes workspace artifacts that are not
needed to inspect the reported evidence or rerun from source.

| Workspace artifact | Approximate size | Release treatment | Reason |
|---|---:|---|---|
| `gpu_results.zip` | 1.4 GB | excluded | historical raw bundle; duplicates selected evidence |
| `whu_hi_honghu_data.zip` | 483 MB | excluded | third-party dataset archive |
| `PaviaC_data.mat` | 160 MB | excluded | third-party raw cube, not one of the nine registered filenames |
| public cubes under `data/` | several GB combined | excluded | third-party terms and repository size |
| old Colab/package ZIPs | many files | excluded | historical transport artifacts |
| exploratory output directories | variable | excluded | superseded by frozen registry/evidence package |
| external baseline source trees | large/third-party | excluded | separate upstream licensing and non-primary status |
| notebook caches, logs, `__pycache__` | variable | excluded | temporary/nonportable |

Included NPY files are curated derived partitions for the nine primary scenes
and automatic-selector audit. They are compact compared with the raw cubes and
are necessary for independent analysis of the reported partitions.

