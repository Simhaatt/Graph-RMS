# Dataset registry and acquisition record

Graph-RMS was evaluated on nine complete public hyperspectral scenes. The
dimensions below are verified against the frozen registry and manuscript.
Reference class count and labelled pixels are evaluation metadata; the primary
Graph-RMS processing path does not receive the class count.

| Dataset | Shape | Bands | Classes | Labelled pixels | Labelled fraction | Role |
|---|---:|---:|---:|---:|---:|---|
| Salinas-A | 83 x 86 | 204 | 6 | 5,348 | 74.92% | development |
| Indian Pines | 145 x 145 | 200 | 16 | 10,249 | 48.75% | development |
| KSC | 512 x 614 | 176 | 13 | 5,211 | 1.66% | development |
| Pavia University | 610 x 340 | 103 | 9 | 42,776 | 20.62% | development |
| WHU-Hi LongKou | 550 x 400 | 270 | 9 | 204,542 | 92.97% | development |
| WHU-Hi HongHu | 940 x 475 | 270 | 22 | 386,693 | 86.61% | development |
| WHU-Hi HanChuan | 1217 x 303 | 274 | 16 | 257,530 | 69.84% | development |
| Botswana | 1476 x 256 | 145 | 14 | 3,248 | 0.86% | development |
| Trento | 166 x 600 | 63 | 6 | 30,214 | 30.34% | independent primary holdout |

## Loader sources used in the frozen code

The loader records the exact acquisition endpoints used during the study:

- Salinas-A: `danaroth/salinas` dataset mirror on Hugging Face.
- Indian Pines: `danaroth/indian_pines` dataset mirror on Hugging Face.
- Pavia University: `danaroth/pavia` dataset mirror on Hugging Face.
- KSC and Botswana: `Tanishq165/HSI_Datasets` dataset mirror on Hugging Face.
- WHU-Hi scenes: `danaroth/whu_hi` dataset mirror on Hugging Face.
- Trento: files `Italy_hsi.mat` and `allgrd.mat` in the public
  `tyust-dayu/Trento` GitHub repository.

These are reproducibility endpoints, not assertions that a mirror controls the
original dataset rights. The release does not redistribute the files.

## Original-provider and licensing checklist

Before public deposition, an author should verify and record for each dataset:

1. the original institutional or project landing page;
2. the preferred citation requested by the provider;
3. the dataset licence or explicit terms of use;
4. whether redistribution of the raw cube and ground truth is prohibited;
5. the download date and, where practical, a checksum.

The supplied evidence did not contain authoritative licence texts for all nine
datasets. Therefore no dataset licence is guessed here. This is a manual FAIR
metadata task, not a computational blocker for inspecting the included derived
outputs.

## Preprocessing used by Graph-RMS

- All pixels in the complete cube are retained.
- Each spectral band is standardized over the complete scene.
- PCA-20 is used for local-neighbour ranking, not for full-spectrum affinity
  weighting or the initial diffusion state.
- No labelled-pixel subset is used to estimate preprocessing parameters.
- Evaluation is restricted to pixels whose reference label is nonzero.

