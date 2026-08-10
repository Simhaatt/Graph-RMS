# Dataset acquisition

Raw hyperspectral cubes and ground-truth maps are deliberately excluded from
this repository. They are third-party research datasets and retain the terms
set by their original providers.

The code loader expects the public mirror filenames below. Running
`scripts/download_data.py` calls the same loader used by the experiments and
stores files below this directory. Mirror availability does not establish a
redistribution licence; users must consult the original provider pages before
use.

| Dataset | Expected files used by the loader |
|---|---|
| Salinas-A | `SalinasA_corrected.mat`, `SalinasA_gt.mat` |
| Indian Pines | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` |
| KSC | `KSC/KSC_data.mat`, `KSC/KSC_gt.mat` |
| Pavia University | `PaviaU.mat`, `PaviaU_gt.mat` |
| WHU-Hi LongKou | `WHU-Hi-LongKou/WHU-Hi-LongKou.{bsq,hdr}` and `_gt.{bsq,hdr}` |
| WHU-Hi HongHu | `WHU-Hi-HongHu/WHU-Hi-HongHu.{bsq,hdr}` and `_gt.{bsq,hdr}` |
| WHU-Hi HanChuan | `WHU-Hi-HanChuan/WHU-Hi-HanChuan.{bsq,hdr}` and `_gt.{bsq,hdr}` |
| Botswana | `Botswana/Botswana_data.mat`, `Botswana/Botswana_gt.mat` |
| Trento | `TrentoRepo/Italy_hsi.mat`, `TrentoRepo/allgrd.mat` |

Examples:

```bash
python scripts/download_data.py --list
python scripts/download_data.py --dataset trento --data-dir data
python scripts/download_data.py --all --data-dir data
```

To work offline, place files using the paths above and run the environment and
shape checks. More detailed dimensions, class counts, provenance notes, and
licensing caveats are in `docs/DATASETS.md`.

