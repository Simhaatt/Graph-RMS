"""Download and load hyperspectral scenes.

Two formats are supported:
  - WHU-Hi scenes (ENVI BSQ), from https://huggingface.co/datasets/danaroth/whu_hi
  - Classic .mat scenes (Salinas-A, Indian Pines), from the danaroth/*
    Hugging Face mirrors (small files, download in seconds, no xet hang).

All loaders return the same HSIScene: (H, W, B) float32 cube + (H, W) uint8
ground truth where 0 = unlabeled/background.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

REPO_ID = "danaroth/whu_hi"
SCENE_FILES = {
    "whu_hi_honghu": {
        "cube_bsq": "WHU-Hi-HongHu/WHU-Hi-HongHu.bsq",
        "cube_hdr": "WHU-Hi-HongHu/WHU-Hi-HongHu.hdr",
        "gt_bsq": "WHU-Hi-HongHu/WHU-Hi-HongHu_gt.bsq",
        "gt_hdr": "WHU-Hi-HongHu/WHU-Hi-HongHu_gt.hdr",
    },
    "whu_hi_longkou": {
        "cube_bsq": "WHU-Hi-LongKou/WHU-Hi-LongKou.bsq",
        "cube_hdr": "WHU-Hi-LongKou/WHU-Hi-LongKou.hdr",
        "gt_bsq": "WHU-Hi-LongKou/WHU-Hi-LongKou_gt.bsq",
        "gt_hdr": "WHU-Hi-LongKou/WHU-Hi-LongKou_gt.hdr",
    },
    "whu_hi_hanchuan": {
        "cube_bsq": "WHU-Hi-HanChuan/WHU-Hi-HanChuan.bsq",
        "cube_hdr": "WHU-Hi-HanChuan/WHU-Hi-HanChuan.hdr",
        "gt_bsq": "WHU-Hi-HanChuan/WHU-Hi-HanChuan_gt.bsq",
        "gt_hdr": "WHU-Hi-HanChuan/WHU-Hi-HanChuan_gt.hdr",
    },
}

CLASS_NAMES_HONGHU = [
    "Unclassified", "Red roof", "Road", "Bare soil", "Cotton",
    "Cotton firewood", "Rape", "Chinese cabbage", "Pakchoi", "Cabbage",
    "Tuber mustard", "Brassica parachinensis", "Brassica chinensis",
    "Small Brassica chinensis", "Lactuca sativa", "Celtuce",
    "Film covered lettuce", "Romaine lettuce", "Carrot", "White radish",
    "Garlic sprout", "Broad bean", "Tree",
]

# Classic .mat scenes: (repo_id, cube_file, cube_key, gt_file, gt_key).
# These are small enough to fetch directly on Colab without the Drive dance.
MAT_SCENES = {
    "salinas_a": ("danaroth/salinas", "SalinasA_corrected.mat", "salinasA_corrected",
                  "SalinasA_gt.mat", "salinasA_gt"),
    "salinas": ("danaroth/salinas", "Salinas_corrected.mat", "salinas_corrected",
                "Salinas_gt.mat", "salinas_gt"),
    "indian_pines": ("danaroth/indian_pines", "Indian_pines_corrected.mat", "indian_pines_corrected",
                     "Indian_pines_gt.mat", "indian_pines_gt"),
    "pavia_university": ("danaroth/pavia", "PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt"),
}

# Direct files from the UPV/EHU Computational Intelligence Group repository.
# These are reserved as untouched v2 validation scenes.
DIRECT_MAT_SCENES = {
    "ksc": {
        "repo_id": "Tanishq165/HSI_Datasets",
        "cube_file": "KSC/KSC_data.mat",
        "cube_key": None,
        "gt_file": "KSC/KSC_gt.mat",
        "gt_key": None,
    },
    "botswana": {
        "repo_id": "Tanishq165/HSI_Datasets",
        "cube_file": "Botswana/Botswana_data.mat",
        "cube_key": None,
        "gt_file": "Botswana/Botswana_gt.mat",
        "gt_key": None,
    },
    "houston13": {
        "repo_id": "Tanishq165/HSI_Datasets",
        "cube_file": "Houston13/houston13_data.mat",
        "cube_key": None,
        "gt_file": "Houston13/houston13_gt.mat",
        "gt_key": None,
    },
}

URL_MAT_SCENES = {
    "trento": {
        "cube_file": "TrentoRepo/Italy_hsi.mat",
        "cube_url": "https://raw.githubusercontent.com/tyust-dayu/Trento/main/Italy_hsi.mat",
        "cube_key": "data",
        "gt_file": "TrentoRepo/allgrd.mat",
        "gt_url": "https://raw.githubusercontent.com/tyust-dayu/Trento/main/allgrd.mat",
        "gt_key": "mask_test",
    },
}

_ENVI_DTYPES = {
    1: np.dtype("u1"), 2: np.dtype("<i2"), 3: np.dtype("<i4"),
    4: np.dtype("<f4"), 5: np.dtype("<f8"), 12: np.dtype("<u2"),
}


@dataclasses.dataclass
class HSIScene:
    cube: np.ndarray       # (H, W, B) float32
    gt: np.ndarray         # (H, W) uint8, 0 = unlabeled/background
    wavelengths: np.ndarray  # (B,) float32, nanometers
    class_names: list[str]
    name: str


def _parse_envi_header(hdr_path: Path) -> dict:
    text = hdr_path.read_text(errors="ignore")
    fields: dict[str, str] = {}
    key = None
    buf = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if key is not None:
            buf.append(line)
            if "}" in line:
                fields[key] = " ".join(buf)
                key = None
                buf = []
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip().lower(), v.strip()
        if v.startswith("{") and "}" not in v:
            key = k
            buf = [v]
        else:
            fields[k] = v
    return fields


def _read_envi_bsq(bsq_path: Path, hdr_path: Path) -> tuple[np.ndarray, dict]:
    fields = _parse_envi_header(hdr_path)
    lines = int(fields["lines"])
    samples = int(fields["samples"])
    bands = int(fields["bands"])
    dtype = _ENVI_DTYPES[int(fields["data type"])]
    offset = int(fields.get("header offset", 0))
    raw = np.fromfile(bsq_path, dtype=dtype, offset=offset)
    raw = raw[: bands * lines * samples]
    cube = raw.reshape(bands, lines, samples).transpose(1, 2, 0)  # (H, W, B)
    return cube, fields


def _download_scene_files(scene: str, data_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    files = SCENE_FILES[scene]
    paths = {}
    try:
        for key, rel_path in files.items():
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=rel_path,
                local_dir=str(data_dir),
            )
            paths[key] = Path(local)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{scene}' from huggingface.co/datasets/{REPO_ID}: {e}\n"
            f"Manual fallback: download these files and place them under {data_dir}:\n"
            + "\n".join(f"  https://huggingface.co/datasets/{REPO_ID}/resolve/main/{p}" for p in files.values())
        ) from e
    return paths


def _load_mat_scene(scene: str, data_dir: Path, subsample: int | None) -> HSIScene:
    from huggingface_hub import hf_hub_download
    from scipy.io import loadmat

    repo_id, cube_file, cube_key, gt_file, gt_key = MAT_SCENES[scene]
    try:
        cube_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=cube_file,
                                     local_dir=str(data_dir))
        gt_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=gt_file,
                                   local_dir=str(data_dir))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{scene}' from huggingface.co/datasets/{repo_id}: {e}\n"
            f"Manual fallback: download these and place them under {data_dir}:\n"
            f"  https://huggingface.co/datasets/{repo_id}/resolve/main/{cube_file}\n"
            f"  https://huggingface.co/datasets/{repo_id}/resolve/main/{gt_file}"
        ) from e

    cube = loadmat(cube_path)[cube_key].astype(np.float32)  # (H, W, B)
    gt = loadmat(gt_path)[gt_key].astype(np.uint8)          # (H, W)
    wavelengths = np.arange(cube.shape[-1], dtype=np.float32)  # .mat scenes ship no wavelength axis

    if subsample is not None:
        cube = cube[:subsample, :subsample]
        gt = gt[:subsample, :subsample]

    class_names = [str(i) for i in range(int(gt.max()) + 1)]
    return HSIScene(cube=cube, gt=gt, wavelengths=wavelengths, class_names=class_names, name=scene)


def _load_direct_mat_scene(scene: str, data_dir: Path, subsample: int | None) -> HSIScene:
    from huggingface_hub import hf_hub_download
    from scipy.io import loadmat

    spec = DIRECT_MAT_SCENES[scene]
    cube_path = data_dir / spec["cube_file"]
    gt_path = data_dir / spec["gt_file"]
    try:
        cube_path = Path(hf_hub_download(
            repo_id=spec["repo_id"], repo_type="dataset",
            filename=spec["cube_file"], local_dir=str(data_dir),
        ))
        gt_path = Path(hf_hub_download(
            repo_id=spec["repo_id"], repo_type="dataset",
            filename=spec["gt_file"], local_dir=str(data_dir),
        ))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download untouched validation scene '{scene}'. "
            f"Place {spec['cube_file']} and {spec['gt_file']} under {data_dir}."
        ) from exc
    def choose_array(path: Path, key: str | None, ndim: int) -> np.ndarray:
        arrays = loadmat(path)
        if key is not None:
            return arrays[key]
        candidates = [value for name, value in arrays.items()
                      if not name.startswith("__") and isinstance(value, np.ndarray)
                      and value.ndim == ndim]
        if len(candidates) != 1:
            raise ValueError(f"expected one {ndim}D array in {path}, found {len(candidates)}")
        return candidates[0]

    cube = choose_array(cube_path, spec["cube_key"], 3).astype(np.float32)
    gt = choose_array(gt_path, spec["gt_key"], 2).astype(np.uint8)
    cube = _align_cube_to_gt(cube, gt)
    if subsample is not None:
        cube = cube[:subsample, :subsample]
        gt = gt[:subsample, :subsample]
    wavelengths = np.arange(cube.shape[-1], dtype=np.float32)
    class_names = [str(i) for i in range(int(gt.max()) + 1)]
    return HSIScene(cube=cube, gt=gt, wavelengths=wavelengths,
                    class_names=class_names, name=scene)


def _load_url_mat_scene(scene: str, data_dir: Path, subsample: int | None) -> HSIScene:
    from scipy.io import loadmat
    from urllib.request import urlretrieve

    spec = URL_MAT_SCENES[scene]
    cube_path = data_dir / spec["cube_file"]
    gt_path = data_dir / spec["gt_file"]
    for path, url in ((cube_path, spec["cube_url"]), (gt_path, spec["gt_url"])):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urlretrieve(url, path)
            except Exception as exc:
                raise RuntimeError(f"Failed to download '{scene}' from {url}") from exc
    cube = loadmat(cube_path)[spec["cube_key"]].astype(np.float32)
    gt = loadmat(gt_path)[spec["gt_key"]].astype(np.uint8)
    cube = _align_cube_to_gt(cube, gt)
    if subsample is not None:
        cube = cube[:subsample, :subsample]
        gt = gt[:subsample, :subsample]
    wavelengths = np.arange(cube.shape[-1], dtype=np.float32)
    class_names = [str(i) for i in range(int(gt.max()) + 1)]
    return HSIScene(cube=cube, gt=gt, wavelengths=wavelengths,
                    class_names=class_names, name=scene)


def _align_cube_to_gt(cube: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Return a hyperspectral cube in (H, W, B) order.

    Some public MATLAB mirrors store cubes as (B, H, W), whereas the classic
    scene files use (H, W, B).  Ground-truth dimensions provide an unambiguous,
    label-value-independent way to identify the two spatial axes.
    """
    if cube.ndim != 3 or gt.ndim != 2:
        raise ValueError(f"expected a 3D cube and 2D ground truth, got {cube.shape} and {gt.shape}")

    permutations = []
    for h_axis in range(3):
        for w_axis in range(3):
            if h_axis == w_axis:
                continue
            if cube.shape[h_axis] == gt.shape[0] and cube.shape[w_axis] == gt.shape[1]:
                band_axis = next(axis for axis in range(3) if axis not in (h_axis, w_axis))
                permutations.append((h_axis, w_axis, band_axis))

    if not permutations:
        raise ValueError(
            f"cube spatial dimensions cannot be aligned with ground truth: "
            f"cube={cube.shape}, gt={gt.shape}"
        )
    if len(permutations) > 1:
        raise ValueError(
            f"cube axis order is ambiguous for ground truth: cube={cube.shape}, gt={gt.shape}"
        )
    return np.transpose(cube, permutations[0])


def load_scene(scene: str = "whu_hi_honghu", data_dir: str | Path = "data",
                subsample: int | None = None) -> HSIScene:
    """Load a hyperspectral scene, downloading via huggingface_hub if not cached.

    Args:
        scene: a WHU-Hi ENVI scene (SCENE_FILES) or a .mat scene (MAT_SCENES).
        data_dir: local cache directory.
        subsample: if set, crop to a subsample x subsample top-left window
            (fast local smoke tests without needing the full cube).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if scene in MAT_SCENES:
        return _load_mat_scene(scene, data_dir, subsample)

    if scene in DIRECT_MAT_SCENES:
        return _load_direct_mat_scene(scene, data_dir, subsample)

    if scene in URL_MAT_SCENES:
        return _load_url_mat_scene(scene, data_dir, subsample)

    if scene not in SCENE_FILES:
        raise ValueError(f"Unknown scene '{scene}', choose from "
                         f"{list(SCENE_FILES) + list(MAT_SCENES) + list(DIRECT_MAT_SCENES) + list(URL_MAT_SCENES)}")

    paths = _download_scene_files(scene, data_dir)

    cube, cube_fields = _read_envi_bsq(paths["cube_bsq"], paths["cube_hdr"])
    gt_cube, _ = _read_envi_bsq(paths["gt_bsq"], paths["gt_hdr"])
    gt = gt_cube[:, :, 0].astype(np.uint8)

    wl_str = cube_fields.get("wavelength", "")
    wl_str = wl_str.strip("{}")
    wavelengths = np.array([float(x) for x in wl_str.replace("\n", " ").split(",") if x.strip()],
                            dtype=np.float32)
    if wavelengths.size != cube.shape[-1]:
        wavelengths = np.arange(cube.shape[-1], dtype=np.float32)

    if subsample is not None:
        cube = cube[:subsample, :subsample]
        gt = gt[:subsample, :subsample]

    class_names = CLASS_NAMES_HONGHU if scene == "whu_hi_honghu" else [
        str(i) for i in range(int(gt.max()) + 1)
    ]

    return HSIScene(
        cube=cube.astype(np.float32),
        gt=gt,
        wavelengths=wavelengths,
        class_names=class_names,
        name=scene,
    )
