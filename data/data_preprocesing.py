import os
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import torchio as tio
except ImportError as e:
    raise ImportError("Please install torchio first: pip install torchio") from e

Array4D = np.ndarray  # (C, D, H, W)

# ---------------------------------
# Utility functions
# ---------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_first_match(folder: str, key_substring: str) -> str:
    key = key_substring.lower()
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if key in name.lower():
            return path
    raise FileNotFoundError(f"Could not find a file with '{key_substring}' in {folder}")


def _read_volume(path: str) -> np.ndarray:
    """Return a numpy array of shape (D, H, W) as float32."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr


def _to_subject(pre: np.ndarray, post: np.ndarray, sub: np.ndarray) -> tio.Subject:
    # pre/post/sub are (D, H, W) from SimpleITK
    pre_t  = torch.from_numpy(pre).permute(2, 1, 0).unsqueeze(0).contiguous().float()   # (1, W, H, D)
    post_t = torch.from_numpy(post).permute(2, 1, 0).unsqueeze(0).contiguous().float()  # (1, W, H, D)
    sub_t  = torch.from_numpy(sub).permute(2, 1, 0).unsqueeze(0).contiguous().float()   # (1, W, H, D)

    return tio.Subject(
        pre=tio.ScalarImage(tensor=pre_t,  affine=np.eye(4)),
        post=tio.ScalarImage(tensor=post_t, affine=np.eye(4)),
        sub=tio.ScalarImage(tensor=sub_t,  affine=np.eye(4)),
    )



def _make_transforms(spatial_size: Tuple[int, int, int], augmentation: bool) -> tio.Transform:
    # spatial_size is (D, H, W); TorchIO expects (W, H, D)
    target_shape_tio = (spatial_size[2], spatial_size[1], spatial_size[0])
    resize = tio.Resize(target_shape=target_shape_tio)
    norm   = tio.RescaleIntensity(out_min_max=(0, 1))  # only for ScalarImage

    if not augmentation:
        return tio.Compose([resize, norm])

    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2)),
        # tio.RandomAffine(scales=(0.9, 1.1), degrees=10, isotropic=False, image_interpolation='linear', p=0.8),
        # tio.RandomElasticDeformation(num_control_points=5, max_displacement=2, p=0.3),
        resize,
        norm,
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.3),
        tio.RandomNoise(std=(0, 0.02), p=0.3),
    ])


def _read_labels_table(labels_path: str, patient_col: str, label_col: str, label_map: Dict[str, int] = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    # read
    if labels_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(labels_path)
    else:
        df = pd.read_csv(labels_path)
    if patient_col not in df.columns or label_col not in df.columns:
        raise KeyError(f"Missing required columns. Columns present: {df.columns.tolist()}")
    df = df[[patient_col, label_col]].dropna()
    df[patient_col] = df[patient_col].astype(str)

    # build or apply mapping to 0..K-1 using string forms to be robust
    if label_map is None:
        uniq = sorted(df[label_col].astype(str).unique().tolist())
        label_map = {v: i for i, v in enumerate(uniq)}
    df["__y__"] = df[label_col].astype(str).map(label_map)
    if df["__y__"].isna().any():
        bad = df[df["__y__"].isna()][[patient_col, label_col]].head()
        raise ValueError(f"Unmapped labels found; extend label_map. Examples:\n{bad}")

    labels = dict(zip(df[patient_col].values, df["__y__"].astype(int).values))
    return labels, label_map



# ---------------------------------
# Dataset
# ---------------------------------

class TumorDatasetTIO(Dataset):
    """Simple dataset that returns x of shape (3, D, H, W) and y in {0,1}.

    root_dir: path containing patient (or patient_side) subfolders
    labels: mapping of patient_id (or patient_id_side) to 0 or 1
    vol_patterns: substrings to locate files in each patient folder
    """

    def __init__(
        self,
        root_dir: str,
        labels: Dict[str, int],
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        vol_patterns: Optional[Dict[str, str]] = None,
        augmentation: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.labels = labels
        # Adjust default patterns to your file names:
        #   pre.nii.gz, post1.nii.gz, sub.nii.gz
        self.vol_patterns = vol_patterns or {"pre": "pre", "post": "post1", "sub": "sub"}
        self.transform = _make_transforms(spatial_size, augmentation)

        all_patients = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.patient_ids: List[str] = sorted([p for p in all_patients if p in labels])
        if not self.patient_ids:
            raise RuntimeError("No matching patient folders found for the provided labels")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int):
        pid = self.patient_ids[idx]
        folder = os.path.join(self.root_dir, pid)

        pre_path = _find_first_match(folder, self.vol_patterns["pre"])
        post_path = _find_first_match(folder, self.vol_patterns["post"])
        sub_path = _find_first_match(folder, self.vol_patterns["sub"])

        pre = _read_volume(pre_path)   # (D, H, W)
        post = _read_volume(post_path) # (D, H, W)
        sub = _read_volume(sub_path)   # (D, H, W)

        subject = _to_subject(pre, post, sub)
        subject = self.transform(subject)

        # TorchIO uses (C, W, H, D); we want (C, D, H, W)
        pre_t  = subject["pre"].data.permute(0, 3, 2, 1).contiguous()
        post_t = subject["post"].data.permute(0, 3, 2, 1).contiguous()
        sub_t  = subject["sub"].data.permute(0, 3, 2, 1).contiguous()

        x = torch.cat([pre_t, post_t, sub_t], dim=0).to(torch.float32)  # (3, D, H, W)
        y = torch.tensor(int(self.labels[pid]), dtype=torch.long)
        return x, y


# ---------------------------------
# Builders: datasets and dataloaders
# ---------------------------------

def build_tumor_datasets(
    root_dir: str,
    labels_path: str,
    patient_col: str = "patient_id",
    label_col: str = "label",
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    vol_patterns: Optional[Dict[str, str]] = None,
    split_csv: Optional[str] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[TumorDatasetTIO, TumorDatasetTIO]:
    """Return train and val datasets. If split_csv is provided, use it, else split here by patient.
    The split_csv, if provided, must have columns train_split and test_split with patient ids.
    """
    labels, label_map = _read_labels_table(labels_path, patient_col, label_col)


    patient_ids = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d in labels])
    if not patient_ids:
        raise RuntimeError("No matching patients found in data folder and labels file")

    if split_csv is not None and os.path.exists(split_csv):
        sp = pd.read_csv(split_csv)
        if "train_split" not in sp.columns or "test_split" not in sp.columns:
            raise KeyError("split CSV must contain columns 'train_split' and 'test_split'")
        train_ids = sp["train_split"].dropna().astype(str).tolist()
        val_ids   = sp["test_split"].dropna().astype(str).tolist()
        train_ids = [pid for pid in train_ids if pid in patient_ids]
        val_ids   = [pid for pid in val_ids   if pid in patient_ids]
    else:
        set_seed(seed)
        rnd = random.Random(seed)
        rnd.shuffle(patient_ids)
        n_val = max(1, int(round(len(patient_ids) * val_fraction)))
        val_ids = patient_ids[:n_val]
        train_ids = patient_ids[n_val:]

    # Build datasets
    train_labels = {pid: labels[pid] for pid in train_ids}
    val_labels   = {pid: labels[pid] for pid in val_ids}

    train_ds = TumorDatasetTIO(
        root_dir=root_dir,
        labels=train_labels,
        spatial_size=spatial_size,
        vol_patterns=vol_patterns,
        augmentation=True,
    )
    val_ds = TumorDatasetTIO(
        root_dir=root_dir,
        labels=val_labels,
        spatial_size=spatial_size,
        vol_patterns=vol_patterns,
        augmentation=False,
    )


    num_classes = len(label_map)
    train_ds.num_classes = num_classes
    val_ds.num_classes   = num_classes
    train_ds.label_map   = label_map
    val_ds.label_map     = label_map
    print(f"Label map: {label_map} | num_classes: {num_classes}")

    print(f"Train patients: {len(train_ds)}; Val patients: {len(val_ds)}")
    return train_ds, val_ds
