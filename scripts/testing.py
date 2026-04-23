import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from data.data_processing import *
from models.M3T_2D_aggregated import M3T


# =========================================================
# Config
# =========================================================
SEED = 17
INPUT_SHAPE = (3, 128, 128, 128)   # (C, D, H, W)
BATCH_SIZE = 2
NUM_WORKERS = 2

ROOT_PATIENTS = r"path/to/breast_side_regions"
LABELS_FILE = r"path/to/breast_side_labels.xlsx"
SPLIT_CSV = r"path/to/train_test_splits.csv"

PATIENT_COL = "patient_id"
LABEL_COL = "label"
VOL_PATTERNS = {"pre": "pre", "post": "post1", "sub": "sub"}

CHECKPOINT_PATH = r"path/to/best_m3t.pt"

SAVE_DIR = r"results/test_results"
PREDICTIONS_CSV = os.path.join(SAVE_DIR, "test_predictions.csv")
PREDICTIONS_XLSX = os.path.join(SAVE_DIR, "test_predictions.xlsx")
METRICS_TXT = os.path.join(SAVE_DIR, "test_metrics.txt")
METRICS_XLSX = os.path.join(SAVE_DIR, "test_metrics.xlsx")


# =========================================================
# Utilities
# =========================================================
def set_all_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DatasetWithPatientID(Dataset):
    """
    Wraps the dataset so we can save patient IDs in predictions.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.patient_ids = base_dataset.patient_ids
        self.labels = base_dataset.labels
        self.label_map = getattr(base_dataset, "label_map", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        pid = self.patient_ids[idx]
        return x, y, pid


def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return 0.0


def compute_binary_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = safe_auc(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = recall

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# =========================================================
# Main
# =========================================================
def main():
    from multiprocessing import freeze_support
    import torch.multiprocessing as mp

    freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.makedirs(SAVE_DIR, exist_ok=True)
    set_all_seeds(SEED)

    # -----------------------------------------------------
    # Build datasets
    # -----------------------------------------------------
    train_ds, test_ds = build_tumor_datasets(
        root_dir=ROOT_PATIENTS,
        labels_path=LABELS_FILE,
        patient_col=PATIENT_COL,
        label_col=LABEL_COL,
        split_csv=SPLIT_CSV,
    )

    # Wrap test dataset to keep patient IDs
    test_ds = DatasetWithPatientID(test_ds)

    print(f"Test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    xb, yb, pidb = next(iter(test_loader))
    print(f"Sample batch shapes: x={tuple(xb.shape)}, y={tuple(yb.shape)}, expected x={INPUT_SHAPE}")
    if tuple(xb.shape[1:]) != INPUT_SHAPE:
        print(f"Warning: got input shape {tuple(xb.shape[1:])}, expected {INPUT_SHAPE}")

    # -----------------------------------------------------
    # Device and model
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = M3T(
        in_channels=3,
        n_classes=2,
        image_size=128,
        agg_mode="mean"
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print("Loaded checkpoint successfully.")
    print("Checkpoint epoch:", checkpoint.get("epoch", "NA"))
    print("Checkpoint val_auc:", checkpoint.get("val_auc", "NA"))

    # Reverse label map for readable names
    reverse_label_map = None
    if getattr(test_ds, "label_map", None) is not None:
        reverse_label_map = {v: k for k, v in test_ds.label_map.items()}

    # -----------------------------------------------------
    # Testing
    # -----------------------------------------------------
    all_patient_ids = []
    all_true = []
    all_pred = []
    all_prob_0 = []
    all_prob_1 = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, pid in tqdm(test_loader, desc="Testing", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            logits = model(x)
            loss = criterion(logits, y)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            total_loss += loss.item() * x.size(0)

            all_patient_ids.extend(list(pid))
            all_true.extend(y.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())
            all_prob_0.extend(probs[:, 0].cpu().numpy().tolist())
            all_prob_1.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob_1 = np.array(all_prob_1)

    test_loss = total_loss / max(1, len(test_loader.dataset))
    metrics = compute_binary_metrics(y_true, y_pred, y_prob_1)

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------
    rows = []
    for i in range(len(all_patient_ids)):
        row = {
            "patient_id": all_patient_ids[i],
            "true_label_id": int(all_true[i]),
            "pred_label_id": int(all_pred[i]),
            "prob_class_0": float(all_prob_0[i]),
            "prob_class_1": float(all_prob_1[i]),
        }

        if reverse_label_map is not None:
            row["true_label_name"] = str(reverse_label_map[int(all_true[i])])
            row["pred_label_name"] = str(reverse_label_map[int(all_pred[i])])

        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    pred_df.to_excel(PREDICTIONS_XLSX, index=False)

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------
    metrics_dict = {
        "test_loss": test_loss,
        **metrics
    }
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.to_excel(METRICS_XLSX, index=False)

    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"AUC: {metrics['auc']:.6f}\n")
        f.write(f"Precision: {metrics['precision']:.6f}\n")
        f.write(f"Recall / Sensitivity: {metrics['recall_sensitivity']:.6f}\n")
        f.write(f"Specificity: {metrics['specificity']:.6f}\n")
        f.write(f"F1 Score: {metrics['f1']:.6f}\n")
        f.write(f"TN: {metrics['tn']}\n")
        f.write(f"FP: {metrics['fp']}\n")
        f.write(f"FN: {metrics['fn']}\n")
        f.write(f"TP: {metrics['tp']}\n")

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------
    print("\n===== Test Results =====")
    print(f"Test Loss:   {test_loss:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall_sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"Confusion:   TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")

    print("\nSaved files:")
    print("Predictions CSV :", PREDICTIONS_CSV)
    print("Predictions XLSX:", PREDICTIONS_XLSX)
    print("Metrics TXT     :", METRICS_TXT)
    print("Metrics XLSX    :", METRICS_XLSX)


if __name__ == "__main__":
    main()
