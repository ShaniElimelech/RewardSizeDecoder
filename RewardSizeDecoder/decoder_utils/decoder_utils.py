from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
from collections import defaultdict
import json, os, re
from pathlib import Path


def compute_eval_metrics(y_pred, y_true):
    """Compute accuracy, precision, recall, F1 score, and specificity using confusion matrix."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Avoid division by zero
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity
    }


def compute_roc(y_probs, y_true):
    """Compute ROC and AUC."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc_score
    }


def confusion_indexes(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]

    return {
        "tn": tn_idx.tolist(),
        "fp": fp_idx.tolist(),
        "fn": fn_idx.tolist(),
        "tp": tp_idx.tolist()
    }


def separate_video_activity(k_pc_video, clf_trial_idx):
    separated_activity = defaultdict(list)
    separated_activity = {k: k_pc_video[v] for k, v in clf_trial_idx.items()}
    return separated_activity


def binarize_target(y, pos_label):
    y = np.asarray(y)
    return np.array(y == pos_label).astype(int)


def _safe_folder_name(name: str) -> str:
    """
    Return a filesystem-safe folder name. Replaces illegal characters and
    collapses whitespace. If the result becomes empty, fallback to 'results'.
    """
    s = str(name).strip()
    # Remove characters invalid on Windows/macOS/Linux and control chars
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', '-', s)
    # Collapse repeated dashes and whitespace prettily
    s = re.sub(r'\s+', ' ', s).strip()
    return s or "my_run"


def _jsonable(x):
    """Make values JSON-serializable (handles numpy types/arrays)."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def to_params_dict(self, exclude=("subject_id", "session", "dj_info", "user_model_params")) -> dict:
    """Return a flat dict of attributes, excluding some keys."""
    params = {k: v for k, v in vars(self).items() if k not in set(exclude)}
    # Make JSON-safe for the JSON file (Excel doesn't care)
    return {k: self._jsonable(v) for k, v in params.items()}


def save_parameters(self,
                    fmt=("json", "excel"),
                    filename_stem="decoder_params",
                    exclude=("subject_id", "session", "dj_info", "user_model_params")):
    """Save parameters as JSON and/or Excel"""

    root = Path(self.saveroot)
    if isinstance(fmt, str):
        fmt = (fmt,)
    fmt = tuple(x.lower() for x in fmt)

    params = self.to_params_dict(exclude=exclude)
    saved = {}

    # JSON
    if "json" in fmt:
        p = root / f"{filename_stem}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        saved["json"] = str(p)

    # Excel (single row from the flat dict)
    if "excel" in fmt:
        try:
            import pandas as pd
            p = root / f"{filename_stem}.xlsx"
            # If some values are lists/arrays, pandas will write them as objects;
            # optionally stringify sequences for nicer cells:
            row = {
                k: (json.dumps(self._jsonable(v), ensure_ascii=False)
                    if isinstance(v, (list, tuple, np.ndarray)) else v)
                for k, v in params.items()
            }
            pd.DataFrame([row]).to_excel(p, index=False)
            saved["excel"] = str(p)
        except ImportError:
            # Fallback to CSV
            p = root / f"{filename_stem}.csv"
            with open(p, "w", encoding="utf-8") as f:
                for k, v in params.items():
                    if isinstance(v, (list, tuple, np.ndarray)):
                        v = json.dumps(self._jsonable(v), ensure_ascii=False)
                    f.write(f"{k},{v}\n")
            saved["excel_fallback_csv"] = str(p)

    return saved










