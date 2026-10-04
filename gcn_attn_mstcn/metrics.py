"""Masked metrics. Zeros in METR-LA are missing readings and are excluded."""
import numpy as np
import torch
from sklearn.metrics import r2_score


def masked_mae_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """True masked MAE (training loss, computed on scaled values)."""
    mask = mask.float()
    loss = torch.abs(preds - labels) * mask
    return torch.nan_to_num(loss, nan=0.0).sum() / mask.sum().clamp(min=1.0)


def masked_mae(preds, labels, null_val=0.0):
    m = labels != null_val
    return float(np.abs(preds[m] - labels[m]).mean())


def masked_rmse(preds, labels, null_val=0.0):
    m = labels != null_val
    return float(np.sqrt(((preds[m] - labels[m]) ** 2).mean()))


def masked_mape(preds, labels, min_speed=3.0):
    """MAPE (%) on targets above `min_speed` mph (avoids tiny denominators)."""
    m = labels > min_speed
    return float(np.mean(np.abs((preds[m] - labels[m]) / labels[m])) * 100)


def masked_r2(preds, labels, null_val=0.0):
    m = labels != null_val
    return float(r2_score(labels[m], preds[m]))


def compute_metrics(preds, labels, mape_min_speed=3.0) -> dict:
    return {
        "mae": masked_mae(preds, labels),
        "rmse": masked_rmse(preds, labels),
        "mape": masked_mape(preds, labels, mape_min_speed),
        "r2": masked_r2(preds, labels),
    }
