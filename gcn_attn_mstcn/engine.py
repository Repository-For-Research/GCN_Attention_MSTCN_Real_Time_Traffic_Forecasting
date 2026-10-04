import json
from pathlib import Path

import numpy as np
import torch

from .metrics import compute_metrics, masked_mae_loss


@torch.no_grad()
def evaluate(model, loader, adj, scaler, device, mape_min_speed=3.0):
    """Return (overall_metrics, per_horizon_metrics) in mph, DCRNN-style."""
    model.eval()
    preds, labels = [], []
    for x, _, y_raw in loader:
        pred = model(x.to(device), adj).cpu().numpy()            # (B, T_out, N) scaled
        preds.append(scaler.inverse_transform(pred))
        labels.append(y_raw.numpy())                              # raw mph, zeros = missing
    preds, labels = np.concatenate(preds), np.concatenate(labels)

    overall = compute_metrics(preds.reshape(-1), labels.reshape(-1), mape_min_speed)
    per_horizon = []
    for h in range(preds.shape[1]):
        row = compute_metrics(preds[:, h].reshape(-1), labels[:, h].reshape(-1), mape_min_speed)
        row["horizon_min"] = (h + 1) * 5
        per_horizon.append(row)
    return overall, per_horizon


def fit(model, bundle, adj, device, cfg, save_dir):
    tcfg = cfg["train"]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=tcfg["lr_factor"], patience=tcfg["lr_patience"])

    best_r2, bad_epochs, history = float("-inf"), 0, []
    ckpt = save_dir / "best_model.pth"

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for x, y_scaled, y_raw in bundle.train:
            x, y_scaled, y_raw = x.to(device), y_scaled.to(device), y_raw.to(device)
            optimizer.zero_grad()
            loss = masked_mae_loss(model(x, adj), y_scaled, y_raw != 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        val, _ = evaluate(model, bundle.val, adj, bundle.scaler, device, cfg["eval"]["mape_min_speed"])
        history.append({"epoch": epoch, "train_loss": total / n, **{f"val_{k}": v for k, v in val.items()}})
        print(f"Epoch {epoch:02d} | train MAE(scaled) {total / n:.4f} | "
              f"val MAE {val['mae']:.3f} RMSE {val['rmse']:.3f} "
              f"MAPE {val['mape']:.2f}% R2 {val['r2']:.4f}")

        if val["r2"] > best_r2:
            best_r2, bad_epochs = val["r2"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                print("Early stopping.")
                break
        scheduler.step(val["r2"])

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, history
