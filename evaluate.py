"""Evaluate a saved checkpoint on the METR-LA test split.

    python evaluate.py --config configs/metr_la.yaml --checkpoint checkpoints/seed42/best_model.pth
"""
import argparse

import pandas as pd
import torch

from gcn_attn_mstcn.data import build_dataloaders, load_adjacency, normalize_adj
from gcn_attn_mstcn.engine import evaluate
from gcn_attn_mstcn.models import GCNAttnMSTCN
from gcn_attn_mstcn.utils import get_device, load_config, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/metr_la.yaml")
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    bundle = build_dataloaders(cfg)
    _, _, adj_np = load_adjacency(cfg["data"]["adj_path"])
    adj = normalize_adj(torch.tensor(adj_np, device=device))

    m = cfg["model"]
    model = GCNAttnMSTCN(bundle.num_features, m["hidden"], m["tcn_layers"],
                         cfg["data"]["output_len"], m["dropout"], m["num_heads"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    overall, per_h = evaluate(model, bundle.test, adj, bundle.scaler, device, cfg["eval"]["mape_min_speed"])
    print({k: round(v, 4) for k, v in overall.items()})
    print(pd.DataFrame(per_h)[["horizon_min", "mae", "rmse", "mape", "r2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
