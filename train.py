"""Train GCN-Attn-MSTCN on METR-LA and report test metrics.

    python train.py --config configs/metr_la.yaml
"""
import argparse
import json

import pandas as pd
import torch

from gcn_attn_mstcn.data import build_dataloaders, load_adjacency, normalize_adj
from gcn_attn_mstcn.engine import evaluate, fit
from gcn_attn_mstcn.models import GCNAttnMSTCN
from gcn_attn_mstcn.utils import ensure_dir, get_device, load_config, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/metr_la.yaml")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(seed)
    device = get_device(cfg["device"])
    print("Device:", device, "| seed:", seed)

    bundle = build_dataloaders(cfg)
    _, _, adj_np = load_adjacency(cfg["data"]["adj_path"])
    adj = normalize_adj(torch.tensor(adj_np, device=device))

    m = cfg["model"]
    model = GCNAttnMSTCN(node_features=bundle.num_features, hidden=m["hidden"],
                         tcn_layers=m["tcn_layers"], out_timesteps=cfg["data"]["output_len"],
                         dropout=m["dropout"], num_heads=m["num_heads"]).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = f"{cfg['output']['checkpoint_dir']}/seed{seed}"
    model, _ = fit(model, bundle, adj, device, cfg, ckpt_dir)

    overall, per_h = evaluate(model, bundle.test, adj, bundle.scaler, device, cfg["eval"]["mape_min_speed"])
    print("\n=== TEST (12 horizons pooled) ===")
    print({k: round(v, 4) for k, v in overall.items()})
    table = pd.DataFrame(per_h)[["horizon_min", "mae", "rmse", "mape", "r2"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = ensure_dir(f"{cfg['output']['results_dir']}/seed{seed}")
    table.to_csv(out / "test_per_horizon.csv", index=False)
    with open(out / "test_overall.json", "w") as f:
        json.dump(overall, f, indent=2)


if __name__ == "__main__":
    main()
