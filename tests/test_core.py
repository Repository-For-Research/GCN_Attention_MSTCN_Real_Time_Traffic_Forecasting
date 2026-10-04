import numpy as np
import torch

from gcn_attn_mstcn.data import normalize_adj
from gcn_attn_mstcn.metrics import compute_metrics, masked_mae_loss
from gcn_attn_mstcn.models import GCNAttnMSTCN


def test_forward_shape_and_attention():
    B, T, N, F = 2, 12, 20, 3
    model = GCNAttnMSTCN(node_features=F, hidden=64, out_timesteps=12).eval()
    adj = normalize_adj((torch.rand(N, N) > 0.7).float())
    x = torch.randn(B, T, N, F)
    assert model(x, adj).shape == (B, 12, N)

    h = model.gcn(x[:, -1], adj)
    _, attn = model.spatial_att(h, adj, return_attention=True)
    assert attn.shape == (B, 4, N, N)
    assert torch.allclose(attn.sum(-1), torch.ones(B, 4, N), atol=1e-4)


def test_masked_metrics_ignore_zeros():
    labels = np.array([10.0, 0.0, 20.0, 0.0])
    preds = np.array([12.0, 99.0, 18.0, 99.0])
    assert compute_metrics(preds, labels)["mae"] == 2.0


def test_masked_loss_ignores_masked_entries():
    p, y = torch.tensor([1.0, 5.0]), torch.tensor([0.0, 0.0])
    assert masked_mae_loss(p, y, torch.tensor([True, False])).item() == 1.0
