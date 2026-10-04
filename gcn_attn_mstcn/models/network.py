import torch
import torch.nn as nn

from .attention import GraphAttention
from .gcn import GCNBlock
from .mstcn import MSTCNBlock


class GCNAttnMSTCN(nn.Module):
    """GCN -> graph attention -> Transformer-style FFN (residual + LayerNorm)
    -> multi-scale TCN -> GRU decoder -> MLP head (all horizons at once).

    Input : (B, T_in, N, F) with F = speed, time-in-day, day-in-week
    Output: (B, T_out, N)   scaled speed predictions
    """

    def __init__(self, node_features=3, hidden=64, tcn_layers=3,
                 out_timesteps=12, dropout=0.2, num_heads=4):
        super().__init__()
        self.hidden = hidden
        self.gcn = GCNBlock(node_features, hidden, dropout)
        self.spatial_att = GraphAttention(hidden, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(),
                                 nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)

        self.tcn_layers = nn.ModuleList([MSTCNBlock(hidden, dropout) for _ in range(tcn_layers)])
        self.decoder_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, out_timesteps),
        )

    def spatial_block(self, x_t, adj):
        """Spatial encoder for one time step: (B, N, F) -> (B, N, hidden)."""
        h = self.gcn(x_t, adj)
        h = self.norm1(h + self.dropout(self.spatial_att(h, adj)))
        return self.norm2(h + self.ffn(h))

    def forward(self, x, adj):
        B, T, N, _ = x.shape
        h = torch.stack([self.spatial_block(x[:, t], adj) for t in range(T)], dim=1)  # (B,T,N,C)
        h = h.permute(0, 2, 3, 1).reshape(B * N, self.hidden, T)

        for layer in self.tcn_layers:
            h = layer(h)

        _, h_last = self.decoder_gru(h.permute(0, 2, 1))
        out = self.fc_out(h_last.squeeze(0)).view(B, N, -1).permute(0, 2, 1)
        return out
