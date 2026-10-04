import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """H = A_hat X W + b (single-hop graph convolution)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)                      # (B, N, out)
        out = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), support)
        return out + self.bias if self.bias is not None else out


class GCNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gcn = GCNLayer(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.dropout(self.act(self.gcn(x, adj)))
