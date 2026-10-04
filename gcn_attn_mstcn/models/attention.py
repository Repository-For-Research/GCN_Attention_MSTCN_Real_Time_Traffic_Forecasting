import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):

    def __init__(self, in_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 alpha: float = 0.2, temperature: float = 0.5):
        super().__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.temperature = temperature

        self.W = nn.Linear(in_dim, in_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj, return_attention: bool = False, return_scores: bool = False):
        B, N, C = x.shape
        Wh = self.W(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,H,N,D)

        f_src = (Wh * self.a_src.view(1, self.num_heads, 1, -1)).sum(-1)
        f_dst = (Wh * self.a_dst.view(1, self.num_heads, 1, -1)).sum(-1)
        scores = self.leaky_relu(f_src.unsqueeze(-1) + f_dst.unsqueeze(-2))           # (B,H,N,N)

        adj_exp = adj.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        scores = scores.masked_fill(adj_exp == 0, -1e9)
        if return_scores:
            return scores.detach().cpu()

        attn = F.softmax(scores / self.temperature, dim=-1)
        out = self.dropout(torch.matmul(attn, Wh))
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        if return_attention:
            return out, attn.detach().cpu()
        return out
