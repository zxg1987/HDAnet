import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ADCA(nn.Module):

    def __init__(self, embed_dim, num_heads, n_nodes, pool_size, init_beta, use_qk_norm):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_nodes = n_nodes
        self.use_qk_norm = use_qk_norm
        self.min_beta = 1e-4
        self.max_beta = 10.0
        ratio = (init_beta - self.min_beta) / (self.max_beta - self.min_beta)
        self.beta_raw = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio))))
        self.q_projs = nn.ModuleList(
            [nn.Conv2d(embed_dim, embed_dim, 1, bias=False) for _ in range(n_nodes)]
        )
        self.k_projs = nn.ModuleList(
            [nn.Conv2d(embed_dim, embed_dim, 1, bias=False) for _ in range(n_nodes)]
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.scale = self.head_dim ** -0.5

    @property
    def beta(self):
        ratio = torch.sigmoid(self.beta_raw)
        return ratio * (self.max_beta - self.min_beta) + self.min_beta

    def _to_heads(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, self.num_heads, self.head_dim, H * W)
        return x.transpose(2, 3).contiguous()

    def _edge(self, q_i, k_j):
        scores = torch.matmul(q_i, k_j.transpose(-2, -1))
        if not self.use_qk_norm:
            scores = scores * self.scale
        attn = F.softmax(scores, dim=-1)
        S_k = attn.shape[-1]
        conc = (attn ** 2).sum(dim=-1)
        edge = (conc - 1.0 / S_k) / (1.0 - 1.0 / S_k)
        return edge.clamp(0.0, 1.0).mean(dim=(1, 2))

    def _symmetrize(self, e_ij, e_ji):
        beta = self.beta.to(e_ij.device, e_ij.dtype)
        pair = torch.stack([beta * e_ij, beta * e_ji], dim=0)
        return (torch.logsumexp(pair, dim=0) - math.log(2.0)) / beta

    def forward(self, feat_maps):
        if torch.is_tensor(feat_maps):
            feat_maps = [feat_maps[:, i] for i in range(self.n_nodes)]

        B = feat_maps[0].shape[0]
        device, dtype = feat_maps[0].device, feat_maps[0].dtype
        N = self.n_nodes

        pooled = [self.spatial_pool(f) for f in feat_maps]
        q_tokens, k_tokens = [], []
        for i in range(N):
            q, k = self.q_projs[i](pooled[i]), self.k_projs[i](pooled[i])
            q, k = self._to_heads(q), self._to_heads(k)
            if self.use_qk_norm:
                q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            q_tokens.append(q)
            k_tokens.append(k)

        zero = torch.zeros(B, device=device, dtype=dtype)
        sym = [[zero if i == j else None for j in range(N)] for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                e = self._symmetrize(
                    self._edge(q_tokens[i], k_tokens[j]),
                    self._edge(q_tokens[j], k_tokens[i]),
                )
                sym[i][j] = sym[j][i] = e

        return torch.stack([torch.stack(row, dim=1) for row in sym], dim=1)
