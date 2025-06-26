import torch
import torch.nn as nn
import torch.nn.functional as F

class KoLeoLoss(nn.Module):
    """KoLeo differential‐entropy regularizer.
    Given a batch of feature‐vectors (B, D), encourages them to spread uniformly."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, D), assumed l2‐normalized
        dists = torch.cdist(x1=feats, x2=feats) + self.eps        # (B, B)
        nn_dist = dists.min(dim=1).values                  # (B,) nearest‐neighbor
        return -nn_dist.log().mean()                       # –E[log d_i]

class SinkhornCentering(nn.Module):
    """Run Sinkhorn–Knopp on a batch of logits to produce
    a ‘doubly‐stochastic’ target distribution."""
    def __init__(self, iters: int = 3, eps: float = 1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
          logits: (B, K) un‐normalized scores
        Returns:
          Q: (B, K) “centered” probabilities (rows&cols≈uniform)
        """
        B, K = logits.shape
        Q = torch.exp(logits / self.eps).t()    # (K, B)
        Q /= Q.sum()                           # make sum=1
        r = torch.ones(K, device=Q.device) / K
        c = torch.ones(B, device=Q.device) / B

        for _ in range(self.iters):
            u = Q.sum(dim=1)
            Q *= (r / u).unsqueeze(1)
            v = Q.sum(dim=0)
            Q *= (c / v).unsqueeze(0)

        return (Q / Q.sum(dim=0, keepdim=True)).t()  # back to (B, K)
