import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfCorrectingBlock(nn.Module):
    def __init__(self, num_channels, num_prototypes=16, hidden_dim=64):
        """
        Args:
          num_channels: number of channels in the feature map you correct
          num_prototypes: size of the codebook (K)
          hidden_dim: hidden size for the gating MLP
        """
        super().__init__()
        # Prototype codebook: K x C
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, num_channels))
        # A small gating MLP: C <- hidden_dim <- C
        self.gate = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Global‐average‐pool each channel: [B, C]
        summary = x.mean(dim=[2,3])  

        # Compute distances to each prototype (broadcasted): [B, K]
        #    here we use squared Euclidean distance
        #    dist[b, k] = ||summary[b] - prototype[k]||^2
        #    you can speed this up via torch.cdist if you have PyTorch >=1.8
        #    but here’s a simple manual version:
        #    summary_sq: [B,1], proto_sq: [1,K], cross: [B,K]
        summary_sq = (summary**2).sum(dim=1, keepdim=True)               # [B, 1]
        proto_sq   = (self.prototypes**2).sum(dim=1, keepdim=True).T     # [1, K]
        cross      = summary @ self.prototypes.T                         # [B, K]
        d2         = summary_sq + proto_sq - 2*cross                     # [B, K]

        # Find the nearest prototype idx for each example
        nearest_idx = d2.argmin(dim=1)                                   # [B]

        # Gather the matched prototype vectors: [B, C]
        matched_proto = self.prototypes[nearest_idx]                     # [B, C]

        # Run the gating MLP on that prototype to get per‐channel scales
        # gating: [B, C] with values in (0,1)
        scales = self.gate(matched_proto)                                # [B, C]

        # Reshape and apply to x
        scales = scales.view(B, C, 1, 1)
        return x * scales                                                 # [B, C, H, W]

