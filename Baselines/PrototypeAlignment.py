import torch
import torch.nn as nn


class PrototypeAlignment(nn.Module):
	"""
	Prototype-based feature alignment. After global pooling, match to nearest prototype,
	then optionally snap features toward prototype by factor alpha.
	"""
	def __init__(self, num_features, num_prototypes=10, alpha=0.5):
		super().__init__()
		self.prototypes = nn.Parameter(torch.randn(num_prototypes, num_features))  # (P, C)
		self.alpha = alpha
	
	def forward(self, x): # (B, C, H, W)
		B, C, H, W = x.size()
		feat = x.view(B, C, -1).mean(dim=2) # H and W wise mean, (B, C)
		dists = torch.cdist(x1=feat, x2=self.prototypes) # (B, P)
		idx = torch.argmin(dists, dim=1) #  closest prototype (B) 
		nearest_p = self.prototypes[idx] # (B, C)
		# Compute channel-wise offset and apply in one step
		delta = self.alpha * (nearest_p - feat)          # (B, C)
		delta = delta.view(B, C, 1, 1).expand_as(x)    # (B, C, H, W)
		return x + delta                               # (B, C, H, W)

