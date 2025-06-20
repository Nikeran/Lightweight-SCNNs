import torch

def koleo_loss(outputs):
    dists = torch.cdist(outputs, outputs) + 1e-6 # Compute pairwise distances
    nn_dist = dists.min(dim=1).values  # Nearest neighbor distances
    return nn_dist.mean()  # Return mean distance as loss