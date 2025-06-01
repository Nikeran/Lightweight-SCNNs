import torch
import torch.nn as nn


class AdaBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, U=1.0):
        super().__init__()
         # input/output: (B, num_features, H, W)
        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum,
                                 affine=affine, track_running_stats=track_running_stats) # (B, num_features, H, W)
        # U: interpolation factor
        self.U = U

    def forward(self, x):
        # Current batch stats
        batch_mean = x.mean([0, 2, 3]) # (C,)
        batch_var = x.var([0, 2, 3], unbiased=False) # (C,)
        # Running(train) stats
        running_mean = self.bn.running_mean # (C,)
        running_var = self.bn.running_var # (C,)
        # Interpolate
        mean = (1 - self.U) * running_mean + self.U * batch_mean # (C,)
        var = (1 - self.U) * running_var + self.U * batch_var # (C,)
        # Apply norm
        # (B, C, H, W)
        x_norm = (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + self.bn.eps)
        if self.bn.affine:
            x_norm = x_norm * self.bn.weight[None, :, None, None] + self.bn.bias[None, :, None, None]
        return x_norm
    


        

        
