import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel-wise attention as per CBAM, using a regular MLP on pooled features.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        # MLP: two Linear layers with ReLU
        self.flatten = nn.Flatten()  # flattens (B, C, 1, 1) -> (B, C)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // reduction, bias=False),  # (B, C) -> (B, C/reduction)
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_channels // reduction, out_features=in_channels, bias=False)  # (B, C/reduction) -> (B, C)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        # squeeze spatial dimensions for avg and max pooling
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)  # (B, C, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, output_size=1)  # (B, C, 1, 1)
        # flatten to (B, C)
        avg_flat = self.flatten(avg_pool)  # (B, C)
        max_flat = self.flatten(max_pool)  # (B, C)
        # MLP on each
        avg_out = self.fc(avg_flat)  # (B, C)
        max_out = self.fc(max_flat)  # (B, C)
        # Sum and activate
        att = avg_out + max_out  # (B, C)
        #broadcast over space
        att = self.sigmoid(att).view(B, C, 1, 1)  # (B, C, 1, 1)
        # scale input
        return x * att  # (B, C, H, W)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2

        #Spatial mask over the mean and avg pool
        self.conv = nn.Conv2d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=False)  # (B, 2, H, W) -> (B, 1, H, W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        out = self.conv(x_cat) # (B, 1, H, W)
        return x * self.sigmoid(out)



class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.c_at = ChannelAttention(in_channels=in_channels, reduction=reduction)
        self.s_at = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x): # x: (B, C, H, W)
        #Apply Ch att
        x = self.c_at(x) # x: (B, C, H, W)
        #Apply Sp att
        x = self.s_at(x) # x: (B, C, H, W)

        return x