import torch
import torch.nn as nn
import torchvision.models as models



from Baselines.AdaBatchNorm       import AdaBatchNorm2d
from Baselines.CBAM               import CBAM
from Baselines.PrototypeAlignment import PrototypeAlignment

class ModifiedResNetBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        use_adabn: bool = False,
        use_cbam: bool = False,
        use_proto: bool = False
    ):
        super().__init__()
        # First conv + norm + activation
        self.conv1 = original_block.conv1  # (B, C_in, H, W) -> (B, C_mid, H, W)
        self.bn1 = (
            AdaBatchNorm2d(num_features=original_block.bn1.num_features)
            if use_adabn
            else original_block.bn1
        )  # (B, C_mid, H, W)
        self.relu = original_block.relu

        # Second conv + norm
        self.conv2 = original_block.conv2  # (B, C_mid, H, W) -> (B, C_out, H, W)
        self.bn2 = (
            AdaBatchNorm2d(num_features=original_block.bn2.num_features)
            if use_adabn
            else original_block.bn2
        )  # (B, C_out, H, W)

        # Optional downsample for matching dimensions
        self.downsample = original_block.downsample  # or None

        # Optional attention / prototype alignment
        self.cbam = (
            CBAM(in_channels=original_block.bn2.num_features)
            if use_cbam
            else None
        )  # (B, C_out, H, W)
        self.proto = (
            PrototypeAlignment(num_features=original_block.bn2.num_features)
            if use_proto
            else None
        )  # (B, C_out, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # (B, C_in, H, W)

        out = self.conv1(x)      # -> (B, C_mid, H, W)
        out = self.bn1(out)      # -> (B, C_mid, H, W)
        out = self.relu(out)     # -> (B, C_mid, H, W)

        out = self.conv2(out)    # -> (B, C_out, H, W)
        out = self.bn2(out)      # -> (B, C_out, H, W)

        if self.downsample is not None:
            identity = self.downsample(x)  # -> (B, C_out, H, W)

        out = out + identity     # residual add
        out = self.relu(out)     # -> (B, C_out, H, W)

        if self.cbam is not None:
            out = self.cbam(out)    # apply CBAM
        if self.proto is not None:
            out = self.proto(out)   # apply prototype alignment

        return out  # (B, C_out, H, W)


# For now only supporting the baseline methods
def build_model(num_classes=10, use_adabn=False, use_cbam=False, use_proto=False):
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    for name, module in model.named_children():
        if name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layers = []
            for block in module:
                layers.append(ModifiedResNetBlock(block,
                                                 use_adabn=use_adabn,
                                                 use_cbam=use_cbam,
                                                 use_proto=use_proto))
            setattr(model, name, nn.Sequential(*layers))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model



