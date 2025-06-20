import torch
import torch.nn as nn
import torchvision.models as models



from Baselines.AdaBatchNorm       import AdaBatchNorm2d
from Baselines.CBAM               import CBAM
from Baselines.PrototypeAlignment import PrototypeAlignment
from Baselines.RBN                import RBN

class ModifiedResNetBlock(nn.Module):
    """
    If use_rbn=True, only the second BN in each block becomes RBN;
    initial and first BNs stay as original (or AdaBN if requested).
    """
    def __init__(self, original_block, use_adabn=False, use_cbam=False, use_proto=False, use_rbn=False):
        super().__init__()
        self.conv1 = original_block.conv1
        # always keep initial bn1 (or AdaBN)
        self.bn1 = (AdaBatchNorm2d(original_block.bn1.num_features) if use_adabn
                    else original_block.bn1)
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        # only replace bn2 when use_rbn
        if use_rbn:
            self.bn2 = RBN(original_block.bn2.num_features)
        else:
            self.bn2 = (AdaBatchNorm2d(original_block.bn2.num_features) if use_adabn
                        else original_block.bn2)
        self.downsample = original_block.downsample
        self.cbam = CBAM(original_block.bn2.num_features) if use_cbam else None
        self.proto = PrototypeAlignment(original_block.bn2.num_features) if use_proto else None
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        if self.cbam is not None:
            out = self.cbam(out)
        if self.proto is not None:
            out = self.proto(out)
        return out


def build_model(num_classes=10, use_adabn=False, use_cbam=False, use_proto=False, use_rbn=False):
    model = __import__('torchvision.models').models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    # always keep initial BN intact (or AdaBN), RBN only in blocks
    model.bn1 = model.bn1
    model.maxpool = nn.Identity()
    for name, module in model.named_children():
        if name.startswith('layer'):
            blocks = []
            for blk in module:
                blocks.append(ModifiedResNetBlock(
                    blk,
                    use_adabn=use_adabn,
                    use_cbam=use_cbam,
                    use_proto=use_proto,
                    use_rbn=use_rbn
                ))
            setattr(model, name, nn.Sequential(*blocks))
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model



