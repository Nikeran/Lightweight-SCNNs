import torch
import torch.nn as nn


from Baselines.AdaBatchNorm       import AdaBatchNorm2d
from Baselines.CBAM               import CBAM
from Baselines.PrototypeAlignment import PrototypeAlignment
from Baselines.RBN                import RBN


class ResBlock18(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ModifiedResNet18Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_adabn=False, use_cbam=False, use_proto=False, use_rbn=False, use_TC=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = (AdaBatchNorm2d(out_channels) if use_adabn else nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        
        if use_rbn:
            self.bn2 = RBN(out_channels)
        else:
            self.bn2 = (AdaBatchNorm2d(out_channels) if use_adabn else nn.BatchNorm2d(out_channels))
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.cbam = CBAM(out_channels) if use_cbam else None
        self.proto = PrototypeAlignment(out_channels) if use_proto else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.cbam is not None:
            out = self.cbam(out)
        if self.proto is not None:
            out = self.proto(out)

        return out

    
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, use_adabn=False, use_cbam=False, use_proto=False, use_rbn=False):
        super(ResNet18, self).__init__()
        self.use_adabn = use_adabn
        self.use_cbam = use_cbam
        self.use_proto = use_proto
        self.use_rbn = use_rbn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ModifiedResNet18Block(in_channels, out_channels, stride, self.use_adabn, self.use_cbam, self.use_proto, self.use_rbn))
        for _ in range(1, blocks):
            layers.append(ModifiedResNet18Block(out_channels, out_channels, 1, self.use_adabn, self.use_cbam, self.use_proto, self.use_rbn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #store feat embedding head
        feat = x
        x = self.fc(x)

        return feat, x
        

class ResBlock50(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock50, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ModifiedResNet50Block(nn.Module):
    def __init__(self,
                    in_channels,
                    out_channels,
                    stride=1,
                    use_adabn=False,
                    use_cbam=False,
                    use_proto=False,
                    use_rbn=False,
                    use_TC=False):
        super().__init__()
        out_channels = out_channels * 4

        # 1×1 reduction
        self.conv1 = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=stride,
                                bias=False)
        self.bn1 = (AdaBatchNorm2d(out_channels)
                    if use_adabn else nn.BatchNorm2d(out_channels))
        # 3×3
        self.conv2 = nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        if use_rbn:
            self.bn2 = RBN(out_channels)
        else:
            self.bn2 = (AdaBatchNorm2d(out_channels)
                        if use_adabn else nn.BatchNorm2d(out_channels))
        # 1×1 expansion
        self.conv3 = nn.Conv2d(out_channels,
                                out_channels,
                                kernel_size=1,
                                bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # downsample path if needed
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=1,
                            stride=stride,
                            bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # optional modules
        self.cbam = CBAM(out_channels) if use_cbam else None
        self.proto = PrototypeAlignment(out_channels) if use_proto else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.cbam is not None:
            out = self.cbam(out)
        if self.proto is not None:
            out = self.proto(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, use_adabn=False, use_cbam=False, use_proto=False, use_rbn=False):
        super(ResNet50, self).__init__()
        self.use_adabn = use_adabn
        self.use_cbam = use_cbam
        self.use_proto = use_proto
        self.use_rbn = use_rbn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ModifiedResNet50Block(in_channels, out_channels // 4, stride, self.use_adabn, self.use_cbam, self.use_proto, self.use_rbn))
        for _ in range(1, blocks):
            layers.append(ModifiedResNet50Block(out_channels // 4 * 4, out_channels // 4, 1, self.use_adabn, self.use_cbam, self.use_proto, self.use_rbn))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = x
        x = self.fc(x)

        return feat, x
        
