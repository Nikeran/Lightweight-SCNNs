import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class TopoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # angles for each filter on a circular manifold
        self.theta = nn.Parameter(torch.rand(out_channels, in_channels, 1, 1) * 2 * math.pi) 

        # radius per circle
        self.radius = nn.Parameter(torch.ones(out_channels, in_channels, 1, 1))

        # base learnable kernel
        self.base_filter = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):

        # x: (B, in_channels, H, W)
        filters = self.radius * torch.cos(self.theta) * self.base_filter + \
                    self.radius * torch.sin(self.theta) * torch.flip(self.base_filter, dims=[-1]) 
        
        # filters: (out_channels, in_channels, kernel_size, kernel_size)
        x = F.conv2d(x, filters, stride=self.stride, padding=self.padding, groups=1)#self.in_channel)

        return x
    

class TopoResNet18Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.topo_conv = TopoConv(out_channels, out_channels,
                                  kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        # now conv1 uses the same `stride`
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.topo_conv(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity  # shapes now match
        return self.relu(x)

class TopoResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()#nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)   

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(TopoResNet18Block(in_channels, out_channels, stride))
        for _ in range(1, n_blocks):
            layers.append(TopoResNet18Block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.conv1(x)  # [B, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [B, 64, H/4, W/4]

        x = self.layer1(x)  # [B, 64, H/4, W/4]
        x = self.layer2(x)  # [B, 128, H/8, W/8]
        x = self.layer3(x)  # [B, 256, H/16, W/16]
        x = self.layer4(x)  # [B, 512, H/32, W/32]

        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        feats = x
        x = self.fc(x)  # [B, num_classes]

        return feats, x



class TopoResNet50Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        width = out_channels
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.topo_conv = TopoConv(width, width, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.topo_conv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class TopoResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(64 * TopoResNet50Block.expansion, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(128 * TopoResNet50Block.expansion, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(256 * TopoResNet50Block.expansion, 512, blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * TopoResNet50Block.expansion, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [TopoResNet50Block(in_channels, out_channels, stride)]
        in_channels = out_channels * TopoResNet50Block.expansion
        for _ in range(1, blocks):
            layers.append(TopoResNet50Block(in_channels, out_channels, stride=1))
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
        feats = x
        x = self.fc(x)
        return feats, x