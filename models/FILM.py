import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, c_dim, feat_dim):
        super().__init__()
        # Transformation to create the aphta and beta parametersfor the shift
        self.proj = nn.Linear(c_dim, feat_dim * 2)

    def forward(self, x, c):
        B, C, H, W = x.shape

        shift_params = self.proj(c)
        alpha, beta = shift_params.chunk(2, dim=1)

        alpha = alpha.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        x = alpha * x + beta
        return x
    

class ResBlockFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.film1 = FiLM(cond_dim, out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.film2 = FiLM(cond_dim, out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x, c):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out, c)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, c)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)
    
# a FiLM‐ResNet18 model with a Self Generating Context and a double pass
class FiLMNet18_SGC(nn.Module):
    def __init__(self, c_dim, num_classes=10):
        super().__init__()
        self.c_dim = c_dim
        self.num_classes = num_classes
        # --- stem ---
        self.conv1   = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # --- FiLM‐ResNet18 blocks ---
        self.layer1 = self._make_layer(64,  64,  c_dim, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128,  c_dim, blocks=2, stride=2)
        self.layer3 = self._make_layer(128,256,  c_dim, blocks=2, stride=2)
        self.layer4 = self._make_layer(256,512,  c_dim, blocks=2, stride=2)

        # --- context generator: from final conv features to c_dim ---
        # we assume x after layer4 is [B,512,H',W']
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),   # [B,256,1,1]
            nn.Flatten(),                  # [B,256]
            nn.Linear(256, c_dim),         # [B,c_dim]
            nn.ReLU(inplace=True),
        )

        # --- head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, c_dim, blocks, stride):
        layers = []
        layers.append(ResBlockFiLM(in_ch, out_ch, c_dim, stride=stride))
        for _ in range(1, blocks):
            layers.append(ResBlockFiLM(out_ch, out_ch, c_dim, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        device = x.device

        # --- first pass with zero context ---
        c0 = torch.zeros(B, self.c_dim, device=device)  # [B,c_dim]
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0 = self.maxpool(x0)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            # each layer is nn.Sequential of ResBlockFiLM
            for block in layer:
                x0 = block(x0, c0)

        # --- build real context from these features ---
        c = self.ctx_conv(x0)  # [B,c_dim]

        # --- second pass with learned context ---
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x1)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x1 = block(x1, c)

        feat   = torch.flatten(self.avgpool(x1), 1)  # [B,512]
        logits = self.fc(feat)                      # [B,num_classes]
        return feat, logits
    

class BottleneckFiLM(nn.Module):
    expansion = 4
    def __init__(self, in_ch, mid_ch, cond_dim, stride=1):
        super().__init__()
        out_ch = mid_ch * self.expansion
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)
        self.film1 = FiLM(cond_dim, mid_ch)
        # 3x3 conv
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.film2 = FiLM(cond_dim, mid_ch)
        # 1x1 conv
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        # downsample for skip
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, c):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out, c)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, c)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class FiLMNet50_SGC(nn.Module):
    def __init__(self, c_dim, num_classes=10):
        super().__init__()
        self.c_dim = c_dim
        # --- stem ---
        self.conv1   = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # --- FiLM-ResNet50 blocks ---
        self.layer1 = self._make_layer(64,  64,  c_dim, blocks=3, stride=1)
        self.layer2 = self._make_layer(256, 128, c_dim, blocks=4, stride=2)
        self.layer3 = self._make_layer(512, 256, c_dim, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024,512, c_dim, blocks=3, stride=2)
        # --- context generator: from final conv features to c_dim ---
        self.ctx_conv = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),  # [B,512,1,1]
            nn.Flatten(),                 # [B,512]
            nn.Linear(512, c_dim),        # [B,c_dim]
            nn.ReLU(inplace=True),
        )
        # --- head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, num_classes)

    def _make_layer(self, in_ch, mid_ch, c_dim, blocks, stride):
        layers = []
        layers.append(BottleneckFiLM(in_ch, mid_ch, c_dim, stride=stride))
        for _ in range(1, blocks):
            layers.append(BottleneckFiLM(mid_ch*BottleneckFiLM.expansion, mid_ch, c_dim, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        device = x.device
        # first pass with zero context
        c0 = torch.zeros(B, self.c_dim, device=device)
        # stem
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0 = self.maxpool(x0)
        # layers
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x0 = block(x0, c0)
        # build real context
        c = self.ctx_conv(x0)
        # second pass with learned context
        x1 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x1)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                x1 = block(x1, c)
        feat   = torch.flatten(self.avgpool(x1), 1)  # [B,2048]
        logits = self.fc(feat)                        # [B,num_classes]
        return feat, logits
