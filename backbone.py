import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

class ImageBackbone(nn.Module):
    def __init__(self, d_model=256, pretrained=True):
        super().__init__()
        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        self.body = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        self.pos_embed = nn.Parameter(torch.randn(1, d_model, 7, 7) * 0.02)
