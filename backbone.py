import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

class ImageBackbone(nn.Module):
    """
    ACT synthesizes images from myltuple viewpoints, joint positions, and a latent style 
    variable z with a transformer encoder, and predicts a sequence of actions with a 
    transformer decoder
    
    Backbone is step 1. It converts raw camera images into the "images from multiple viewpoints".

    Everything downstream depends on quality of visual features.

    The vision backbone uses ResNet-18 to process images from multiple camera viewpoints.
    """
    def __init__(self, d_model=256, pretrained=True):
        super().__init__()
        # resnet18 is the full pretrained model including classifier head
        # will remove parts we don't want
        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        # original resnet ends with avgpool and fc layers (collapses 7x7 -> 1x1 and maps to 1000 ImageNet classes respectively)
        # stop before both because we want the spatial 7x7 grid of features and not a single class prediction
        # transformer downstream needs to know where things are in the image rather than what's in it
        # layer 1 through 4 is each a group of residual blocks

        # a regular block looks like conv2d (stride 2, halves spatial dimensions) -> batchnorm -> relu -> 
        # conv2d -> batchnorm -> shortcut add (conv1x1, only needed when channel count changes 64 -> 128, 
        # otherwise add input directly with identity) -> relu
        self.body = nn.Sequential(
            resnet.conv1,   # (3, 224, 224) → (64, 112, 112)
            resnet.bn1,     # normalize 64 channels
            resnet.relu,    # zero out negatives
            resnet.maxpool, # (64, 112, 112) → (64, 56, 56)
            resnet.layer1,  # (64, 56, 56)  → (64, 56, 56)   [2 residual blocks]
            resnet.layer2,  # (64, 56, 56)  → (128, 28, 28)  [2 residual blocks]
            resnet.layer3,  # (128, 28, 28) → (256, 14, 14)  [2 residual blocks]
            resnet.layer4   # (256, 14, 14) → (512, 7, 7)    [2 residual blocks]
        )

        # 512 channels, transformer expects everything to be d_model=256 dimensional
        # 1x1 conv is pointwise projection - at every spatial position, mixes channels without mixing spatial positions
        # just a learned linear map 512 -> 256 applied at the 49 grid positions independently to change the channel count
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        # learnable 2D positional embedding which adds a learned spatial address to each grid cell before flattening
        # gets updated by gradient descent alongside other weights to find what spatial encoding is most useful for downstream transformer
        # 0.02 initialization ensures positional embeddings start small so model can learn to use them without overwhelming actual image features early in training
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, 7, 7) * 0.02)

        def forward(self, x):
            B, num_cams, C, H, W = x.shape # x : (B, num_cams, C, H, W)

            # merge batch + camera dims for efficient parallel processing
            x = rearrange(x, 'b n c h w -> (b n) c h w')

            feat = self.body(x) # (B * num_cams, 512, 7, 7), now have 7x7 grid of 512-dim feature vectors per image
            feat = self.proj(feat) # (B * num_cams, d_model, 7, 7), projected from 512 -> d_model = 256 channels
            feat = feat + self.pos_embed # add spatial position signal

            # flatten spatial grid into sequence of tokens and seperate the batch and camera dims back out
            feat = rearrange(feat, '(b n) d h w -> b (n h w) d', b=B, n=num_cams)
            # each toen is a 256-dim vector describing one spatial location in one camera view, and the sequence is the concatenation of all 7x7 grid locations across all camera views
            # output: (B, num_cams * 7 * 7, d_model)
            return feat
