import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2 as cv
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from enum import Enum

# Double Convolution of UNet block
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # handle odd input sizes
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ResUNetModes(Enum):
    both = 0,
    onlyClassification = 1,
    onlySegmentation = 2
    
class ResUNet(nn.Module):
    def __init__(self, pretrained_encoder : models.ResNet, num_classes=1):
        super().__init__()
        
        # ----------- Modes -----------
        self.mode : ResUNetModes = ResUNetModes.both

        # ---------- Encoder ----------
        resnet = pretrained_encoder

        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )                       # 64, H/2
        self.encoder1 = resnet.layer1  # 64, H/4
        self.encoder2 = resnet.layer2  # 128, H/8
        self.encoder3 = resnet.layer3  # 256, H/16
        self.encoder4 = resnet.layer4  # 512, H/32

        self.pool = resnet.maxpool

        # ---------- Decoder ----------
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)

        # ---------- Segmentation Head ----------
        self.segHead = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # ---------- Classification Head ----------
        self.fc = resnet.fc
        
        self.classHead = nn.Sequential(
            resnet.avgpool,
            nn.Flatten(),
            self.fc  # pretrained fc layer
        )

        
        self.decoder_params =list(self.up4.parameters()) + \
                            list(self.up3.parameters()) + \
                            list(self.up2.parameters()) + \
                            list(self.up1.parameters()) + \
                            list(self.segHead.parameters())
        
        self.encoder_params =    list(self.encoder0.parameters()) + \
                            list(self.pool.parameters()) + \
                            list(self.encoder1.parameters()) + \
                            list(self.encoder2.parameters()) + \
                            list(self.encoder3.parameters()) + \
                            list(self.encoder4.parameters()) + \
                            list(self.classHead.parameters())

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        classification = None
        
        if self.mode is not ResUNetModes.onlySegmentation:
            classification = self.classHead(e4)

        if self.mode is ResUNetModes.onlyClassification:
            return classification
        
        # Decoder
        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        
        segmentation = self.segHead(d1)
        # ---------- Interpolate segmentation to match input size ----------
        segmentation = F.interpolate(segmentation, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        if self.mode is ResUNetModes.onlySegmentation:
            return segmentation

        return segmentation, classification
    
    def setModeOnlyClassification(self):
        self.mode = ResUNetModes.onlyClassification
    
    def setModeOnlySegmentation(self):
        self.mode = ResUNetModes.onlySegmentation
    
    def setModeBoth(self):
        self.mode = ResUNetModes.both