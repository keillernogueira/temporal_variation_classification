import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from networks.utils import initialize_weights


class _ResNet18(nn.Module):
    def __init__(self, input_channels, pretrained):
        super(_ResNet18, self).__init__()

        # loading backbone
        backbone = models.resnet18(pretrained=pretrained, progress=False)

        if input_channels == 3:
            self.init = nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool
            )
        else:
            self.init = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                backbone.bn1,
                backbone.relu,
                backbone.maxpool
            )

        self.layer1 = backbone.layer1  # output feat = 64
        self.layer2 = backbone.layer2  # output feat = 128
        self.layer3 = backbone.layer3  # output feat = 256
        self.layer4 = backbone.layer4  # output feat = 512

    def forward(self, x):
        fv_init = self.init(x)
        fv1 = self.layer1(fv_init)
        fv2 = self.layer2(fv1)
        fv3 = self.layer3(fv2)
        fv4 = self.layer4(fv3)

        return fv2, fv4


class _ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(_ASPPBlock, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # atrous convs
        self.atrous_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=atrous_rates[0], dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=atrous_rates[1], dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=atrous_rates[2], dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.pool_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        fv_init = self.init_conv(x)
        fv_c1 = self.atrous_conv1(x)
        fv_c2 = self.atrous_conv2(x)
        fv_c3 = self.atrous_conv3(x)

        fv_pc = self.pool_conv(x)
        fv_pc_r = F.interpolate(fv_pc, size=(h, w), mode="bilinear", align_corners=False)

        cat = torch.cat([fv_init, fv_c1, fv_c2, fv_c3, fv_pc_r], dim=1)
        return self.final_conv(cat)


class DeepLabV3Plus(nn.Module):
    def __init__(self, input_channels, num_classes, pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = _ResNet18(input_channels, pretrained)

        self.low_level_feat_proj = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.aspp = _ASPPBlock(512, 256, atrous_rates=[2, 4, 6])

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

        if not pretrained:
            initialize_weights(self)

    def forward(self, x):
        low_level, out = self.backbone(x)

        low_level_fv = self.low_level_feat_proj(low_level)

        aspp_feat = self.aspp(out)
        aspp_feat = F.interpolate(aspp_feat, size=low_level_fv.shape[2:], mode="bilinear", align_corners=False)

        final = self.classifier(torch.cat([low_level_fv, aspp_feat], dim=1))

        return F.interpolate(final, size=x.shape[2:], mode="bilinear", align_corners=False)
