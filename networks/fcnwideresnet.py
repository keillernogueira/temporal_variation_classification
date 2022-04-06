import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

from networks.utils import initialize_weights


class FCNWideResNet50(nn.Module):
    def __init__(self, input_channels, num_classes, pretrained=True, skip=True):
        super(FCNWideResNet50, self).__init__()
        self.skip = skip

        # load pre-trained model
        wideresnet = models.wide_resnet50_2(pretrained=pretrained, progress=False)

        if input_channels == 3:
            self.init = nn.Sequential(
                wideresnet.conv1,
                wideresnet.bn1,
                wideresnet.relu,
                wideresnet.maxpool
            )
        else:
            self.init = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
                wideresnet.bn1,
                wideresnet.relu,
                wideresnet.maxpool
            )
        self.layer1 = wideresnet.layer1  # output feat = 256
        self.layer2 = wideresnet.layer2  # output feat = 512
        self.layer3 = wideresnet.layer3  # output feat = 1024
        self.layer4 = wideresnet.layer4  # output feat = 2048

        if self.skip:
            self.classifier1 = nn.Sequential(
                nn.Conv2d(2048 + 512, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout2d(0.5),
            )
        else:
            self.classifier1 = nn.Sequential(
                nn.Conv2d(2048, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout2d(0.5),
            )
        self.final = nn.Conv2d(128, num_classes, kernel_size=3, padding=1)

        if not pretrained:
            initialize_weights(self)
        else:
            initialize_weights(self.classifier1)
            initialize_weights(self.final)

    def forward(self, x):
        fv_init = self.init(x)
        fv1 = self.layer1(fv_init)
        fv2 = self.layer2(fv1)
        fv3 = self.layer3(fv2)
        fv4 = self.layer4(fv3)

        if self.skip:
            # Forward on FCN with Skip Connections.
            fv_final = torch.cat([F.interpolate(fv2, x.size()[2:], mode='bilinear', align_corners=False),
                                  F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False)], 1)
        else:
            # Forward on FCN without Skip Connections.
            fv_final = F.interpolate(fv4, x.size()[2:], mode='bilinear', align_corners=False)

        classif1 = self.classifier1(fv_final)
        output = self.final(classif1)

        return output
