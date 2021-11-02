''' 
@Description:YOLOX Model Network
@Author:Zigar
@Date:2021/10/31 19:58:12
'''
import torch
import torch.nn as nn
from head import YOLOXHead
from neck import YOLOPAFPN


class YOLOX(nn.Module):
    def __init__(self, num_classes, version):
        super().__init__()
        # yolox的深度和宽度
        self.num_classes = num_classes
        self.depth_width = {
            "nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.50],
            "m": [0.67, 0.75],
            "l": [1.00, 1.00],
            "x": [1.33, 1.25],
        }
        self.depth, self.width = self.depth_width[version]
        self.in_channels = [256, 512, 1024]
        self.depthwise = True if version == "nano" else False
        self.backbone = YOLOPAFPN(self.depth, self.width, in_channels=self.in_channels, depthwise=self.depthwise)
        self.head = YOLOXHead(self.num_classes, self.width, in_channels=self.in_channels, depthwise=self.depthwise)


    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs
