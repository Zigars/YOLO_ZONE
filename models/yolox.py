''' 
@Description: YOLOX Model Network
@Author:Zigar
@Date:2021/10/31 19:58:12
'''
import torch.nn as nn
from models.neck import YOLOPAFPN
from models.head import YOLOXHead


#---------------------------------------------------------------#
# YOLOX Models
#---------------------------------------------------------------#
class YOLOX(nn.Module):
    def __init__(
        self, 
        num_classes, 
        version="s",
    ):
        super().__init__()
        num_classes = num_classes
        depth_width = {
            "nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.50],
            "m": [0.67, 0.75],
            "l": [1.00, 1.00],
            "x": [1.33, 1.25],
        }
        depth, width = depth_width[version]
        depthwise = True if version == "nano" else False
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs