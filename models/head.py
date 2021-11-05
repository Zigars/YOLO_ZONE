'''
@Description:head: Decoupling head for YOLOX
@Author:Zigar
@Date:2021/11/02 13:23:54
'''

import torch
import torch.nn as nn

from .common import BaseConv, DWConv


#---------------------------------------------------------------#
# Yolox's Decoupling head
#---------------------------------------------------------------#
class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()

        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems     = nn.ModuleList()

        # P3, P4, P5
        # 256/8, 512/16, 1024/32
        for i in range(len(in_channels)):
            # 1x1 Conv for Channel integration
            # in_channels -> 256
            self.stems.append(
                BaseConv(
                    in_channels  = int(in_channels[i] * width),
                    out_channels = int(256 * width),
                    ksize        = 1,
                    stride       = 1,
                    act          = act
                )
            )

            # Class branch
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels  = int(256 * width),
                            out_channels = int(256 * width),
                            ksize        = 3,
                            stride       = 1,
                            act          = act
                        ),
                        Conv(
                            in_channels  = int(256 * width),
                            out_channels = int(256 * width),
                            ksize        = 3,
                            stride       = 1,
                            act          = act
                        ),
                    ]
                )
            )

            # class pred branch
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

            # # Regression branch
            # Include region pred branch(4) and obj(IOU) pred branch(1)
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels  = int(256 * width),
                            out_channels = int(256 * width),
                            ksize        = 3,
                            stride       = 1,
                            act          = act
                        ),
                        Conv(
                            in_channels  = int(256 * width),
                            out_channels = int(256 * width),
                            ksize        = 3,
                            stride       = 1,
                            act          = act
                        ),
                    ]
                )
            )


            # region pred branch(4)
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

            # obj(IOU) pred branch(1)
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

    def forward(self, inputs):
        # inputs: P3, P4, P5
        # 256/8, 512/16, 1024/32
        outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)

            # class branch
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)

            # regression branch
            reg_feat = self.reg_convs[k](x)

            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        return outputs
