'''
@Description:neck: PAFPN/FPN
@Author:Zigar
@Date:2021/11/02 11:34:20
'''
import torch
import torch.nn as nn
from backbone import Darknet, CSPDarknet
from common import BaseConv, CSPLayer, DWConv



#---------------------------------------------------------------#
# YOLOPAFPN
# CSPDarknet is the default backbone of this model
#---------------------------------------------------------------#
class YOLOPAFPN(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu"
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise,act=act)
        self.in_features= ("dark3","dark4","dark5")
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # P5_Upsample
        # 1024 -> 512
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), ksize=1, stride=1,act=act
        )

        # C3_p4
        # [512, 512] -> 512
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # P4_Upsample
        # 512 -> 256
        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), ksize=1, stride=1, act=act
        )

        # C3_p3
        # [256,256] -> 256
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

        # P3_bottom-up
        # 256 -> 256
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), ksize=3, stride=2, act=act
        )

        # C3_n3
        # [256, 256] -> 512
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )


        # P4_bottom-up
        # 512 -> 512
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), ksize=3, stride=2, act=act
        )

        # C3_n4
        # [512, 512] -> 1024
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            shortcut=False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, inputs):
        # backbone
        out_features = self.backbone.forward(inputs)
        [x2, x1, x0] = [out_features[f] for f in self.in_features]

        # Stage5
        fpn_out0 = self.lateral_conv0(x0)    # 1024 -> 512/32
        f_out0 = self.upsample(fpn_out0)     # 512/16 

        # Stage4
        f_out0 = torch.cat([f_out0, x1], 1)  # 512 -> 1024/16
        f_out0 = self.C3_p4(f_out0)          # 1024 -> 512/16

        fpn_out1 = self.reduce_conv1(f_out0) # 512 -> 256/16
        f_out1 = self.upsample(fpn_out1)     # 256 -> 128/8

        # Stage3
        f_out1 = torch.cat([f_out1, x2], 1)  # 256 -> 512/8
        pan_out2 = self.C3_p3(f_out1)        # 512 -> 256/8 -> out
        
        # Stage4
        p_out1 = self.bu_conv2(pan_out2)     # 256 -> 256/16 
        p_out1 = torch.cat([p_out1, fpn_out1], 1) # 256->512/16
        pan_out1 = self.C3_n3(p_out1)        # 512 -> 512/16 ->out

        # stage5
        p_out0 = self.bu_conv1(pan_out1)     # 512 -> 512/32 
        p_out0 = torch.cat([p_out0, fpn_out0], 1) # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)        # 1024 -> 1024/32 ->out

        # 256/8, 512/16, 1024/32
        outputs = (pan_out2, pan_out1, pan_out0)

        return outputs



#---------------------------------------------------------------#
# YOLOFPN module. 
# Darknet 53 is the default backbone of this model.
#---------------------------------------------------------------#
class YOLOFPN(nn.Module):
    def __init__(
        self,
        depth=53,
        in_features=["dark3", "dark4", "dark5"],
    ):
        super().__init__()
        
        self.backbone = Darknet(depth)
        self.in_features = in_features

        # dark5_upsample
        # 512->256
        self.dark5_cbl = self.make_cbl(512, 256, ksize=1)
        self.dark5_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # dark4_embedding
        # [512, 256] -> 256
        self.dark4_embedding = self.make_embedding(512 + 256, [256, 512])

        # dark4_upsample
        # 256->128
        self.dark4_cbl = self.make_cbl(256, 128, ksize=1)
        self.dark4_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # dark3_embedding
        # [256, 128] -> 128
        self.dark3_embedding = self.make_embedding(256 + 128, [128, 256])
        

    def forward(self, inputs):
        # backbone
        out_features = self.backbone.forward(inputs)
        dark3, dark4, dark5 = [out_features[f] for f in self.in_features]

        # dark4
        dark5_upsample = self.dark5_upsample(self.dark5_cbl(dark5))
        dark4 = torch.cat([dark5_upsample, dark4], 1)
        dark4 = self.dark4_embedding(dark4)

        # dark3
        dark4_upsample = self.dark4_upsample(self.dark4_cbl(dark4))
        dark3 = torch.cat([dark4_upsample, dark3], 1)
        dark3 = self.dark3_embedding(dark3)

        return (dark3, dark4, dark5)


    def make_cbl(self, in_channels, out_channels, ksize=1):
        m = BaseConv(
                in_channels, out_channels, ksize=ksize, stride=1, act="lrelu"
            )
        return m

    def make_embedding(self, in_channels, channels_list):
        m = nn.Sequential(
            *[
                self.make_cbl(in_channels, channels_list[0], ksize=1),
                self.make_cbl(channels_list[0], channels_list[1], ksize=3),
                self.make_cbl(channels_list[1], channels_list[0], ksize=1),
                self.make_cbl(channels_list[0], channels_list[1], ksize=3),
                self.make_cbl(channels_list[1], channels_list[0], ksize=1),
            ]
        )
        return m

