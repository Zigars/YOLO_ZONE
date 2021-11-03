'''
@Description:
Common network module:
BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
@Author:Zigar
@Date:2021/10/31 19:57:43
'''

import torch
import torch.nn as nn

#---------------------------------------------------------------#
# Get activate
# SiLU/ReLU/LeakyReLU
#---------------------------------------------------------------#
def get_activation(name="silu", inplace=True):
    if   name == "silu":
         module = nn.SiLU(inplace=inplace)
    elif name == "relu":
         module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
         module = nn.LeakyReLU(0.1, inplace=inplace)
    else: 
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


#---------------------------------------------------------------#
# Basic Convolution
# Conv2D -> BatchNorm -> Activate(SiLU/ReLU/LeakyReLU)
#---------------------------------------------------------------#
class BaseConv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        ksize, 
        stride,
        groups=1, 
        bias=False, 
        act="silu" 
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = ksize,
            stride       = stride,
            padding      = pad,
            groups       = groups,
            bias         = bias,
        )
        self.bn  = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


#---------------------------------------------------------------#
# Depthwise Convolution
# dconv -> pconv
#---------------------------------------------------------------#
class DWConv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        ksize, 
        stride=1, 
        act="silu"
    ):
        super().__init__()
        self.dconv = BaseConv(
            in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1,act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


#---------------------------------------------------------------#
# Standard Bottleneck
# used in CSPDarknet
# BaseConv_1*1 -> BaseConv_3*3
# shortcut or not shortcut
#---------------------------------------------------------------#
class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut  = True,
        expansion = 0.5,
        depthwise = False,
        act       = "silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, act=act
        )
        self.conv2 = Conv(
            hidden_channels, out_channels, ksize=3, stride=1, act=act
        )
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


#---------------------------------------------------------------#
# Resdual layer with 'in_channels' input
# used in Darknet
# BaseConv_1*1 -> BaseConv_3*3 
# shortcut
#---------------------------------------------------------------#
class ResLayer(nn.Module):
    def __init__(
        self,
        in_channels: int
    ):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu" 
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


#---------------------------------------------------------------#
# Spatial pyramid pooling layer
# used in YOLOv3-spp 
# BaseConv -> Concat[MaxPool2d * [1, 5, 9, 13]] -> BaseConv
#---------------------------------------------------------------#
class SPPBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksizes = (5, 9, 13),
        act     = "silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, act=act
        )
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k_s, stride=1, padding=k_s // 2) for k_s in ksizes
            ]
        )
        conv2_channels = hidden_channels * (len(ksizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, ksize=1, stride=1, act=act
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


#---------------------------------------------------------------#
# CSP Bottleneck with 3 Convolution
# C3 in yolov5
# BaseConv -> Cat[BaseConv, BaseConv -> Bottleneck*n] -> BaseConv
#---------------------------------------------------------------#
class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n         = 1,
        shortcut  = True,
        expansion = 0.5,
        depthwise = False,
        act       = "silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, act=act
        )
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, act=act
        )
        self.conv3 = BaseConv(
            2 * hidden_channels, out_channels, ksize=1, stride=1, act=act
        )
        module_list = [
            Bottleneck(
                hidden_channels, 
                hidden_channels, 
                shortcut,
                expansion=1.0,
                depthwise=depthwise, 
                act=act 
            ) 
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


#---------------------------------------------------------------#
# Focus
# focus width and height information into channel space
# (b,c,w,h) -> (b,4c,w/2,h/2)
#---------------------------------------------------------------#
class Focus(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        ksize  = 1,
        stride = 1,
        act    = "silu"
    ):
        super().__init__()
        self.conv = BaseConv(
            in_channels * 4, out_channels, ksize, stride, act=act
        )
    
    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
