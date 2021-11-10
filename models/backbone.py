'''
@Description:backbone: CSPDarknet/Darknet
@Author:Zigar
@Date:2021/11/01 11:17:46
'''

from torch import nn
from models.common import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


#---------------------------------------------------------------#
# CSPDarkNet
# for yolox, also for yolov5
#---------------------------------------------------------------#
class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features = ("dark3", "dark4", "dark5"),
        depthwise    = False,
        act          = "silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        # desired output layer name
        self.out_features = out_features
        # DWConv or BaseConv(DWConv use in yolox-nano)
        Conv = DWConv if depthwise else BaseConv

        # use depth and width to control yolox's version
        # (stem->dark2->dark3->dark4->dark5)
        # defaul depth: [1, 3, 9, 9, 3]  
        base_depth = max(round(dep_mul * 3), 1) # 3
        depths = [int(depth * base_depth) for depth in [1, 3, 3, 1]]
        # defaul witdh: [64, 128, 256, 512, 1024]
        base_channels = int(wid_mul * 64)  # 64
        in_channels = [int(width * base_channels) for width in [2, 4, 8, 16]]

        # stem = Focus
        # 3 -> 12 -> 64
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        
        # dark2 = Downsample + CSPLayer
        # 64 -> 128
        self.dark2 = nn.Sequential(
            Conv(
                base_channels, in_channels[0], ksize=3, stride=2, act=act
            ),
            CSPLayer(
                in_channels  = in_channels[0], # 128
                out_channels = in_channels[0], # 128
                n            = depths[0],   # 3
                depthwise    = depthwise,
                act          = act
            )
        )

        # dark3 = Downsample + CSPLayer
        # 128 -> 256
        self.dark3 = nn.Sequential(
            Conv(
                in_channels[0], in_channels[1], ksize=3, stride=2, act=act
            ),
            CSPLayer(
                in_channels  = in_channels[1], # 256
                out_channels = in_channels[1], # 256
                n            = depths[1],   # 9
                depthwise    = depthwise,
                act          = act
            )
        )

        # dark4 = Downsample + CSPLayer
        # 256 -> 512
        self.dark4 = nn.Sequential(
            Conv(
                in_channels[1], in_channels[2], ksize=3, stride=2, act=act
            ),
            CSPLayer(
                in_channels  = in_channels[2], # 512
                out_channels = in_channels[2], # 512
                n            = depths[2],   # 9
                depthwise    = depthwise,
                act          = act
            )
        )

        # dark5 = Downsample + SPPBottleneck + CSPLayer
        # The last CSPLayer's shortcut is False
        # Otherwise there will be a loss of accuracy
        # 512 -> 1024 -> 1024
        self.dark5 = nn.Sequential(
            Conv(
                in_channels[2], in_channels[3], ksize=3, stride=2, act=act
            ),
            SPPBottleneck(
                in_channels[3], in_channels[3], act=act
            ),
            CSPLayer(
                in_channels  = in_channels[3], # 1024
                out_channels = in_channels[3], # 1024
                n            = depths[3],   # 3
                shortcut     = False,
                depthwise    = depthwise,
                act          = act
            )
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}



#---------------------------------------------------------------#
# Darknet
# include Darknet21 and Darknet53
# for yolov3-spp and yolov3-tiny
#---------------------------------------------------------------#
class Darknet(nn.Module):
    # number of blocks from dark21 to dark53
    depth2blocks = {21: [1, 1, 2, 2, 1], 53: [1, 2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels       = 3,
        stem_out_channels = 32,
        out_features      = ("dark3", "dark4", "dark5"),
        act = "lrelu"
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        # desired output layer name
        self.out_features = out_features
        # activation
        self.act = act
        # number of each blocks in darknet21 or darnet53
        num_blocks = Darknet.depth2blocks[depth]

        # stage1
        # 3 -> 32 -> 64
        self.stem = nn.Sequential(
            BaseConv(
                in_channels, stem_out_channels, ksize=3, stride=1, act=act
            ),
            *self.make_group_layer(stem_out_channels, stem_out_channels*2, num_blocks[0], stride=2)
        )

        # stage2
        # 64 -> 128
        in_channels = stem_out_channels * 2 # 64
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, in_channels*2, num_blocks[1], stride=2)
        )

        # stage3
        # 128 -> 256
        in_channels *= 2 # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, in_channels*2, num_blocks[2], stride=2)
        )

        # stage4
        # 256 -> 512
        in_channels *= 2 # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, in_channels*2, num_blocks[3], stride=2)
        )

        # stage5
        # 512 -> 1024 -> 512
        in_channels *= 2 # 512
        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, in_channels * 2, num_blocks[4], stride=2),
            *self.make_spp_layer(in_channels * 2, in_channels, stride=1)
        )
        

    # make group of Layer
    # Downsample + ResLayer
    def make_group_layer(self, in_channels, out_channls, num_blocks, stride=1):
        m = nn.Sequential(
            *[
                BaseConv(
                    in_channels, out_channls, ksize=3, stride=stride, act=self.act
                ),
                *[
                    (ResLayer(out_channls)) for _ in range(num_blocks)
                ]
            ]
        )
        return m

    # make spp layer
    # BaseConv -> SPPBottleneck -> BaseConv
    def make_spp_layer(self, in_channels, out_channels, stride=1):
        hidden_channels = in_channels // 2
        m = nn.Sequential(
            *[
                BaseConv(
                    in_channels, hidden_channels, ksize=1, stride=stride, act=self.act
                ),
                BaseConv(
                    hidden_channels, in_channels, ksize=3, stride=stride, act=self.act
                ),
                SPPBottleneck(
                    in_channels, hidden_channels, act=self.act
                ),
                BaseConv(
                    hidden_channels, in_channels, ksize=3, stride=stride, act=self.act
                ),
                BaseConv(
                    in_channels, out_channels, ksize=1, stride=stride, act=self.act
                )
            ] 
        )
        return m


    def forward(self, x):
        output = {}
        x = self.stem(x)
        output["stem"] = x
        x = self.dark2(x)
        output["dark2"] = x
        x = self.dark3(x)
        output["dark3"] = x
        x = self.dark4(x)
        output["dark4"] = x
        x = self.dark5(x)
        output["dark5"] = x
        return {k:v for k, v in output.items() if k in self.out_features}



if __name__ == '__main__':
    print(CSPDarknet(dep_mul=1, wid_mul=1))