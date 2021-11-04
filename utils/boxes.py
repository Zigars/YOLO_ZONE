'''
@Description:boxes related
@Author:Zigar
@Date:2021/11/03 12:56:02
'''


 #---------------------------------------------------------------#
 # decode models outputs
 #---------------------------------------------------------------#
from numpy.core.fromnumeric import shape
import torch


def decode_outputs(outputs, input_size):
    grids = []
    strides = []

    hw  = [x.shape[-2:] for x in outputs]

    # [bs, reg+pred+nc, h, w] * 3stride
    # -> [(bs, h * w * 3 stride, reg+pred+nc)]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    
    for h, w in hw:
        # generate grid
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((*shape, 1), input_size / h))


    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())

    # decode
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

    return outputs


#---------------------------------------------------------------#
# xywh2x1y1x2y2 for inference
#---------------------------------------------------------------#
def xywh2x1y1x2y2(outputs):
    box_corner = outputs.new(outputs.shape)
    box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2
    box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2
    box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2
    box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2
    outputs[:, :, :4] = box_corner[:, :, :4]
    return outputs
