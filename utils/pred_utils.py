'''
@Description:utils for predict
@Author:Zigar
@Date:2021/11/09 09:38:57
'''
import time

import torch
import cv2
import numpy as np

import torch.nn as nn
import torchvision

from models.yolox import YOLOX


#---------------------------------------------------------------#
# get labels names
#---------------------------------------------------------------#
def get_label(label_dir):
    label_names = open(label_dir, 'r').readlines()
    label_names = [line.strip('\n') for line in label_names]
    return label_names


#---------------------------------------------------------------#
# load models
#---------------------------------------------------------------#
def get_model(num_classes, version, weight_dir, device):
    # print(version)
    print("Loading model: YOLOX, Version: {}".format(version))
    t0 = time.time()
    # load model
    model = YOLOX(num_classes=num_classes, version=version)

    # nano Conv's nn.BatchNorm2d need set different value
    if version == "nano":
        def init_yolo(Module):
            for m in Module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        model.apply(init_yolo)

    # load pretrained weights
    pretrained_state_dict = torch.load(weight_dir, map_location=lambda storage, loc: storage)["model"]
    model.load_state_dict(pretrained_state_dict, strict=True)

    model.to(device=device).eval()
    t1 = time.time()
    print(f'Done. ({t1 - t0:.3f}s)')
    return model




#---------------------------------------------------------------#
# resize and padding for image inference
#---------------------------------------------------------------#
def padding_img(image, resize_size):
    # original image size
    ori_h, ori_w = image.shape[:2]
    # padding img mask
    padded_img = np.ones((resize_size, resize_size, 3), np.float32) * 114.0
    # ratio of original image and padding image
    ratio = min(resize_size / ori_h, resize_size / ori_w)
    # resize original to image_size
    resized_img = cv2.resize(image, (int(ori_w * ratio), int(ori_h * ratio)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    # put resized_img in left-up of mask
    padded_img[: int(ori_h * ratio), : int(ori_w * ratio)] = resized_img
    return padded_img, ratio


#---------------------------------------------------------------#
# decode models outputs
#---------------------------------------------------------------#
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


#---------------------------------------------------------------#
# preprocess for one inference
#---------------------------------------------------------------#
def preprocess(img, input_size, dtype, device):
    # resize and padding for image inference
    inputs, ratio = padding_img(img, input_size)

    # BGR2RGB(do not support np.float64)
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
    # (H,W,C) ——> (C,H,W)
    inputs = inputs.transpose(2, 0, 1)
    # (C,H,W) ——> (1,C,H,W)
    inputs = inputs[np.newaxis, :, :, :]
    # NumpyArray -> Tensor.float32
    inputs = torch.from_numpy(inputs.copy())
    inputs = inputs.type(dtype)
    # select device
    inputs = inputs.to(device)
    return inputs, ratio


#---------------------------------------------------------------#
# postprocess for one inference
#---------------------------------------------------------------#
def postprocess(outputs, ratio, input_size, num_classes, conf_thres, iou_thres):
        # decode models outputs
        outputs = decode_outputs(outputs, input_size)
        
        # xywh2x1y1x2y2
        image_pred = xywh2x1y1x2y2(outputs)[0]

        # get the max class conf and index(class pred)
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_score = image_pred[:, 4].unsqueeze(dim=1) * class_conf

        # detections.shape=[8400,6]
        detections = torch.cat((image_pred[:, :4], conf_score, class_pred.float()), 1)

        # conf_thresh
        conf_mask = (conf_score >= conf_thres).squeeze()
        detections = detections[conf_mask]

        # no target's conf_score >= conf_thresh
        if not detections.size(0):
            return None

        # nms
        nms_out_index = torchvision.ops.boxes.batched_nms(detections[:, :4], detections[:, 4], detections[:, 5],
                                                    iou_thres)
        detections = detections[nms_out_index]

        # anti_resize
        detections = detections.data.cpu().numpy()
        bboxes = (detections[:, :4] / ratio).astype(np.int64)
        scores = detections[:, 4]
        cls_id = detections[:, 5].astype(np.int64)

        return bboxes, scores, cls_id


#---------------------------------------------------------------#
# pytorch-accurate time
#---------------------------------------------------------------#
def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()