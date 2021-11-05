'''
@Description:predict
@Author:Zigar
@Date:2021/11/03 14:12:30
'''
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

import numpy as np
import cv2
from torch.nn.modules.module import T

from utils.general import get_label, get_model, load_source, padding_img
from utils.boxes import decode_outputs, xywh2x1y1x2y2
from utils.plots import vis


class Predictor(object):
    def __init__(
        self,
        input_size,
        device,
        num_classes,
        conf_thres,
        iou_thres,
        label_dir,
        weights_dir,
        version,
        source
    ):
        super().__init__()
        # inference size (pixels)
        self.input_size = input_size

        self.device = torch.device(device)
        self.num_classes = num_classes

        # threshold in postprocess
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # inference dtype
        self.dtype = torch.float32

        # label
        self.label = get_label(label_dir)
        # model
        self.model = get_model(self.num_classes, version, weights_dir, self.device)
        # source: image / image_path / video / video_path / webcom
        self.source = load_source(source)


    def predict(self):
        # data preprocess
        inputs, ratio = self.preprocess()

        # model inference
        outputs = self.model(inputs)
        
        # data postprocess
        results = self.postprocess(outputs, ratio)

        if results is not None:
            bboxes, scores, labels = results
            image = vis(self.source, bboxes, scores, labels, conf=self.conf_thres, class_names=self.label)
        cv2.imwrite('1.jpg', image)
        return results

    def preprocess(self):
        # resize and padding for image inference
        inputs, ratio = padding_img(self.source, self.input_size)
        # (H,W,C) ——> (C,H,W)
        inputs = inputs.transpose(2, 0, 1)
        # (C,H,W) ——> (1,C,H,W)
        inputs = inputs[np.newaxis, :, :, :]
        # NumpyArray -> Tensor.float32
        inputs = torch.from_numpy(inputs.copy())
        inputs = inputs.type(self.dtype)
        # select device
        inputs = inputs.to(self.device)
        return inputs, ratio

        

    def postprocess(self, outputs, ratio):
        # decode models outputs
        outputs = decode_outputs(outputs, self.input_size)
        
        # xywh2x1y1x2y2
        outputs = xywh2x1y1x2y2(outputs)
        
        # delate batch
        image_pred = outputs[0]
        
        # get the max class conf and index(class pred)
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)
        conf_score = image_pred[:, 4].unsqueeze(dim=1) * class_conf

        # detections.shape=[8400,6]
        detections = torch.cat((image_pred[:, :4], conf_score, class_pred.float()), 1)

        # conf_thresh
        conf_mask = (conf_score >= self.conf_thres).squeeze()
        detections = detections[conf_mask]

        # no target's conf_score >= conf_thresh
        if not detections.size(0):
            return None

        # nms
        nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4], detections[:, 5],
                                                    self.iou_thres)
        detections = detections[nms_out_index]

        # anti_resize
        detections = detections.data.cpu().numpy()
        bboxes = (detections[:, :4] / ratio).astype(np.int64)
        scores = detections[:, 4]
        labels = detections[:, 5].astype(np.int64)

        return bboxes, scores, labels
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device,cuda or cpu')
    parser.add_argument('--num_classes', type=int, default=80, help='number of classes')
    parser.add_argument('--weights-dir', nargs='+', type=str, default='../weights/yolox_tiny.pth', help='model.pth path(s)')
    parser.add_argument('--version', type=str, default='tiny', help='version of models')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--label-dir', type=str, default='../labels/coco_label.txt', help='label names')
    
    parser.add_argument('--source', type=str, default='../data/img/street.jpg', help='source')  # file/folder, 0 for webcam

    # TODO
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # parser.add_argument('--name', default='exp', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)


    with torch.no_grad():
        Predictor(
            opt.input_size,
            opt.device,
            opt.num_classes,
            opt.conf_thres,
            opt.iou_thres,
            opt.label_dir,
            opt.weights_dir,
            opt.version,
            opt.source
        ).predict()

