'''
@Description:general utils
@Author:Zigar
@Date:2021/11/03 15:31:38
'''
import torch
import cv2
import numpy as np
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
    model = YOLOX(num_classes=num_classes, version=version)
    pretrained_state_dict = torch.load(weight_dir, map_location=lambda storage, loc: storage)["model"]
    model.load_state_dict(pretrained_state_dict, strict=True)
    model.to(device=device)
    model.eval()
    return model


#---------------------------------------------------------------#
# load souse
# include: image / image_path / video / video_path / webcom
#---------------------------------------------------------------#
def load_source(source):
    image = cv2.imread(source)
    assert image is not None, 'Image Not Found ' + source
    return image


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
    # BGR2RGB(do not support np.float64)
    padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    return padded_img, ratio
