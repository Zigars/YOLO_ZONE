'''
@Description:predict
@Author:Zigar
@Date:2021/11/03 14:12:30
'''

import torch
import torch.nn as nn

import numpy as np
import cv2

from models.yolox import YOLOX
from utils.plots import vis

