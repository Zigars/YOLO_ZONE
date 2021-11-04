'''
@Description:Model Summary
@Author:Zigar
@Date:2021/11/03 13:30:50
'''

import torch
from torchsummary import summary
from yolox import YOLOX

if __name__ == '__main__':
    # cuda or cpu
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YOLOX(80, "nano").to(device)
    
    print(m)
    summary(m, input_size=(3, 640, 640))