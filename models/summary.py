'''
@Description:Model Summary
@Author:Zigar
@Date:2021/11/03 13:30:50
'''

import torch
from torchsummary import summary
from yolox import YOLOX

if __name__ == '__main__':
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m       = YOLOX(80, "nano").to(device)
    
    summary(m, input_size=(3, 640, 640))