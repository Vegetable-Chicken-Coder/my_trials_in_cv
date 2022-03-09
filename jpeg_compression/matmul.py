# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:52:35 2022

@author: 15123
"""
import torch
import numpy as np
import torch.nn as nn
import jpeg_utils
from PIL import Image, ImageOps

img = Image.open('../images/origin/ILSVRC2012_val_00000005.jpeg') # [333, 500, 3]
img = np.array(img) / 255.
img_r = np.transpose(img, [2, 0, 1]) # [3, 333, 500]
img_tensor = torch.from_numpy(img_r).unsqueeze(0).float() # [1, 3, 333, 500]
avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2), count_include_pad=False) 
cb = avg_pool(img_tensor[:, 1, :, :].unsqueeze(1))# [1, 166, 250] -> [1, 1, 166, 250]
cr = avg_pool(img_tensor[:, 2, :, :].unsqueeze(1))# [1, 166, 250] -> [1, 1, 166, 250]
cb = cb.permute(0, 2, 3, 1) # [1, 166, 250, 1]
cr = cr.permute(0, 2, 3, 1) # [1, 166, 250, 1]
cb = cb.squeeze(3) # [1, 166, 250, 1]
cr = cr.squeeze(3) # [1, 166, 250, 1]


