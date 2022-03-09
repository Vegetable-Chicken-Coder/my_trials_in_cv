# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:04:02 2022

@author: 15123
"""
import cv2
import itertools
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import jpeg_utils
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
if __name__ == '__main__':
    ''' test JPEG compress and decompress'''
    img = Image.open('../images/origin/ILSVRC2012_val_00000005.jpeg') # W x H x 3
    img = ImageOps.fit(img, (400, 400))#裁剪图片至指定大小
    img = np.array(img) / 255.
    img_r = np.transpose(img, [2, 0, 1]) # 3 x W x H
    img_tensor = jpeg_utils.torch.from_numpy(img_r).unsqueeze(0).float()
    recover = jpeg_utils.jpeg_compress_decompress(img_tensor)

    recover_arr = recover.detach().squeeze(0).numpy()
    recover_arr = np.transpose(recover_arr, [1, 2, 0]) # W x H x 3

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(recover_arr)
    plt.show()
