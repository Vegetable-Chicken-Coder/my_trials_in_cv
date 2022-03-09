# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 17:15:53 2022

@author: 15123
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
#%% 读取图片
img = Image.open('../images/origin/ILSVRC2012_val_00000005.jpeg').convert('L')#读取图片并转化为灰度图 
img = ImageOps.fit(img, (400, 400))#裁剪图片至指定大小
img = np.array(img)#转化为np.array
img1 = img.astype('float')

#%% 自编写函数实现 
#https://blog.csdn.net/James_Ray_Murphy/article/details/79173388
C_temp = np.zeros(img.shape)
dst = np.zeros(img.shape)
 
m, n = img.shape
N = n
C_temp[0, :] = 1 * np.sqrt(1/N)
 
for i in range(1, m):
     for j in range(n):
          C_temp[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )
) * np.sqrt(2 / N )
 
dst = np.dot(C_temp , img1)
dst = np.dot(dst, np.transpose(C_temp))
 
dst1= np.log(abs(dst))  #进行log处理
 
img_recor = np.dot(np.transpose(C_temp) , dst)
img_recor1 = np.dot(img_recor, C_temp)

#%% 使用cv2.dct实现
 
img_dct = cv2.dct(img1)         #进行离散余弦变换
 
img_dct_log = np.log(abs(img_dct))  #进行log处理
 
img_recor2 = cv2.idct(img_dct)    #进行离散余弦反变换
#%% 可视化
plt.subplot(231)
plt.imshow(img1, 'gray')
plt.title('original')
plt.xticks([]), plt.yticks([])

plt.subplot(232)
plt.imshow(dst1,'gray')
plt.title('DCT_manual')
plt.xticks([]), plt.yticks([])
 
plt.subplot(233)
plt.imshow(img_recor1, 'gray')
plt.title('IDCT_manual')
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(img, 'gray')
plt.title('original')
 
plt.subplot(235)
plt.imshow(img_dct_log,'gray')
plt.title('DCT2_cv2')
 
plt.subplot(236)
plt.imshow(img_recor2,'gray')
plt.title('IDCT2_cv2')
 
plt.show()
