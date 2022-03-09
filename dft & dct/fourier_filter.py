# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#%% （1）高通滤波器（提取轮廓）

#可视化高通滤波器
laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]])
f = np.fft.fft2(laplacian)
f1shift = np.fft.fftshift(f)
img = np.log(np.abs(f1shift))
plt.imshow(img,'gray')

img = Image.open('../images/origin/ILSVRC2012_val_00000005.jpeg').convert('L')#读取图片并转化为灰度图 
img = ImageOps.fit(img, (400, 400))#裁剪图片至指定大小
img = np.array(img)#转化为np.array
plt.subplot(121),plt.imshow(img,'gray'),plt.title('origial')
plt.xticks([]),plt.yticks([])
#--------------构造滤波器----------------
rows,cols = img.shape
mask = np.ones(img.shape,np.uint8)
mask[170:230, 170:230] = 0

f1 = np.fft.fft2(img)
f1shift = np.fft.fftshift(f1)
f1shift = f1shift*mask#用滤波器在频域进行操作（将低频成分过滤）
f2shift = np.fft.ifftshift(f1shift) #对新频域进行逆变换
img_new = np.fft.ifft2(f2shift)
#出来的是复数，无法显示
img_new = np.abs(img_new)
#进行minmax归一化方便显示
img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
plt.subplot(122),plt.imshow(img_new,'gray'),plt.title('Highpass')
plt.xticks([]),plt.yticks([])

#%% (2)低通滤波器（模糊轮廓）
mask = np.zeros(img.shape,np.uint8)
mask[170:230, 170:230] = 1

f1 = np.fft.fft2(img)
f1shift = np.fft.fftshift(f1)
f1shift = f1shift*mask
f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
img_new = np.fft.ifft2(f2shift)
#出来的是复数，无法显示
img_new = np.abs(img_new)
#调整大小范围便于显示
img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
plt.subplot(122),plt.imshow(img_new,'gray'),plt.title('Lowpass')
plt.xticks([]),plt.yticks([])

#%% (3)带通滤波器(保留一部分低频一部分高频)
mask1 = np.ones(img.shape,np.uint8)
mask1[192:208, 192:208] = 0
mask2 = np.zeros(img.shape,np.uint8)
mask2[120:280,120:280] = 1
mask = mask1*mask2

f1 = np.fft.fft2(img)#二维变换
f1shift = np.fft.fftshift(f1)#中心化
f1shift = f1shift*mask#滤波
f2shift = np.fft.ifftshift(f1shift) #对新的进行逆变换
img_new = np.fft.ifft2(f2shift)
#出来的是复数，无法显示
img_new = np.abs(img_new)
#调整大小范围便于显示
img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
plt.subplot(122),plt.imshow(img_new,'gray'),plt.title('Bandpass')
plt.xticks([]),plt.yticks([])
