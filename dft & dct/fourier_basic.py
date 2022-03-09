# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% (1)中心化操作——np.fft.fftshift
img = cv2.imread('../images/origin/ILSVRC2012_val_00000005.jpeg',0) #直接读为灰度图像
f = np.fft.fft2(img)#进行快速傅里叶变换
fshift = np.fft.fftshift(f)
#取绝对值：将复数变化成实数
#取对数的目的为了将数据变化到较小的范围（比如0-255）
s1 = np.log(np.abs(f))
s2 = np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(s1,'gray'),plt.title('original')#不进行中心化的图片
plt.subplot(122),plt.imshow(s2,'gray'),plt.title('center')#进行中心化后的图片

#%% (2)傅里叶逆变换——np.fft.ifft2
plt.subplot(131),plt.imshow(img,'gray'),plt.title('original')
plt.subplot(132),plt.imshow(s2,'gray'),plt.title('center')
# 逆变换
f1shift = np.fft.ifftshift(fshift)#逆中心化
img_back = np.fft.ifft2(f1shift)#逆傅里叶
#出来的是复数，无法显示
img_back = np.abs(img_back)#将复数转化为实数
plt.subplot(133),plt.imshow(img_back,'gray'),plt.title('img back')


