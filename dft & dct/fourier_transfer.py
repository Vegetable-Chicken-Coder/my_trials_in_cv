# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

img_bed=Image.open('../images/origin/ILSVRC2012_val_00000005.jpeg').convert('L')#读取图片并转化为灰度图 
img_bed=ImageOps.fit(img_bed, (400, 400))#裁剪图片至指定大小
img_bed = np.array(img_bed)#转化为np.array

img_cup=Image.open('../images/origin/ILSVRC2012_val_00000004.jpeg').convert('L')#读取图片并转化为灰度图 
img_cup=ImageOps.fit(img_cup, (400, 400))#裁剪图片至指定大小
img_cup = np.array(img_cup)#转化为np.array

plt.subplot(221),plt.imshow(img_bed,'gray'),plt.title('img_bed')
plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(img_cup,'gray'),plt.title('img_cup')
plt.xticks([]),plt.yticks([])
#--------------------------------
f1 = np.fft.fft2(img_bed)
f1shift = np.fft.fftshift(f1)
f1_A = np.abs(f1shift) #取振幅
f1_P = np.angle(f1shift) #取相位
#--------------------------------
f2 = np.fft.fft2(img_cup)
f2shift = np.fft.fftshift(f2)
f2_A = np.abs(f2shift) #取振幅
f2_P = np.angle(f2shift) #取相位
#---Bed Amplitude + Cup Phase--------------------
img_new1_f = np.zeros(img_bed.shape,dtype=complex) 
img1_real = f1_A*np.cos(f2_P) #取实部
img1_imag = f1_A*np.sin(f2_P) #取虚部
img_new1_f.real = np.array(img1_real) 
img_new1_f.imag = np.array(img1_imag) 
f3shift = np.fft.ifftshift(img_new1_f) #对新的进行逆变换
img_new1 = np.fft.ifft2(f3shift)
#出来的是复数，无法显示
img_new1 = np.abs(img_new1)
#调整大小范围便于显示
img_new1 = (img_new1-np.amin(img_new1))/(np.amax(img_new1)-np.amin(img_new1))
plt.subplot(223),plt.imshow(img_new1,'gray'),plt.title('Bed Amplitude + Cup Phase')
plt.xticks([]),plt.yticks([])
#---Cup Amplitude + Bed Phase--------------------
img_new2_f = np.zeros(img_cup.shape,dtype=complex) 
img2_real = f2_A*np.cos(f1_P) #取实部
img2_imag = f2_A*np.sin(f1_P) #取虚部
img_new2_f.real = np.array(img2_real) 
img_new2_f.imag = np.array(img2_imag) 
f4shift = np.fft.ifftshift(img_new2_f) #对新的进行逆变换
img_new2 = np.fft.ifft2(f4shift)
#出来的是复数，无法显示
img_new2 = np.abs(img_new2)
#调整大小范围便于显示
img_new2 = (img_new2-np.amin(img_new2))/(np.amax(img_new2)-np.amin(img_new2))
plt.subplot(224),plt.imshow(img_new2,'gray'),plt.title('Cup Amplitude + Bed Phase')
plt.xticks([]),plt.yticks([])
