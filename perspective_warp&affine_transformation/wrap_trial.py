# -*- coding: utf-8 -*-
import random
import numpy as np
import torchgeometry
import cv2
from dataset import StegaData
import torch
import model
from torch.utils.data import DataLoader
from torchvision import transforms

width, height = 400, 400
rnd_trans = 0.1
batchsize = 1
borders = 'no_edge'

def get_rand_transform_matrix(image_size, d, batch_size):
    Ms = np.zeros((batch_size, 2, 3, 3))
    for i in range(batch_size):
        tl_x = random.uniform(-d, d)     # Top left corner, top
        tl_y = random.uniform(-d, d)    # Top left corner, left
        bl_x = random.uniform(-d, d)   # Bot left corner, bot
        bl_y = random.uniform(-d, d)    # Bot left corner, left
        tr_x = random.uniform(-d, d)     # Top right corner, top
        tr_y = random.uniform(-d, d)   # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)   # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y +  image_size]], dtype = "float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
#注意！同一图像经过Ms_0的变形再经过Ms_1的变形并不会返回原状！！
        Ms[i, 0, :, :] = M_inv #Ms_0 与 Ms_1 互为逆矩阵
        Ms[i, 1, :, :] = M 
    Ms = torch.from_numpy(Ms).float()
    return Ms

def main():
    Ms = get_rand_transform_matrix(width, np.floor(width * rnd_trans), batchsize)
    encoder = model.StegaStampEncoder()
    
    #初始化数据集
    dataset = StegaData(data_path=r'D:\Research\Datasets\mirflickr25k\mirflickr',secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batchsize, shuffle=True, pin_memory=True)
    
    #输出原始图像    
    image_input, secret_input = next(iter(dataloader))#返回第一个值
    img = transforms.ToPILImage(mode='RGB')(image_input[0])
    img.save("./wrap_trial/origin.jpg") 
    
    #对原始图像进行形变 image input -> warped_input   
    input_warped = torchgeometry.warp_perspective(image_input, Ms[:, 1, :, :], dsize=(400, 400), flags='bilinear') 
    '''
    img = transforms.ToPILImage(mode='RGB')(input_warped[0])
    img.save("./images_trial/origin1.jpg")  
    input_unwarped = torchgeometry.warp_perspective(image_input, Ms[:, 0, :, :], dsize=(400, 400), flags='bilinear') 
    img = transforms.ToPILImage(mode='RGB')(input_unwarped[0])
    img.save("./wrap_trial/origin2.jpg") 
    '''             
    #补齐上一步形变造成的黑边
    mask_warped = torchgeometry.warp_perspective(torch.ones_like(input_warped), Ms[:, 1, :, :], dsize=(400, 400),flags='bilinear')
    input_warped += (1 - mask_warped) * image_input
    img = transforms.ToPILImage(mode='RGB')(input_warped[0])
    img.save("./wrap_trial/warped_input.jpg")    
    
    #对形变后的图片进行编码得到形变后的residual warped_input -> residual_warped
    residual_warped = encoder((secret_input, input_warped)) #对图像进行编码
    # 形变后的编码图 = 形变后的residual + 形变后的input
    encoded_warped = residual_warped + input_warped
    img = transforms.ToPILImage(mode='RGB')(encoded_warped[0])
    img.save("./wrap_trial/encoded_warped.jpg")
    
    #计算不经过形变的residual,即通过形变后residual的进一步形变获得
    residual = torchgeometry.warp_perspective(residual_warped, Ms[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    
    # no edge
    encoded_image = image_input + residual
    img = transforms.ToPILImage(mode='RGB')(encoded_image[0])
    img.save("./wrap_trial/no_edge.jpg")
    
    # black
    encoded_image = residual_warped + input_warped
    encoded_image = torchgeometry.warp_perspective(encoded_image, Ms[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    img = transforms.ToPILImage(mode='RGB')(encoded_image[0])
    img.save("./wrap_trial/black.jpg")

    # white
    mask = torchgeometry.warp_perspective(torch.ones_like(residual), Ms[:, 0, :, :], dsize=(400, 400),flags='bilinear')
    encoded_image = residual_warped + input_warped
    encoded_image = torchgeometry.warp_perspective(encoded_image, Ms[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    encoded_image += (1 - mask) * torch.ones_like(residual)
    img = transforms.ToPILImage(mode='RGB')(encoded_image[0])
    img.save("./wrap_trial/white.jpg")    

    # image
    mask = torchgeometry.warp_perspective(torch.ones_like(residual), Ms[:, 0, :, :], dsize=(400, 400),flags='bilinear')
    encoded_image = residual_warped + input_warped
    encoded_image = torchgeometry.warp_perspective(encoded_image, Ms[:, 0, :, :], dsize=(400, 400), flags='bilinear')
    encoded_image += (1 - mask) * torch.roll(image_input, 1, 0)
    img = transforms.ToPILImage(mode='RGB')(encoded_image[0])
    img.save("./wrap_trial/image.jpg")    

if __name__ == '__main__':
    main()

