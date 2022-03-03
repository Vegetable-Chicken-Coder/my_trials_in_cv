# -*- coding: utf-8 -*-
from dataset import StegaData
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from torch.nn import functional as F
batchsize = 1

def main():
    #初始化数据集
    dataset = StegaData(data_path=r'D:\Research\Datasets\mirflickr25k\mirflickr',secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batchsize, shuffle=True, pin_memory=True)

    #输出原始图像    
    image_input, secret_input = next(iter(dataloader))#返回第一个值
    img = transforms.ToPILImage(mode='RGB')(image_input[0])
    img.save("./affine_trial/origin.jpg")     
    
    #[1]平移：向右0.2,向下0.4
    theta = torch.tensor([
    [1, 0, -0.2],
    [0, 1, -0.4]
    ], dtype=torch.float)
    # theta: batchsize x 2 x 3, image_input: batchsize x 3 x 400 x 400
    grid = F.affine_grid(theta.unsqueeze(0), image_input.size(), align_corners=False)        
    transformed_image = F.grid_sample(image_input, grid, align_corners=False)
    img = transforms.ToPILImage(mode='RGB')(transformed_image[0])
    img.save("./affine_trial/pingyi.jpg")     

    #[2]缩放：长度方向方法2倍，宽度方向缩减一半
    theta = torch.tensor([
    [2, 0, 0],
    [0, 0.5, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), image_input.size(), align_corners=False)        
    transformed_image = F.grid_sample(image_input, grid, align_corners=False)
    img = transforms.ToPILImage(mode='RGB')(transformed_image[0])
    img.save("./affine_trial/suofang.jpg")      

    #[3]旋转：顺时针旋转30°
    angle = -30*math.pi/180
    theta = torch.tensor([
    [math.cos(angle),math.sin(-angle),0],
    [math.sin(angle),math.cos(angle) ,0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), image_input.size(), align_corners=False)        
    transformed_image = F.grid_sample(image_input, grid, align_corners=False)
    img = transforms.ToPILImage(mode='RGB')(transformed_image[0])
    img.save("./affine_trial/xuanzhuan.jpg") 

    #【4】转置、图片resize
    theta = torch.tensor([
    [0, 1, 0],
    [1, 0, 0]
    ], dtype=torch.float)
    #resize H x W -> W x H
    B, C, H, W = image_input.size()
    grid = F.affine_grid(theta.unsqueeze(0), torch.Size((B, C, W, H)), align_corners=False)      
    transformed_image = F.grid_sample(image_input, grid, align_corners=False)
    img = transforms.ToPILImage(mode='RGB')(transformed_image[0])
    img.save("./affine_trial/zhuanzhi.jpg")  
    
if __name__ == '__main__':
    main()

