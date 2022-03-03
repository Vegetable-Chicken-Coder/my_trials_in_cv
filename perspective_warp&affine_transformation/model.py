import sys
import os
import utils
import torch
import numpy as np
from torch import nn
import torchgeometry
from kornia import color
import torch.nn.functional as F
from torchvision import transforms

#线性层+relu
class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        # initialization
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))#注意此处padding
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)#拉平向量


class StegaStampEncoder(nn.Module):
    def __init__(self):
        super(StegaStampEncoder, self).__init__()
        #将输入编码转换成7500 = 3 * 50 * 50维，方便嵌入图像
        self.secret_dense = Dense(100, 7500, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, inputs):
        secrect, image = inputs
        secrect = secrect - .5 # 4 x 100
        image = image - .5 # 4 x 3 x 400 x 400

        #将输入编码转换成7500 = 3 * 50 * 50维，方便嵌入图像
        secrect = self.secret_dense(secrect)
        secrect = secrect.reshape(-1, 3, 50, 50) # 4 x 100 -> 4 x 3 x 50 x 50
        secrect_enlarged = nn.Upsample(scale_factor=(8, 8))(secrect) # 4 x 3 x 50 x 50 -> 4 x 3 x 400 x 400

        inputs = torch.cat([secrect_enlarged, image], dim=1) # 4 x 6 x 400 x 400
        conv1 = self.conv1(inputs) # 4 x 32 x 400 x 400
        conv2 = self.conv2(conv1) # 4 x 32 x 200 x 200
        conv3 = self.conv3(conv2) # 4 x 64 x 100 x 100
        conv4 = self.conv4(conv3) # 4 x 128 x 50 x 50
        conv5 = self.conv5(conv4) # 4 x 256 x 25 x 25
        up6 = self.up6(nn.Upsample(scale_factor=(2, 2))(conv5)) # 4 x 256 x 50 x 50 -> 4 x 128 x 50 x 50
        merge6 = torch.cat([conv4, up6], dim=1) # 4 x 256 x 50 x 50
        conv6 = self.conv6(merge6) # 4 x 128 x 50 x 50
        up7 = self.up7(nn.Upsample(scale_factor=(2, 2))(conv6)) # 4 x 128 x 100 x 100 -> 4 x 64 x 100 x 100
        merge7 = torch.cat([conv3, up7], dim=1) # 4 x 128 x 100 x 100
        conv7 = self.conv7(merge7) # 4 x 64 x 100 x 100
        up8 = self.up8(nn.Upsample(scale_factor=(2, 2))(conv7)) # 4 x 64 x 200 x 200 -> 4 x 32 x 200 x 200
        merge8 = torch.cat([conv2, up8], dim=1)# 4 x 64 x 200 x 200
        conv8 = self.conv8(merge8) # 4 x 32 x 200 x 200
        up9 = self.up9(nn.Upsample(scale_factor=(2, 2))(conv8)) # 4 x 32 x 400 x 400 -> 4 x 32 x 400 x 400
        merge9 = torch.cat([conv1, up9, inputs], dim=1) # 4 x 70(32+32+6) x 400 x 400
        conv9 = self.conv9(merge9) # 4 x 32 x 400 x 400
        residual = self.residual(conv9) # 4 x 3 x 400 x 400
        return residual


