import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class StegaData(Dataset):
    def __init__(self, data_path, secret_size=100, size=(400, 400)):
        self.data_path = data_path
        self.secret_size = secret_size
        self.size = size
        self.files_list = glob(os.path.join(self.data_path, '*.jpg'))#获取训练数据集下的全部.jpg格式图片路径
        self.to_tensor = transforms.ToTensor()#转化为tensor

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]#按index提取图片的路径

        img_cover = Image.open(img_cover_path).convert('RGB')#读取图片并转化为RGB三通道格式
        img_cover = ImageOps.fit(img_cover, self.size)#裁剪图片
        img_cover = self.to_tensor(img_cover)
        # img_cover = np.array(img_cover, dtype=np.float32) / 255.

        secret = np.random.binomial(1, 0.5, self.secret_size)#随机采样100维的输入编码
        secret = torch.from_numpy(secret).float()#转化为tensor

        return img_cover, secret

    def __len__(self):
        return len(self.files_list)


if __name__ == '__main__':
    #dataset = StegaData(data_path=r'/mntnfs/sec_data2/tianshuoge/mirflickr', secret_size=100, size=(400, 400))
    #print(len(dataset))
    # img_cover, secret = dataset[10]
    # print(type(img_cover), type(secret))
    # print(img_cover.shape, secret.shape)
    dataset = StegaData(data_path=r'D:\Research\Datasets\mirflickr25k\mirflickr', secret_size=100, size=(400, 400))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    image_input, secret_input = next(iter(dataloader))#返回第一个值
    print(type(image_input), type(secret_input))
    print(image_input.shape, secret_input.shape)
    #print(image_input.max())
