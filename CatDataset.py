
import os
import time

import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# 显示与处理后的图片
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)

class Cat(Dataset):

    def __init__(self, root, resize, mode, filename):

        '''
        初始化
        :param root: 数据集所在路径
        :param resize: resize尺寸
        :param mode: {train, test, val}
        :param filename: csv文件名
        '''

        super(Cat, self).__init__()

        # RGB三色均值
        self.mean = [0.485, 0.456, 0.406]
        # RGB三色方差
        self.std = [0.229, 0.224, 0.225]

        self.root = root
        self.resize = resize

        # 加载csv文件数据
        self.images, self.labels = self.load_csv(filename)

        # 根据不同需求，裁剪数据信息
        if mode == 'train':
            # 用于训练
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            # 用于验证
            self.images = self.images[int(0.6 * len(self.images)): int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)): int(0.8 * len(self.labels))]
        elif mode == 'test':
            # 用于测试
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
        else:
            # 输入有误
            raise Exception('wrong mode')

    def load_csv(self, filename):

        '''
        加载csv文件
        :param filename: 文件名
        :return images: 图片路径
        :return labels: 标签
        '''

        images, labels = [], []

        # 打开文件
        with open(os.path.join(self.root, filename), 'r') as f:

            # 读取csv文件
            records = f.readlines()
            # 随机打散
            random.shuffle(records)

            for line in records:

                img, label = line.split('\t')
                images.append(os.path.join(self.root, img.strip()))
                labels.append(int(label.strip()))

        assert len(images) == len(labels)

        return images, labels

    def denormalize(self, x_encode):

        '''
        反归一化处理
        :param x_encode: 归一化后的图片表示
        :return img: 反归一化后的图片
        '''

        # 将mean、std由1维变为3维
        mean = torch.tensor(self.mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(self.std).unsqueeze(1).unsqueeze(1)

        # 反归一化处理
        img = x_encode * std + mean

        return img

    def __len__(self):

        '''
        返回数据集长度
        :return length: 数据集长度
        '''

        return len(self.images)

    def __getitem__(self, idx):

        '''
        重构可迭代数据，返回图片及标签
        :param idx: 索引位置
        :return img: 图片的tensor
        :return label: 图片的标签
        '''

        img, label = self.images[idx], self.labels[idx]

        # 显示原始图像
        # images = Image.open(img)
        # images.show()

        transform = transforms.Compose([
            lambda x: Image.open(img).convert('RGB'),
            # 水平随机翻转
            transforms.RandomHorizontalFlip(0.5),
            # 竖直随机翻转
            transforms.RandomVerticalFlip(0.5),
            # 调整到大小的1.25倍
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            # 随机旋转[-15, 15]度
            transforms.RandomRotation(15),
            # 中心裁剪到resize
            transforms.CenterCrop(self.resize),
            # 转换成tensor
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        # 转换图片
        img = transform(img)

        # 输出预处理后的图片

        # plt.figure()
        # imshow(img)
        # plt.show()
        #
        # plt.imshow(transforms.functional.to_pil_image(img))
        # plt.axis("off")
        # plt.show()
        #
        # time.sleep(100)

        # 转换标签
        label = torch.tensor(label)

        return img, label


def main():

    import time
    import visdom

    # 可视化工具
    viz = visdom.Visdom()

    # 数据集路径
    data_path = os.path.abspath('data')
    # 文件名
    filename = 'train_list.txt'
    # 获取数据集
    dataset = Cat(data_path, 224, 'train', filename)
    # 获取单条数据
    image, label = next(iter(dataset))
    # 在可视化工具上进行展示
    viz.image(dataset.denormalize(image), win='image', opts=dict(title='image'))
    viz.text(str(label.numpy()), win='label', opts=dict(title='label'))

    # 批量加载数据
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # 遍历批处理数据
    for x, y in loader:

        # 展示
        viz.images(dataset.denormalize(x), nrow=8, win='image', opts=dict(title='image'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='label'))
        # 停留10秒
        time.sleep(10)


if __name__ == '__main__':
    main()
