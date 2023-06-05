
import torch
from torch import nn
from torch.nn import functional as F


class Flatten(nn.Module):

    '''
    将图片拉平，keras中有Flatten，torch中没有
    '''

    def __init__(self):

        super(Flatten, self).__init__()

    def forward(self, x):

        shape = torch.prod(torch.tensor(x.shape[1:])).item()

        return x.view(-1, shape)


class ResBlk(nn.Module):

    '''
    ResNetBlock，用于创建ResNet
    '''

    def __init__(self, in_channel, out_channel, stride=1):

        '''
        初始化
        :param in_channel: 输入channel
        :param out_channel: 输出channel
        :param stride: 卷积步长
        '''

        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.extra = nn.Sequential()

        if in_channel != out_channel:

            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):

        '''
        前向传播
        :param x: 输入x
        :return output: 计算后输出
        '''

        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = self.extra(x) + output
        output = F.relu(output)

        return output


class ResNet18(nn.Module):

    '''
    ResNet18
    '''

    def __init__(self, num_class):

        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(8)
        )

        self.blk1 = ResBlk(8, 16, 2)
        self.blk2 = ResBlk(16, 32, 2)
        self.blk3 = ResBlk(32, 64, 2)
        self.blk4 = ResBlk(64, 128, 2)

        self.outlayer = nn.Linear(128*5*5, num_class)

    def forward(self, x):

        '''
        前项传播
        :param x: 输入
        :return x: 计算后的输出
        '''

        x = F.relu(self.conv1(x))
        x = self.blk4(self.blk3(self.blk2(self.blk1(x))))
        x = F.adaptive_avg_pool2d(x, [5, 5])
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)

        return x


def main():

    # 创建resnet模型
    model = ResNet18(num_class=5)
    # 创建测试数据，模拟4张3通道224像素图片
    test_data = torch.randn(4, 3, 224, 224)
    # 经过model得到输出
    out = model(test_data)

    print(out.shape)


if __name__ == '__main__':
    main()
