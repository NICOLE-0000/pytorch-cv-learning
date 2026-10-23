import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # 提取图像特征：卷积 + 池化
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # 全连接
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

        # 形状自检：假 batch 只走卷积部分，确认 flatten 前是 [2, 16, 5, 5]
        tmp = torch.zeros(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print('conv_unit 输出形状:', out.shape)


if __name__ == '__main__':
    net = LeNet5()
    print(net)

    print('conv1 weight:', net.conv_unit[0].weight.shape)   # [6, 3, 5, 5]
    print('conv2 weight:', net.conv_unit[2].weight.shape)   # [16, 6, 5, 5]
    print('fc1   weight:', net.fc_unit[0].weight.shape)     # [120, 400]
