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

    def forward(self,x):
        x = self.conv_unit(x)
        x = x.view(x.size(0),-1)
        logits = self.fc_unit(x)
        return logits

if __name__ == '__main__':
    net = LeNet5()
    print(net)

    x = torch.zeros(2, 3, 32, 32)
    logits = net(x)
    print('logits.shape:', logits.shape)
    print('logits:\n', logits)
    print('softmax 行和:', logits.softmax(dim=1).sum(dim=1))
