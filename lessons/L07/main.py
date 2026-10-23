import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

#残差块提取图像特征
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.blk1 = ResBlk(64, 64)
        self.blk2 = ResBlk(64, 128)
        self.blk3 = ResBlk(128, 256)
        self.blk4 = ResBlk(256, 512)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = ResNet18()
    x = torch.randn(2, 3, 32, 32)

    x = F.relu(net.conv1(x))
    print('conv1 :', x.shape)
    for name in ['blk1', 'blk2', 'blk3', 'blk4']:
        x = getattr(net, name)(x)
        print(f'{name}  :', x.shape)

    flat = x.view(x.size(0), -1)
    print('flatten:', flat.shape)
    print('fc.weight:', net.fc.weight.shape)

