import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        #卷积层+BN（双层卷积）
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

if __name__ == '__main__':
    blk = ResBlk(64, 128)
    x = torch.randn(2, 64, 32, 32)
    out = blk(x)
    print('in :', x.shape)
    print('out:', out.shape)

    blk_same = ResBlk(64, 64)
    x2 = torch.randn(2, 64, 32, 32)
    print('same-ch in :', x2.shape)
    print('same-ch out:', blk_same(x2).shape)
    print('extra(64->128):', blk.extra)
    print('extra(64->64) :', blk_same.extra)
