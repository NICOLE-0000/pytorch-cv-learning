import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_unit(x)
        return logits

def get_dataloaders(batch_size=64):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616)),
    ])
    train_set = datasets.CIFAR10(root='data', train=True, transform=tf, download=True)
    test_set = datasets.CIFAR10(root='data', train=False, transform=tf, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def train_one_epoch(net, train_loader, criterion, optimizer, device):
    net.train()
    total_loss = 0.0
    for step, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)

        logits = net(images) #前向计算
        loss = criterion(logits, labels) #计算损失

        optimizer.zero_grad() #梯度清零
        loss.backward() #反向传播
        optimizer.step() #优化器根据梯度更新权重参数

        total_loss += loss.item()
        if step % 100 == 0:
            print(f'step {step}/{len(train_loader)}  loss={loss.item():.4f}')

    return total_loss / len(train_loader)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_loader, test_loader = get_dataloaders(batch_size=64)

    net = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    avg_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
    print(f'epoch 1: avg_loss={avg_loss:.4f}')
