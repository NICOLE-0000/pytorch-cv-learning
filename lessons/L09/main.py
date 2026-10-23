import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_in != ch_out or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.blk1 = ResBlk(64, 64, stride=2)
        self.blk2 = ResBlk(64, 128, stride=2)
        self.blk3 = ResBlk(128, 256, stride=2)
        self.blk4 = ResBlk(256, 512, stride=2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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

        logits = net(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f'step {step}/{len(train_loader)}  loss={loss.item():.4f}')

    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate(net, test_loader, device):
    net.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_loader, test_loader = get_dataloaders(batch_size=64)

    net = ResNet18().to(device)
    print('params:', sum(p.numel() for p in net.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    avg_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
    acc = evaluate(net, test_loader, device)
    print(f'epoch 1: avg_loss={avg_loss:.4f}  test_acc={acc:.4f}')
