import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# 1.变换流水线:先转 Tensor，再逐通道标准化
tf = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize(
		mean=(0.4914,0.4822,0.4465),
		std=(0.2470,0.2435,0.2616),
),
]
)

# 2.数据集:train=True 是 5 万张训练集，False 是 1 万张测试集
train_set = datasets.CIFAR10(root="data",train=True,transform=tf,download=True)
test_set = datasets.CIFAR10(root="data",train=False,transform=tf,download=True)

# 3.DataLoader:训练集打乱，测试集不打乱
train_loader = DataLoader(train_set,batch_size=64,shuffle=True,num_workers=0)
test_loader= DataLoader(test_set,batch_size=64,shuffle=False,num_workers=0)

# 4.取一个 batch:iter() 拿到迭代器，next() 取第一批
images,labels = next(iter(train_loader))

# 5.打印，用输出证明链路正确
print(f"训练集样本数：{len(train_set)}")
print(f"测试集样本数：{len(test_set)}")
print(f"images.shape：{images.shape}")
print(f"images.dtype：{images.dtype}")
print(f"images.min/max：{images.min().item():.4f} / {images.max().item():.4f}")
print(f"labels[:8]：{labels[:8].tolist()}")
print(f"classes：{train_set.classes}")
print(f"len(train_loader)：{len(train_loader)}")
