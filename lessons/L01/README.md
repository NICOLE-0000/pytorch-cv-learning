# L01 CIFAR10 数据加载

## 这一集做了什么
搭建CIFAR10图像预处理流水线，划分训练集、测试集，用Dataset和DataLoader组装批次；取出一个batch打印张量形状、类型、数值范围等信息，验证预处理transform是否生效。

## 关键知识点
1. transforms.Compose 是变换组合器，按书写顺序串行执行预处理操作，前一个变换的输出作为后一个变换的输入。
2. transforms.ToTensor() 同时完成三件事：图像维度 HWC → CHW、像素类型 uint8 转为 float32、像素值 0–255 缩放到 0–1，为后续标准化提供合法输入范围。
3. transforms.Normalize 实现逐通道标准化，公式 $x_{\text{new}} = \frac{x - \text{mean}}{\text{std}}$，把像素分布调整到近似 0 均值，帮助模型更快收敛。
4. Dataset 负责定义单张样本怎么读取；DataLoader 在 Dataset 之上实现分批、打乱、多进程加载，方便在循环里迭代批次用于训练与评估。
5. 张量图像标准格式 `[N,C,H,W]`：N=批次大小，C=通道数，H=图像高度，W=图像宽度，是PyTorch图像任务约定的输入维度。

## 我踩的坑
1. 坑1：现象：运行代码报错 `ModuleNotFoundError: No module named 'torch'`；原因：没有激活项目venv虚拟环境，使用了没有安装PyTorch的全局Python；解决：先激活虚拟环境，再执行脚本。
2. 坑2：现象：代码提示找不到 `transforms.Totensor`；原因：Python标识符区分大小写，正确名称是`ToTensor`；解决：修改为`transforms.ToTensor()`。
3. 坑3：现象：CIFAR10默认源下载速度极低、频繁TLS握手失败；原因：官方源在海外，网络链路不稳定；解决：切换国内镜像源，并且校验md5保证数据集文件完整。

## 运行方式
在**仓库根目录**执行：

```bash
# 激活虚拟环境（根据自己venv实际路径修改）
source ~/code/pytorch-env/bin/activate
python lessons/L01/main.py
```

首次运行默认会尝试自动下载约162MB的CIFAR10数据集。
补充：官方源容易超时、速度慢。可手动从国内镜像下载数据集，放入 `data/` 目录，校验md5：`c58f30108f718f92721af3b95e74349a`，校验通过后程序直接读取本地文件，无需在线下载。`data/` 目录已加入 `.gitignore`，不会提交到仓库。

## 输出
> ⚠️ 说明：训练集开启 `shuffle=True`，每次运行拿到的batch不一样，标签、min/max数值都会变化，属于正常现象。

```text
训练集样本数: 50000
测试集样本数: 10000
images.shape: torch.Size([64, 3, 32, 32])
images.dtype: torch.float32
images.min/max: -2.4291 / 2.7537
labels[:8]: [6, 9, 1, 5, 7, 3, 8, 0]
classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
len(train_loader): 782
```

## 我的理解
transform 不是在数据集初始化时一次性处理全部图片，而是在 `__getitem__` 读取单张样本时实时执行，预处理开销分散在每一轮迭代。
