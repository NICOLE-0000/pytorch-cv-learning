# L02 LeNet5 网络结构

## 这一集做了什么
用 `nn.Module` 搭建 LeNet5 的骨架，只写 `__init__`：`conv_unit` 负责卷积特征提取，`fc_unit` 负责分类，最后用一个假 batch 做形状自检，`forward` 留到之后。

## 关键知识点

1. 卷积输出尺寸推导是这一集的核心。公式：`H_out = (H_in + 2*padding - kernel_size) / stride + 1`。CIFAR10 输入是 32×32×3，整条链如下：

| 层 | 尺寸计算 | 输出 H×W | 输出通道 |
|---|---|---|---|
| 输入 | — | 32×32 | 3 |
| Conv2d(3, 6, 5) | (32 − 5) / 1 + 1 = 28 | 28×28 | 6 |
| AvgPool2d(2) | 28 / 2 = 14 | 14×14 | 6 |
| Conv2d(6, 16, 5) | (14 − 5) / 1 + 1 = 10 | 10×10 | 16 |
| AvgPool2d(2) | 10 / 2 = 5 | 5×5 | 16 |
| flatten | 16 × 5 × 5 | 400 | — |

所以 `nn.Linear` 的 `in_features` 必须是 400。

2. `nn.Module` 通过重写 `__setattr__` 维护参数注册表。`super().__init__()` 创建 `_modules` / `_parameters`，之后执行 `self.conv_unit = ...` 才会被登记进注册表；漏掉这一步，要么直接抛 `AttributeError`，要么参数静默丢失。

3. `nn.Sequential` 是按顺序执行的容器，它本身也是 `Module`，所以赋给 `self.xxx` 就自动完成注册。层支持下标访问，`net.conv_unit[0]` 就是第一个 `Conv2d`。

4. `Conv2d` 的权重形状是 `[out_channels, in_channels, kH, kW]`，实测 `[6, 3, 5, 5]` 和 `[16, 6, 5, 5]`；`Linear` 的权重形状是 `[out_features, in_features]`，实测 `[120, 400]`。权重形状可以反过来验证层参数有没有写错。

5. `AvgPool2d(kernel_size, stride=None)` 在不指定 `stride` 时默认等于 `kernel_size`，所以 `AvgPool2d(2)` 的步长就是 2，特征图尺寸正好减半。

## 我踩的坑

1. 坑1：现象：`SyntaxError: invalid syntax`，指向 `net LeNet5()`；原因：手敲时漏了赋值号 `=`；解决：改成 `net = LeNet5()`。
2. 坑2：现象：`IndentationError: unindent does not match any outer indentation level`；原因：在 nano 里混用了 Tab 和 4 个空格，Python 把 Tab 按 8 列计算，回退缩进时找不到对应的层级；解决：统一用 4 个空格，不按 Tab 键，用 `cat -A`（Tab 显示为 `^I`）或 `python -m tabnanny` 自查。
3. 坑3：现象：构造器完全没有执行；原因：`__init__` 写成了 `_init_`，前后各少一个下划线；解决：改成 `__init__`，同理 `__name__`、`__main__` 也都是双下划线。
4. 坑4：现象：卷积单元实际没有生效；原因：`self.conv_unit = nn.Sequential` 之后换行才写 `(`，Python 把它们当成两条独立语句，括号里那一串层组成元组后被直接丢弃；解决：把左括号和 `nn.Sequential` 写在同一行。
5. 坑5：现象：特征图通道数对不上；原因：`nn.Conv2d(6, 6, ...)` 的第二个 `6` 应该是输出通道 16，写错后 16×5×5=400 这个数就是假的；解决：改成 `nn.Conv2d(6, 16, ...)`，再用权重形状 `[16, 6, 5, 5]` 验证。
6. 坑6：现象：`AttributeError: cannot assign module before Module.__init__() call`；原因：没调用 `super().__init__()` 就给 `self.conv_unit` 赋值，参数注册表还不存在；解决：构造器第一行必须调用。更隐蔽的版本是用普通 list 装层，不报错但参数量为 0，必须用 `nn.ModuleList`。

## 运行方式
在**仓库根目录**执行：

```bash
# 激活虚拟环境
source ~/code/pytorch-env/bin/activate
python lessons/L02/main.py
```

本集不读数据、不训练，秒级输出。

## 输出

```text
conv_unit 输出形状: torch.Size([2, 16, 5, 5])
LeNet5(
  (conv_unit): Sequential(
    (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
    (1): AvgPool2d(kernel_size=2, stride=2, padding=0)
    (2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    (3): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (fc_unit): Sequential(
    (0): Linear(in_features=400, out_features=120, bias=True)
    (1): ReLU()
    (2): Linear(in_features=120, out_features=84, bias=True)
    (3): ReLU()
    (4): Linear(in_features=84, out_features=10, bias=True)
  )
)
conv1 weight: torch.Size([6, 3, 5, 5])
conv2 weight: torch.Size([16, 6, 5, 5])
fc1   weight: torch.Size([120, 400])
```

## 我的理解
LeNet5 的结构不是随手堆层：先把每一层的输出尺寸算清楚，再让 `nn.Linear` 的输入长度去匹配 flatten 之后的 400。尺寸链算对是网络跑起来的前提。
