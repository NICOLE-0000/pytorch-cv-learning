# L09 adaptive_avg_pool2d 修维度并训练 ResNet18

## 这一集做了什么

用 `F.adaptive_avg_pool2d(x, [1, 1])` 把 `[b, 512, 2, 2]` 压成 `[b, 512, 1, 1]`，再用 `view(x.size(0), -1)` 展成 `[b, 512]`，接上 `Linear(512, 10)`；然后把 L04/L05 的训练循环和评估函数搬到 ResNet18 上，在 CIFAR10 上跑 1 个 epoch，得到 `avg_loss=1.2119`、`test_acc=0.6258`。

## 关键知识点

1. 全局平均池化（GAP）把每个通道的 `H × W` 个值平均成一个数：`[b, 512, 2, 2]` → `[b, 512, 1, 1]` → `view` → `[b, 512]`。分类头的输入维度从此只跟通道数有关，跟特征图分辨率无关。

2. `adaptive_avg_pool2d` 指定的是**输出尺寸**，核大小和步长由它自己算；`avg_pool2d(x, 2)` 指定的是核大小，输入变 `4×4` 就输出 `2×2`，还得再池化一次。ResNet 系列用 adaptive 版本，是为了换输入分辨率时不用改网络。

3. 不修池化、直接把 `fc` 改成 `Linear(2048, 10)` 也能跑通，但会把 `2×2` 的空间位置写死（输入尺寸一变就崩）、参数多 15360 个（20490 对 5130），而且丢掉 GAP 的聚合与正则作用。

4. `net.eval()` 在这一集从"可选"变成"必需"：ResNet18 里全是 BatchNorm，`train()` 用当前 batch 的均值和方差并更新滑动统计量，`eval()` 用训练时积累的滑动统计量。评估时忘了切，得到的准确率不可信。

5. 参数量 4,910,282，对比 LeNet5 约 6.2 万，相差约 80 倍。参数量不是准确率的直接保证，但能说明两者的容量差距。

## 我踩的坑

1. 坑1（真实复现后修好）：L08 的 `mat1 and mat2 shapes cannot be multiplied (2x2048 and 512x10)`，原因是展平维度和 `fc` 的 `in_features` 对不上；解决：在 `blk4` 之后插 `F.adaptive_avg_pool2d(x, [1, 1])`，而不是去改 `Linear` 的输入维度。

2. 坑2（真实结果与参考值不符）：视频里工程 B 的 `acc` 是 `0.688–0.708`，本机实测 `0.6258`；原因有四个，都还没验证哪个影响最大：① 本仓库每个 stage 只有 1 个 `ResBlk`，标准 ResNet18 是 2 个（参数量 4.9M 对约 11M）；② `blk1` 就用了 `stride=2`，特征图最后到 `2×2`，比常见 CIFAR 版 ResNet 的 `4×4` 更早丢空间信息；③ 只训了 1 个 epoch；④ 没固定随机种子。解决：不修改数字，先如实记录；`final-project` 里做 LeNet5 与 ResNet18 对照时再统一条件比较。

3. 坑3（认知）：以为"参数更多 = 准确率更高"。LeNet5 的 `test_acc` 是 0.4622，ResNet18 是 0.6258，确实更高，但这是同类任务、同样 1 个 epoch 下的结果，不能推成"参数量决定准确率"。

## 运行方式

在仓库根目录执行；`data/` 复用 L01 已下载的 CIFAR10。

```bash
source ~/code/pytorch-env/bin/activate
python lessons/L09/main.py
```

CPU-only，ResNet18 比 LeNet5 慢得多，1 个 epoch 需要较长时间，中途不要中断。

## 输出

```text
device: cpu
params: 4910282
step 100/782  loss=1.2583
step 200/782  loss=1.3704
step 300/782  loss=1.2816
step 400/782  loss=1.1386
step 500/782  loss=1.0488
step 600/782  loss=0.9626
step 700/782  loss=0.9467
epoch 1: avg_loss=1.2119  test_acc=0.6258
```

## 我的理解

`adaptive_avg_pool2d` 解决的不是"数字凑巧对上"，而是让分类头不再依赖特征图尺寸；ResNet18 用 1 个 epoch 就到了 0.6258，说明残差结构加 BN 确实比 LeNet5 更能学，但离参考值还有差距，原因要留到后面的对照实验里查。
