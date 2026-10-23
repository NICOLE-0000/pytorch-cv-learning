# L08 ResBlk 加 stride 与全连接维度不匹配

## 这一集做了什么

给 `ResBlk` 加 `stride` 参数，让它能主动降采样：`conv1` 用 `stride`，shortcut 的 `1×1` 卷积也同步用同一个 `stride`，判断条件从 `ch_in != ch_out` 扩成 `ch_in != ch_out or stride != 1`。四个块全部 `stride=2`，特征图从 `32×32` 一路降到 `2×2`，展平后是 2048 维，送进 `Linear(512, 10)` 时撞上维度不匹配，复现第三个 bug。

## 关键知识点

1. `stride` 是卷积核在输入上的滑动步长：`stride=2` 让 `H`、`W` 各减半，输出尺寸是 `floor((H + 2p - k)/s) + 1`，所以 `32 → 16`。

2. shortcut 必须和主分支**同步降采样**：通道对齐靠 `1×1` 卷积的 `out_channels`，空间对齐靠它的 `stride`。所以 `extra` 里的 `stride` 要跟着块走，固定写 1 就只对齐了通道、对齐不了空间。

3. 判断条件必须扩成 `ch_in != ch_out or stride != 1`：即使通道没变（`blk1` 是 64→64），只要 `stride=2`，主分支已经变成 `16×16`、shortcut 还是 `32×32`，空 `Sequential` 就会让相加崩掉。

4. 展平维度是 `C × H × W = 512 × 2 × 2 = 2048`，而 `Linear(512, 10)` 的 weight 形状是 `[10, 512]`，要求输入最后一维**严格等于** 512。"差不多"没用，必须相等。

5. 报错措辞随版本变：视频里是 `size mismatch, m1: [2048 x 2], m2: [512 x 10]`，本机 2.14.1 是 `mat1 and mat2 shapes cannot be multiplied (2x2048 and 512x10)`。两边的数字含义一样：batch 是 2，特征维是 2048，Linear 要的是 512。

6. 同类的通道错误：视频里的工程 A 把 `blk4` 写成 `ResBlk(512, 1024)`，末端通道 1024 和 `Linear(512, 10)` 永远对不上；工程 B 的正确写法是 `blk4 = ResBlk(512, 512, stride=2)` 配 `Linear(512, 10)`。本仓库的 `blk4` 是 256→512，末端正好 512，所以只暴露了"空间维度太大"这一半问题。

## 我踩的坑

1. 坑1（推理确认、没真报）：以为 `ch_in == ch_out` 时 `extra` 一定可以是空的；现象会是 `out = self.extra(x) + out` 那行报形状不匹配；原因：`stride=2` 让主分支 `[B, 64, 16, 16]`、shortcut 还是 `[B, 64, 32, 32]`；解决：条件加上 `or stride != 1`。

2. 坑2（真实复现）：现象：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x2048 and 512x10)`；原因：`view` 把 `[2, 512, 2, 2]` 展成 `[2, 2048]`，而 `fc` 的 `in_features=512`；解决：L09 用 `F.adaptive_avg_pool2d(x, [1, 1])` 把空间压成 `1×1`，展平就是 512。

3. 坑3（版本差异）：视频报的是 `size mismatch, m1: [2048 x 2], m2: [512 x 10]`，本机报的是 `mat1 and mat2 shapes cannot be multiplied (2x2048 and 512x10)`；原因：PyTorch 版本不同，`F.linear` 内部走的是 `addmm` 还是 `matmul`；解决：记形状含义而不是背报错句子。

## 运行方式

在仓库根目录执行，纯形状自检，秒级。复现 bug 3 时在末尾临时加一行 `net(torch.randn(2, 3, 32, 32))`（已删除，报错见「我踩的坑」）。

```bash
source ~/code/pytorch-env/bin/activate
python lessons/L08/main.py
```

## 输出

```text
conv1 : torch.Size([2, 64, 32, 32])
blk1  : torch.Size([2, 64, 16, 16])
blk2  : torch.Size([2, 128, 8, 8])
blk3  : torch.Size([2, 256, 4, 4])
blk4  : torch.Size([2, 512, 2, 2])
flatten: torch.Size([2, 2048])
fc.weight: torch.Size([10, 512])
```

## 我的理解

`stride` 不只是"加速"，它决定了主分支和 shortcut 能不能对齐；而 2048 和 512 的差距说明，卷积网络的末端尺寸必须靠池化压到和 `fc` 一致，不能指望展平凑巧对上。
