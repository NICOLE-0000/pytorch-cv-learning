# L06 ResBlk 残差块

## 这一集做了什么

写出 ResBlk：主分支是两层 `3×3` 卷积 + BatchNorm，shortcut 分支用 `self.extra` 一个字段实现两种状态（通道相同→空 `Sequential` 恒等，通道不同→`1×1` 卷积 + BN 投影），再用假 batch 自检 `64→128` 和 `64→64` 两种形状。

## 关键知识点

1. 残差块学的是**残差**而不是完整映射：`out = F(x) + x`，网络只需要拟合 `F(x)`。要做恒等映射时，普通堆叠层必须把权重精确调成"还原输入"，而残差块只要把权重压到 0，输出天然等于 `x`，所以深层网络更容易训。shortcut 还给了梯度一条不经过卷积的回传路径。

2. `self.extra` 的两态技巧：`nn.Sequential()` 不塞任何层时，被调用就直接返回输入（它内部只是循环遍历子模块，零个子模块等于什么都不做），所以能当恒等映射用；通道数不同时才换成 `Sequential(Conv2d(k=1), BatchNorm2d)`。`1×1` 卷积只改通道数，不动 `H, W`。

3. `x + out` 要求两个张量的 `(b, c, h, w)` 完全一致。`kernel_size=3, padding=1` 让卷积保持 `H, W` 不变（`(32 - 3 + 2)/1 + 1 = 32`），所以相加时只有通道数需要靠 `extra` 对齐。

4. 顺序是 `conv → bn → relu`，最后一个 `relu` 作用在**相加之后**。残差分支 `F(x)` 自己必须保留负值，提前 ReLU 会把负的部分截成 0，函数就变了。ResNet v2 的 pre-activation 变体是 `BN → ReLU → Conv` 的另一种排列，不等于"先 ReLU 再相加"。

5. BatchNorm 是这一集新出现的层：按通道在 batch 维上标准化，再乘可学习的 γ、加 β。它意味着 L05 学的 `net.eval()` 从此是必需项——评估时要冻结 running mean/var。

## 我踩的坑

1. 坑1（推理出来、没真报）：删掉 `if ch_in != ch_out` 那段，错误不在构造时暴露，而是在 `out = x + out` 那一行报 `RuntimeError`，因为 `x` 是 64 通道、`out` 已经是 128 通道。张量加法只认形状，不认语义。

2. 坑2（推理出来、没真报）：把 `padding=1` 改成 `padding=0`，主分支 `32 → 30 → 28`，shortcut 还是 `32`，最后在相加那一步挂掉：`[2, 128, 32, 32]` 加 `[2, 128, 28, 28]`。`padding` 不是可选项，是用来把尺寸对齐的。

3. 坑3：答题时把"先对卷积分支 ReLU 再相加"和 pre-activation ResNet v2 混成一回事。区别在于 v2 是 `BN → ReLU → Conv` 的块内重排，不是把 ReLU 挪到相加之前；标准块必须是"相加后再 ReLU"。

## 运行方式

在仓库根目录执行，只做形状自检，不训练，CPU 上秒级完成。

```bash
source ~/code/pytorch-env/bin/activate
python lessons/L06/main.py
```

## 输出

```text
in : torch.Size([2, 64, 32, 32])
out: torch.Size([2, 128, 32, 32])
same-ch in : torch.Size([2, 64, 32, 32])
same-ch out: torch.Size([2, 64, 32, 32])
extra(64->128): Sequential(
  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, bias=True, track_running_stats=True)
)
extra(64->64) : Sequential()
```

## 我的理解

残差块把"学一个完整变换"换成"在输入上加一点修正"；`extra` 的两种状态说明，同一个类既能处理通道不变的拼接，也能处理通道变化的投影，差别只在 shortcut 要不要先做一次 `1×1` 卷积。
