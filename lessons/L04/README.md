# L04 LeNet5 训练循环

## 这一集做了什么
先故意删掉 `forward` 里的 `return logits`，用真实数据跑出一个 `NoneType` 报错，看清缺失返回值时错误在哪一层暴露；恢复后写出完整训练循环（Adam + CrossEntropyLoss），在 CIFAR10 训练集上跑完 1 个 epoch。

## 关键知识点

1. 训练循环固定五步：前向 `logits = net(images)` → 算损失 `loss = criterion(logits, labels)` → 清梯度 `optimizer.zero_grad()` → 反向 `loss.backward()` → 更新 `optimizer.step()`。顺序不能乱。

2. **PyTorch 的梯度默认是累加的**，`backward()` 只是把新梯度加到 `.grad` 上，不清空。所以 `zero_grad()` 必须写在 `backward()` 之前，否则每步都在累加上一步的梯度，等效于偷偷放大学习率。

3. `nn.CrossEntropyLoss` 的输入是**未归一化的 logits** 和**类别索引 labels**（不是 one-hot），它内部完成 `log_softmax` + `NLLLoss`。`forward` 里再手动 softmax 就是做了两次，梯度是错的。

4. `Adam(net.parameters(), lr=1e-3)` 用一阶矩和二阶矩自适应地调整每个参数的学习率，比裸 SGD 对学习率取值不敏感，是这套实战代码的默认选择。

5. 判断模型有没有学到东西要有基线：10 类均匀随机猜的交叉熵是 `ln(10) ≈ 2.3026`。本轮 `avg_loss = 1.6627`，低于基线，说明参数确实在往正确方向走。

6. 单个 batch 的 loss 抖动是正常的——每个 batch 是不同的 64 张图，loss 高低取决于这一批的难易。只有整轮平均 `avg_loss` 才能反映趋势。

## 我踩的坑

1. 坑1：现象：`TypeError: cross_entropy_loss(): argument 'input' (position 1) must be Tensor, not NoneType`；原因：`forward` 里漏写 `return logits`，Python 函数默认返回 `None`，而 `net(images)` 返回 `None` 本身不会报错，直到 `criterion` 拿到 `None` 才崩；解决：补上 `return logits`。
2. 坑2：现象：同样这个 bug，视频里的报错却是 `AttributeError: 'NoneType' object has no attribute 'shape'`；原因：PyTorch 版本差异——老版本直接去读 `input.shape`，2.14.1 在碰 `.shape` 之前先做参数类型校验；解决：本质是同一个 bug，报错文本随版本变化，定位时要看调用栈而不是只记这句话。
3. 坑3：现象：误以为 `step` 那几行 loss 忽高忽低说明训练不稳定；原因：把单 batch loss 当成整体趋势；解决：看 `avg_loss`，781 个 batch 的平均才代表这一轮的水平。

## 运行方式
在**仓库根目录**执行；需要先有 `data/` 下的 CIFAR10（L01 已下载，`download=True` 会直接复用）。

```bash
source ~/code/pytorch-env/bin/activate
python lessons/L04/main.py
```

本机无 NVIDIA GPU，`device` 会是 `cpu`，1 个 epoch 需要几分钟。

## 输出

```text
device: cpu
step 100/782  loss=1.9605
step 200/782  loss=1.8550
step 300/782  loss=1.7482
step 400/782  loss=1.7541
step 500/782  loss=1.4974
step 600/782  loss=1.8109
step 700/782  loss=1.7206
epoch 1: avg_loss=1.6627
```

## 我的理解
训练循环就是把"前向算得分、交叉熵算差距、反向算梯度、优化器改参数"这四件事重复 782 次；`zero_grad` 不是可选项，因为 PyTorch 的梯度会累加而不是覆盖。
