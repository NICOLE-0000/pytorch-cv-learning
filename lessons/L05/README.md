# L05 LeNet5 测试段与准确率

## 这一集做了什么

在 L04 的训练循环后面补上 `evaluate()`：训练完 1 个 epoch 后，在 CIFAR10 测试集（10000 张没参与训练的图）上算准确率，第一次拿到 `test_acc=0.4622` 这个能量化"模型到底行不行"的指标。

## 关键知识点

1. `net.train()` / `net.eval()` 只是**模式开关**，不改变网络结构。LeNet5 里没有 Dropout 和 BatchNorm，此刻把 `net.eval()` 删掉结果一模一样；但后面的 ResNet18 一旦含 BatchNorm，评估时必须切回 `eval()`，否则测试数据的统计量会污染 BN 的 running mean/var。

2. `@torch.no_grad()` 是装饰器，作用在整个 `evaluate` 函数上。它关闭自动求导，评估时不再构建计算图，中间激活不必留在图里，省内存也更快。这台机器是 CPU-only，省的是内存而不是显存。

3. `logits.argmax(dim=1)`：`logits` 形状 `[b, 10]`，第 1 维才是 10 个类别，所以 `dim=1` 每张图返回一个 0–9 的类别索引，结果形状 `[b]`，才能和 `labels` 逐元素比较。

4. 准确率的分子分母都要按**样本数**累加：`(pred == labels).sum().item()` 累加答对数，`labels.size(0)` 累加总数，最后 `correct / total` 才是整个测试集的比例。只拿某一个 batch 的准确率当结论会被小样本误导。

5. 两个随机基线：loss 的基线是 `ln(10) ≈ 2.3026`，准确率的基线是 0.10。本轮 `avg_loss=1.6802`、`test_acc=0.4622` 都明显好于基线，但这只是 1 个 epoch 的 LeNet5，不是这套网络的最终水平。

## 我踩的坑

1. 坑1：现象：想过不写 `net.eval()`，反正现在删掉也不报错、结果也不变；原因：把 `eval()` 当成"评估时才用的功能"，其实它是给 Dropout/BatchNorm 这类"训练和测试行为不同"的层切开关，现在没感觉只是因为 LeNet5 没有这种层；解决：保留这个习惯，L06 上 ResNet18 时它是必需项。

2. 坑2：现象：把 `argmax(dim=1)` 改成 `argmax(dim=0)`，用 `[64, 10]` 的假 logits 实测，`pred` 形状变成 `[10]`，再和 `labels`（`[64]`）比较直接报 `RuntimeError: The size of tensor a (10) must match the size of tensor b (64) at non-singleton dimension 0`；原因：`dim=0` 取的是"每个类别里得分最高的那张图在 batch 里的下标"，方向完全反了；解决：类别维是第 1 维，永远用 `dim=1`。

3. 坑3：现象：本轮 `avg_loss=1.6802`，和 L04 的 `1.6627` 对不上；原因：没有设 `torch.manual_seed`，参数初始化和 `shuffle=True` 的打乱顺序每次都不一样；解决：做对比实验（比如 Adam vs SGD）时必须固定随机种子，否则分不清差异是算法带来的还是随机性带来的。

## 运行方式

在**仓库根目录**执行；`data/` 复用 L01 已下载的 CIFAR10。

```bash
source ~/code/pytorch-env/bin/activate
python lessons/L05/main.py
```

本机无 NVIDIA GPU，`device` 是 `cpu`；1 个 epoch 训练加 157 个测试 batch 一共几分钟。

## 输出

```text
device: cpu
step 100/782  loss=2.1593
step 200/782  loss=1.7416
step 300/782  loss=1.7301
step 400/782  loss=1.6316
step 500/782  loss=1.4809
step 600/782  loss=1.4200
step 700/782  loss=1.4433
epoch 1: avg_loss=1.6802  test_acc=0.4622
```

## 我的理解

训练是"看着答案改参数"，评估是"把答案遮住再考一次"；`test_acc=0.4622` 说明 1 个 epoch 的 LeNet5 已经能答对将近一半，但离这套网络的上限还远。
