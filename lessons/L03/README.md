# L03 LeNet5 前向传播与 logits

## 这一集做了什么
给 LeNet5 补上 `forward`，打通 `[b, 3, 32, 32] → [b, 16, 5, 5] → [b, 400] → [b, 10]` 这条链路；用假 batch 验证输出维度，再用 softmax 行和验证 logits 和概率不是一回事。

## 关键知识点

1. `forward` 定义数据怎么流过网络。调用时写 `net(x)`，不写 `net.forward(x)`——`nn.Module.__call__` 会自动转发到 `forward`，并在前后挂上 hook 机制，直接调 `forward` 会绕过这些机制。

2. 形状变换的关键是 `x.view(x.size(0), -1)`：第一维 `x.size(0)` 是 batch，必须原样保留；`-1` 让 PyTorch 自动推断剩下的维，这里等于 16×5×5=400。写成 `x.view(-1, 400)` 就把 batch 和通道混着算了。

3. `view`、`reshape`、`flatten` 三者的区别：`view` 要求内存连续，不连续直接报错；`reshape` 不报错，但会在需要时隐式拷贝；`torch.flatten(x, 1)` 语义最明确（从第 1 维开始拍平），也最不容易写错。本层前接 AvgPool2d，输出连续，所以 `view` 是安全的。

4. 网络输出 `[b, 10]` 是 **logits**，不是概率：可以是负数，也不要求和为 1。`softmax(dim=1)` 沿**类别维**归一化后行和才为 1；写成 `dim=0` 就是沿 batch 归一化，语义完全变了。

5. `forward` 返回 logits 而不是概率，是因为 `nn.CrossEntropyLoss` 内部自带 `log_softmax` + `NLLLoss`。提前 softmax 等于做两次，梯度错误；而且 log 空间的 log-sum-exp 比"先 softmax 再取 log"数值更稳。

## 我踩的坑

1. 坑1：现象：以为 `reshape` 有潜在风险、`view` 更安全；原因：把两者的失败模式记反了——真正会报错的是 `view`（要求连续内存），`reshape` 不报错但会隐式拷贝；解决：本层输入连续用 `view`，写法与视频一致，语义更清楚时用 `flatten`。
2. 坑2：现象：回答"为什么返回 logits"时说成"因为没有经过归一化"；原因：把 `transforms.Normalize`（图像像素标准化）和 softmax（把得分归一化成概率）混为一谈；解决：两个"归一化"不是一回事，前者在数据加载阶段，后者在损失函数内部。
3. 坑3：现象：拿全零张量测试时两行 logits 完全相同；原因：输入全零时卷积输出只剩 bias，两条样本输入等价；解决：这是测试数据的特性，不是 bug，换成真实图片就不会相同。

## 运行方式
在**仓库根目录**执行：

```bash
# 激活虚拟环境
source ~/code/pytorch-env/bin/activate
python lessons/L03/main.py
```

本集不做训练，只做一次前向，秒级输出。

## 输出

```text
logits.shape: torch.Size([2, 10])
logits:
 tensor([[ 0.1260,  0.0520, -0.0790,  0.0810, -0.0152, -0.0394, -0.0711,  0.1148,
         -0.0480, -0.0362],
        [ 0.1260,  0.0520, -0.0790,  0.0810, -0.0152, -0.0394, -0.0711,  0.1148,
         -0.0480, -0.0362]], grad_fn=<AddmmBackward0>)
softmax 行和: tensor([1.0000, 1.0000], grad_fn=<SumBackward1>)
```

`grad_fn=<AddmmBackward0>` 说明计算图正在跟踪梯度，L04 的 `loss.backward()` 依赖它。

## 我的理解
`forward` 的实质是把二维特征图拍平成一个 400 维向量，再映射到 10 类得分；输出保持 logits 而不是概率，是因为交叉熵损失需要原始得分自己完成 log_softmax，提前归一化反而破坏梯度。
