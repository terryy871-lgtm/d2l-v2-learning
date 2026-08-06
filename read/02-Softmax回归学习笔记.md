# 02 Softmax 回归学习笔记

这份笔记是结合你的文件 [softmax.ipynb](</D:/Learn/d2l-test/Linear Neural Networks/softmax.ipynb>) 写的，目标是帮你把这章真正看懂，而不只是把代码敲出来。

你现在学到 Softmax 回归，其实说明你已经从：

- 预测一个连续数值的“回归问题”

走到了：

- 预测属于哪一类的“分类问题”

所以这一章最重要的不是代码变复杂了，而是：

```text
任务类型变了：从回归变成了多分类
```

## 1. 先用一句人话理解 Softmax 回归

线性回归学的是：

```text
输入 -> 预测一个数值
```

比如：

- 预测房价
- 预测温度
- 预测销量

Softmax 回归学的是：

```text
输入 -> 判断它属于哪个类别
```

比如：

- 这张图片是 T-shirt 还是 sneaker
- 这封邮件是不是垃圾邮件
- 这个数字是 0 到 9 里的哪一个

你这份 notebook 用的是 `Fashion-MNIST` 数据集，所以任务是：

```text
输入一张衣服图片 -> 预测它属于 10 个类别中的哪一个
```

## 2. 为什么线性回归不够用了

在线性回归里，输出通常是一个实数：

```text
y_hat = Xw + b
```

这适合预测数值。

但分类问题不一样。

比如衣服分类，你不是要预测一个连续数值，而是要回答：

- 第 0 类的可能性多大？
- 第 1 类的可能性多大？
- ...
- 第 9 类的可能性多大？

所以模型输出不能再只是一个数，而应该是：

```text
10 个类别各自的得分
```

然后再把这些得分变成概率。

这就是 Softmax 回归出现的原因。

## 3. Softmax 回归的整体结构

你可以先把 Softmax 回归看成两步：

### 第一步：线性变换，得到每个类别的分数

```text
o = XW + b
```

这里：

- `X`：输入
- `W`：权重矩阵
- `b`：偏置
- `o`：每个类别对应的分数，通常叫 logits

### 第二步：把分数变成概率

```text
y_hat = softmax(o)
```

softmax 的作用是：

- 把每个类别的分数变成非负数
- 并且让一张图片对应的所有类别概率加起来等于 1

这样就能把模型输出解释成：

```text
这张图片属于每个类别的概率分布
```

## 4. 先看你这份 notebook 在做什么

你的 notebook 主线非常完整，主要包括这几步：

1. 读取 Fashion-MNIST 数据
2. 初始化参数 `W` 和 `b`
3. 写 `softmax`
4. 写分类模型 `net`
5. 写交叉熵损失 `cross_entropy`
6. 写准确率函数 `accuracy`
7. 写评估函数 `evaluate_accuracy`
8. 写训练一个 epoch 的函数
9. 写总训练函数
10. 用 SGD 训练模型

如果用一句话概括这份 notebook：

```text
它是在手写一个 10 分类的线性分类器
```

## 5. 按你的代码顺序详细讲

## 5.1 读取数据

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
```

这一步是在读数据集。

`Fashion-MNIST` 是一个服装图片分类数据集，特点是：

- 图片是灰度图
- 每张图片大小是 `28 x 28`
- 一共有 `10` 个类别

例如这 10 类通常包括：

- T-shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

### `batch_size = 256` 是什么意思

意思是：

- 每次训练时，不是只拿 1 张图片
- 也不是一次拿完整个数据集
- 而是每次拿 `256` 张图片一起训练

所以一个 batch 大致可以理解成：

- `X`：256 张图片
- `y`：256 个标签

## 5.2 初始化模型参数

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)
```

这一段特别关键，因为它告诉你模型结构是什么。

### 为什么 `num_inputs = 784`

因为每张图是 `28 x 28`。

把图片拉平成一维向量以后：

```text
28 * 28 = 784
```

所以每张图片可以表示成一个 784 维向量。

### 为什么 `num_outputs = 10`

因为这个分类任务一共有 10 个类别。

模型最后要输出：

```text
这张图属于 10 个类别中每一类的分数/概率
```

### 为什么 `W` 的形状是 `(784, 10)`

这是 Softmax 回归里最需要想明白的形状关系之一。

如果一个 batch 的输入经过拉平以后是：

```text
X.shape = (batch_size, 784)
```

为了得到 10 个类别的输出：

```text
XW 的形状必须是 (batch_size, 10)
```

所以：

```text
(batch_size, 784) × (784, 10) = (batch_size, 10)
```

因此 `W` 必须是 `(784, 10)`。

### 为什么 `b` 是 `(10,)`

因为每个类别都有一个偏置项。

最终每个样本会得到 10 个输出分数，所以偏置也要对应 10 个类别。

你可以把这层理解成：

```text
输入 784 维特征 -> 输出 10 个类别得分
```

## 5.3 理解 Softmax 的前置例子

```python
X  = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
X.sum(0,keepdim=True),X.sum(1,keepdim=True)
```

这段不是正式模型代码，它是在帮你理解：

- 按列求和
- 按行求和

这里最重要的是：

```python
X.sum(1, keepdim=True)
```

意思是：

- 对每一行求和
- 并且保留二维结构

为什么 Softmax 要对“每一行”求和？

因为：

- 一行代表一个样本
- 一行里有 10 个类别的分数
- 你要把这一行归一化成一个概率分布

所以归一化一定是“按样本做”，也就是按行做。

## 5.4 定义 Softmax

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition
```

这是 Softmax 回归的核心函数之一。

我们把它拆开理解。

### 第一步：对每个分数取指数

```python
X_exp = torch.exp(X)
```

为什么要取指数？

因为：

- 指数以后所有值都变成正数
- 分数越大，指数后的值也越大
- 可以把“相对大小”放大成更明显的差异

### 第二步：对每一行求和

```python
partition = X_exp.sum(1,keepdim=True)
```

这里的作用是：

- 把每个样本对应的所有类别指数值加起来
- 得到归一化常数

### 第三步：每个元素除以这一行的总和

```python
return X_exp/partition
```

这样做以后，每一行都有两个性质：

1. 所有值都是正数
2. 一整行加起来等于 1

这就可以把每一行解释成：

```text
这个样本属于每个类别的概率
```

### 直观理解 Softmax

假设某张图片线性层输出的分数是：

```text
[2.0, 1.0, 0.1]
```

经过 softmax 后可能变成：

```text
[0.66, 0.24, 0.10]
```

这表示：

- 第 0 类概率最高
- 模型最倾向于认为这张图属于第 0 类

### Softmax 不是在“直接选类别”

它做的事情是：

- 把分数转成概率

真正选类别通常是后面做：

```python
argmax
```

也就是选择概率最大的那个类别。

## 5.5 测试 Softmax 函数

```python
X=torch.normal(0,1,(2,5))
X_porb = softmax(X)
X_porb,X_porb.sum(1)
```

这一段是验证 softmax 是否写对。

如果写对了，`X_porb.sum(1)` 的结果应该是：

```text
[1, 1]
```

因为：

- 每一行都应该是一个概率分布
- 一行内所有类别概率之和必须等于 1

## 5.6 定义模型

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W) + b)
```

这是你的 Softmax 回归模型本体。

这行代码可以拆成三步。

### 第一步：把图片拉平

```python
X.reshape((-1, W.shape[0]))
```

由于原始图片通常是：

```text
(batch_size, 1, 28, 28)
```

模型里的线性层希望输入的是二维张量：

```text
(batch_size, 784)
```

所以必须先拉平。

这里的 `W.shape[0]` 就是 `784`。

### 第二步：线性变换

```python
torch.matmul(..., W) + b
```

这一步对应公式：

```text
O = XW + b
```

输出 `O` 的形状是：

```text
(batch_size, 10)
```

这 10 个值不是概率，而是每个类别的原始分数。

### 第三步：做 softmax

```python
softmax(...)
```

把每一行 10 个分数变成 10 个概率。

所以整个模型流程就是：

```text
图片 -> 拉平成 784 维 -> 线性层输出 10 个分数 -> softmax 转成 10 个概率
```

## 5.7 理解标签和预测

```python
y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
y_hat[[0,1],y]
```

这段是在帮你理解：

- 标签 `y` 是怎么表示的
- 怎么从预测概率里取出“正确类别”的概率

### `y = [0, 2]` 表示什么

意思是：

- 第 1 个样本的真实类别是 0
- 第 2 个样本的真实类别是 2

### `y_hat` 表示什么

```python
y_hat = [
  [0.1, 0.3, 0.6],
  [0.3, 0.2, 0.5]
]
```

意思是：

- 第 1 个样本对应 3 个类别的预测概率
- 第 2 个样本对应 3 个类别的预测概率

### `y_hat[[0,1], y]` 为什么重要

它取出的正是：

- 第 1 个样本真实类别对应的预测概率
- 第 2 个样本真实类别对应的预测概率

也就是：

```text
第 1 个样本取 y_hat[0,0]
第 2 个样本取 y_hat[1,2]
```

这正是交叉熵要用到的内容。

## 5.8 定义交叉熵损失

```python
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
cross_entropy(y_hat,y)
```

这是 Softmax 回归里最关键的损失函数。

### 它到底在做什么

它只关心一件事：

```text
真实类别对应的预测概率够不够大
```

如果真实类别概率很大：

- `log(概率)` 不那么负
- 再加个负号以后，loss 就小

如果真实类别概率很小：

- `log(概率)` 会非常负
- 再加负号以后，loss 就很大

### 举个例子

假设真实类别是 2。

如果模型预测：

```text
[0.1, 0.2, 0.7]
```

说明正确类别概率是 `0.7`，不错，所以 loss 比较小。

如果模型预测：

```text
[0.7, 0.2, 0.1]
```

说明正确类别概率只有 `0.1`，很差，所以 loss 很大。

### 为什么交叉熵只取正确类别的概率

因为分类任务最关心的是：

```text
模型有没有把正确类别的概率推高
```

只要正确类别概率高，说明模型判断方向是对的。

## 5.9 定义准确率

```python
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())
```

这一步和 loss 不一样，它不是给训练用的，而是给你“看模型表现”用的。

### 准确率怎么理解

准确率回答的是：

```text
预测对了多少个样本
```

### `argmax(axis=1)` 在做什么

softmax 输出的是概率分布，比如：

```text
[0.1, 0.2, 0.7]
```

真正的类别预测应该是：

```text
选概率最大的那一类
```

这就是：

```python
y_hat.argmax(axis=1)
```

### 后面几行在做什么

```python
cmp=y_hat.type(y.dtype)==y
return float(cmp.type(y.dtype).sum())
```

是在统计：

- 哪些位置预测对了
- 一共有几个预测对

注意，这个函数返回的是：

```text
预测正确的样本个数
```

不是比例。

真正的准确率通常还要再除以样本总数。

## 5.10 简单测试准确率函数

```python
accuracy(y_hat, y) / len(y)
```

这一步就是把“正确个数”除以“样本个数”，得到准确率比例。

## 5.11 定义评估函数

```python
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
```

这个函数是在整个数据集上评估准确率。

### 逐步理解

```python
if isinstance(net,torch.nn.Module):
    net.eval()
```

如果模型是 `nn.Module`，就切换到评估模式。

虽然你当前这个 `net` 不是 `nn.Module` 类，而是手写函数，但这个写法是通用写法。

```python
with torch.no_grad():
```

表示评估时不需要计算梯度。

因为：

- 评估只是看效果
- 不需要更新参数

```python
metric.add(accuracy(net(X),y),y.numel())
```

这是在累计：

1. 预测正确的总个数
2. 样本总个数

最后：

```python
return metric[0]/metric[1]
```

返回整体准确率。

## 5.12 Accumulator 的作用

```python
class Accumulator:
    def __init__(self,n):
        self.data=[0,0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
```

这个类的目的是：

- 累加多个统计量

例如在训练时经常要同时累计：

- 总损失
- 正确预测数
- 总样本数

### 这里有一个小问题

你现在写的是：

```python
self.data=[0,0]*n
```

更合理的写法应该是：

```python
self.data=[0.0]*n
```

原因是：

- `[0,0]*n` 会得到长度为 `2n` 的列表
- 比如 `n=2` 时会变成 `[0,0,0,0]`
- 虽然有时前几个位置刚好还能用，但逻辑上不对

建议你改成：

```python
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
```

这个改动虽然不大，但对你后面继续学很重要，因为统计器逻辑要写准。

## 5.13 评估未训练模型

```python
evaluate_accuracy(net, test_iter)
```

这一步一般是想看：

- 在还没训练之前
- 模型准确率大概多低

由于参数是随机初始化的，所以刚开始准确率通常不会高。

对于 10 分类问题，如果纯随机瞎猜，准确率大概接近：

```text
10%
```

## 5.14 训练一个 epoch

```python
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return  metric[0]/metric[2],metric[1]/metric[2]
```

这段是整个训练过程的核心。

### 先看主线

每个 batch 都在重复这四步：

1. `y_hat = net(X)`：做预测
2. `l = loss(y_hat, y)`：计算损失
3. `backward()`：计算梯度
4. `updater(...)`：更新参数

这和线性回归的训练流程本质完全一样。

### 为什么说 Softmax 回归本质上还是“线性模型 + 损失 + 梯度下降”

因为虽然任务从回归变成了分类，但训练框架并没有变：

```text
前向传播 -> 算损失 -> 反向传播 -> 更新参数
```

### 分支判断在干什么

```python
if isinstance(updater,torch.optim.Optimizer):
```

这一段是在兼容两种更新器：

1. PyTorch 自带优化器
2. 手写更新函数

你的 notebook 目前走的是手写更新函数这一条。

### 对手写更新器时发生了什么

```python
l.sum().backward()
updater(X.shape[0])
```

意思是：

1. 先把一个 batch 的损失求和
2. `backward()` 算出参数梯度
3. 再调用手写 SGD 更新参数

### metric.add 记录了什么

```python
metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
```

累计了三个量：

1. 总损失
2. 正确预测总数
3. 样本总数

最后返回：

```python
metric[0]/metric[2]  # 平均损失
metric[1]/metric[2]  # 平均准确率
```

## 5.15 Animator

```python
class Animator:
    ...
```

这部分主要是为了画训练曲线：

- train loss
- train acc
- test acc

它对你现在理解模型本身不是最核心的，你可以先把它当成：

```text
训练过程可视化工具
```

如果当前觉得类定义太长，看不懂也不用焦虑，先抓住训练主线更重要。

## 5.16 总训练函数

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3,0.9]
                        ,legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

这个函数是在做完整训练。

### 它的逻辑是

对每个 epoch：

1. 训练一轮
2. 在测试集上算准确率
3. 把结果画出来

### `assert` 是干什么

这几行是在做结果检查。

意思是：

- 如果训练结果太差
- 就直接报错提醒你

例如：

```python
assert test_acc > 0.7
```

表示：

- 测试集准确率至少应该大于 70%

这也是在帮助你判断代码有没有写崩。

## 5.17 定义更新器

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

这里是在定义参数更新规则。

注意这不是 PyTorch 的 `optimizer` 对象，而是调用了 `d2l` 封装好的 SGD。

本质仍然是：

```text
新参数 = 旧参数 - 学习率 × 梯度
```

这里的：

- `[W, b]`：要更新的参数
- `lr = 0.1`：学习率
- `batch_size`：用于控制更新尺度

## 5.18 开始训练

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

这一段就是正式训练：

- 用 `cross_entropy` 当损失函数
- 训练 `10` 轮
- 每轮都更新参数并评估测试准确率

## 6. 这一章和线性回归最本质的区别

你已经学过线性回归，所以现在最该建立的是“区别意识”。

### 线性回归

- 输出：一个连续值
- 任务：回归
- 损失：平方损失

### Softmax 回归

- 输出：多个类别分数/概率
- 任务：多分类
- 损失：交叉熵损失

### 但两者共同的训练主线没变

都是：

```text
定义模型 -> 定义损失 -> forward -> backward -> 更新参数
```

这件事非常重要，因为它说明：

```text
你现在不是在重新学一种完全不同的训练方法，而是在同一个训练框架下学新的任务形式
```

## 7. 你最应该看懂的 4 个关键点

如果这章只抓最重要的内容，我建议你一定要彻底弄懂这 4 件事。

### 7.1 为什么 `W` 是 `(784, 10)`

因为：

- 每张图片拉平后是 `784` 维
- 要输出 `10` 个类别分数

所以必须是：

```text
(batch_size, 784) × (784, 10) = (batch_size, 10)
```

### 7.2 Softmax 为什么能输出概率

因为它做了两件事：

1. 先对每个元素取指数，让所有值都变成正数
2. 再按行归一化，让每一行和为 1

这样一行就可以解释成一个概率分布。

### 7.3 交叉熵为什么只看正确类别的概率

因为分类问题最关心的是：

```text
模型有没有把真实类别的概率提上去
```

正确类别概率高，说明模型学对了方向。

### 7.4 为什么训练还是 `backward + SGD`

因为虽然任务变成分类了，但参数学习方式没变：

- 损失定义目标
- `backward()` 计算梯度
- SGD 用梯度更新参数

## 8. 你现在这份代码里值得注意的小问题

最明显的是 `Accumulator` 初始化这里：

当前写法：

```python
self.data=[0,0]*n
```

建议改成：

```python
self.data=[0.0]*n
```

这是一个小 bug，建议你尽快改掉。

另外还有两个小点你也可以留意：

1. `X_porb` 变量名像是拼写笔误，通常更常见的是 `X_prob`
2. 如果后面学更正式的 PyTorch 写法，通常会直接把 `softmax` 和 `cross_entropy` 交给官方函数处理，避免数值稳定性问题

不过对你当前“从零理解原理”的阶段来说，手写版本是有价值的。

## 9. 这章学习的核心心智模型

你可以把 Softmax 回归想成：

```text
先用线性层给每个类别打分
再把分数转成概率
再检查正确类别的概率够不够高
如果不够高，就通过梯度下降调整参数
```

这就是这章全部的本质。

## 10. 学完这章，你至少应该能用自己的话讲清楚这 6 句话

1. Softmax 回归是一个多分类模型
2. 输入图片拉平后是 784 维，输出是 10 个类别分数
3. Softmax 把分数变成概率分布
4. 交叉熵损失关心正确类别的概率是否足够大
5. 准确率通过 `argmax` 判断预测类别是否正确
6. 训练流程仍然是 `预测 -> 算损失 -> backward -> 更新参数`

如果这 6 句话你都能自己顺下来讲清楚，说明你已经真的理解这章的主线了。

## 11. 自测问题

你可以不看代码，先试着回答这些问题：

1. 为什么一张 `28x28` 图片要变成 `784` 维向量？
2. 为什么 `W` 的形状是 `(784, 10)`？
3. Softmax 为什么要按行求和？
4. 为什么交叉熵只取正确类别的概率？
5. `argmax(axis=1)` 表示什么？
6. 为什么准确率和损失函数不是一回事？
7. 为什么分类任务不能继续用线性回归那种单输出形式？
8. Softmax 回归和线性回归的训练流程，哪些一样，哪些不一样？

如果这些问题你能答清楚，这章你就不是“会抄代码”，而是真的开始理解了。

## 12. 最后用最简单的话总结

Softmax 回归就是把线性模型从“输出一个数”改成了“输出多个类别的概率”。

它的主线是：

```text
图片 -> 拉平 -> 线性变换输出 10 个分数 -> softmax 变概率 -> 交叉熵算损失 -> backward 求梯度 -> SGD 更新参数
```

你现在最应该做的，不是继续往后赶，而是把这条主线和每个张量形状彻底吃透。
