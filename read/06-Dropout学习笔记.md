# 06 Dropout 学习笔记

这份笔记是结合你的文件 [06_dropout.ipynb](</D:/Learn/d2l-test/Multilayer Perceptron/06_dropout.ipynb>) 写的，目标是帮助你真正理解：

- 什么是 Dropout
- 为什么它能缓解过拟合
- 训练阶段和测试阶段为什么要区别对待
- 你的 notebook 里手写 Dropout 和 `nn.Dropout` 分别在做什么
- Dropout 应该加在网络的什么位置

如果用一句话概括这一章：

```text
Dropout 的核心思想是：训练时随机丢掉一部分神经元，减少模型对局部特征的过度依赖
```

它是一种非常经典的正则化方法。

## 1. 先用一句人话理解 Dropout

前面你已经学过：

- 欠拟合与过拟合
- 权重衰减

权重衰减是在说：

```text
不要让参数太大
```

而 Dropout 的思路不太一样，它更像是在训练时故意“制造不稳定环境”：

```text
每次训练时，随机让一部分神经元暂时失效
```

这样模型就不能总是依赖同一部分特征或同一条固定路径。

你可以把它理解成：

```text
训练时不让神经网络“抱团取暖”
```

这样模型会被迫学到更鲁棒、更分散的表示。

## 2. 为什么 Dropout 能缓解过拟合

过拟合的一个重要原因是：

```text
模型对训练集中的某些细节依赖得太死
```

尤其是当网络比较大、参数比较多时，模型很容易记住训练数据里的局部模式。

Dropout 的作用就是打断这种过强依赖。

### 直觉上它做了什么

假设某一层有很多神经元。

如果不使用 Dropout：

- 每次训练都使用完整网络
- 某些神经元可能变得特别关键

如果使用 Dropout：

- 每次前向传播都会随机屏蔽一部分神经元
- 网络每次都像是在一个“残缺版本”上训练

这会逼着模型：

- 不要过于依赖单个神经元
- 学会更冗余、更稳定的表示

所以：

```text
Dropout 本质上是在训练时给模型加噪声
```

## 3. Dropout 的基本数学含义

假设某一层输出是：

```text
H
```

Dropout 会先随机生成一个掩码 `mask`：

- 某些位置是 0
- 某些位置是 1

然后做：

```text
H_dropout = H * mask
```

但为了保持期望一致，通常还会做缩放：

```text
H_dropout = H * mask / (1 - dropout)
```

这里的：

- `dropout`：丢弃概率
- `1 - dropout`：保留概率

这个缩放非常重要，因为它保证：

```text
训练时虽然随机丢掉了一部分神经元，但整体输出量级不会系统性变小
```

## 4. 训练阶段和测试阶段为什么不同

这是 Dropout 最必须弄懂的一点。

### 训练阶段

训练时：

- 要随机丢弃一部分神经元

因为我们希望通过这种随机扰动提高泛化能力。

### 测试阶段

测试时：

- 不能继续随机丢神经元

因为测试时我们想要的是：

```text
稳定、确定的预测结果
```

所以测试时要使用完整网络。

这就是为什么：

```text
Dropout 只在训练阶段开启，在推理/测试阶段关闭
```

## 5. 你的 notebook 在做什么

你的 notebook 主线可以概括成：

1. 手写一个 Dropout 层函数
2. 用一个简单张量测试 Dropout 的行为
3. 在一个有两个隐藏层的 MLP 中手动插入 Dropout
4. 训练模型并观察效果
5. 再用 `nn.Dropout` 简洁实现同样的结构

这份 notebook 的重点不是数学推导，而是：

```text
理解 Dropout 在网络中的位置、作用和训练阶段行为
```

## 6. 按你的代码顺序详细讲

## 6.1 导入包和手写 Dropout 层

```python
from d2l import torch as d2l
import torch
from numpy.ma.core import reshape
from torch import nn

def dropout_layer(X,dropout):
    assert 0 <= dropout <=1
    if dropout ==1:
        return torch.zeros_like(X)# 在本情况中，所有元素都被丢弃
    if dropout ==0:
        return X # 在本情况中，所有元素都被保留
    mask = (torch.rand(X.shape) < dropout).float()
    return X * mask/(1.0-dropout)
```

这部分是在手写 Dropout 的底层逻辑。

### `assert 0 <= dropout <= 1`

表示：

- Dropout 概率必须在 0 到 1 之间

这是基本合法性检查。

### `dropout == 1`

```python
if dropout ==1:
    return torch.zeros_like(X)
```

表示：

- 如果丢弃概率是 1
- 那么所有元素都被丢弃

最终输出全 0。

### `dropout == 0`

```python
if dropout ==0:
    return X
```

表示：

- 如果丢弃概率是 0
- 那么没有任何元素被丢掉

输出和原来完全一样。

### 随机掩码 `mask`

```python
mask = (torch.rand(X.shape) < dropout).float()
```

这里是在为 `X` 生成一个和它形状相同的随机掩码。

`torch.rand(X.shape)` 会生成：

- 和 `X` 同形状
- 每个元素在 `[0,1)` 之间随机采样的张量

然后通过条件判断把它变成：

- 0
- 1

### 这一行里有一个逻辑问题

你当前的实现写的是：

```python
mask = (torch.rand(X.shape) < dropout).float()
```

这意味着：

- 有大约 `dropout` 比例的位置会被设成 1

但按照 Dropout 的定义：

```text
dropout 是丢弃概率，不是保留概率
```

所以更常见、更正确的写法应该是：

```python
mask = (torch.rand(X.shape) > dropout).float()
```

或者等价地：

```python
mask = (torch.rand(X.shape) < 1 - dropout).float()
```

这样保留下来的比例才是：

```text
1 - dropout
```

### 缩放部分

```python
return X * mask/(1.0-dropout)
```

这一步是在做“反向缩放”。

目的是：

```text
虽然随机丢掉了一部分神经元，但整体期望值保持不变
```

如果不做这个缩放，输出整体会偏小。

### 建议你理解成

Dropout 这层本质在做三件事：

1. 随机采样哪些位置保留
2. 让被丢掉的位置变成 0
3. 对保留下来的部分做缩放

## 6.2 用小张量测试 Dropout

```python
X = torch.arange(16,dtype=torch.float32).reshape((2,8))
print(X)
print(dropout_layer(X,0))
print(dropout_layer(X,0.5))
print(dropout_layer(X,1))
```

这一段非常适合帮助你建立直觉。

### `dropout_layer(X, 0)`

说明：

- 不丢任何元素
- 输出应和原输入相同

### `dropout_layer(X, 0.5)`

说明：

- 大约丢弃一半元素
- 保留下来的元素还会被缩放

### `dropout_layer(X, 1)`

说明：

- 全部丢弃
- 输出全 0

这部分的目的不是训练，而是帮你直观看到：

```text
Dropout 的行为是随机屏蔽 + 缩放
```

## 6.3 定义模型结构

```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

这里定义了一个两层隐藏层的 MLP。

含义是：

- 输入维度 784
- 输出维度 10
- 第一隐藏层 256 单元
- 第二隐藏层 256 单元

这和你前面学的 MLP 很像，只不过这里开始在隐藏层后面插入 Dropout。

## 6.4 手写模型

```python
dropout1,dropout2=0.2,0.5
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu = nn.ReLU()

    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1,dropout1)# 在第一个全连接层之后添加一个dropout层
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout2) # 在第二个全连接层之后添加一个dropout层
        out  = self.lin3(H2)
        return out
net = Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)
```

这段是 notebook 的主体。

### 模型整体结构

它可以写成：

```text
输入 -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> 输出
```

### 为什么 Dropout 放在隐藏层后面

这是很常见的做法。

因为我们希望：

- 对隐藏表示做随机扰动
- 减少隐藏层之间的共适应

一般不会把 Dropout 放在最终输出层之后。

### `dropout1 = 0.2`, `dropout2 = 0.5`

表示：

- 第一隐藏层丢弃率 0.2
- 第二隐藏层丢弃率 0.5

这说明：

- 越靠后的隐藏层，有时可以用更强的随机失活

当然这不是绝对规则，而是一种常见设置。

### `self.training = is_training`

这里是在自己手动控制：

- 当前模型是不是训练状态

因为你手写了 Dropout，所以需要自己判断：

```python
if self.training == True:
```

只有训练状态下才应用 Dropout。

### `forward` 里的流程

#### 第一步

```python
H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
```

作用：

- 把输入图片拉平
- 经过第一层线性变换
- 过 ReLU

#### 第二步

```python
if self.training == True:
    H1 = dropout_layer(H1,dropout1)
```

作用：

- 在第一层隐藏表示后应用 Dropout

#### 第三步

```python
H2 = self.relu(self.lin2(H1))
```

作用：

- 第二层线性变换
- 再过 ReLU

#### 第四步

```python
if self.training == True:
    H2 = dropout_layer(H2,dropout2)
```

作用：

- 在第二层隐藏表示后再次应用 Dropout

#### 最后输出

```python
out = self.lin3(H2)
```

这里是输出层，没有再做 Dropout。

## 6.5 训练模型

```python
num_epochs,lr,batch_size=10,0.5,256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

这一步和你之前的 MLP 分类任务很像。

### 为什么损失函数还是交叉熵

因为任务没有变：

- 仍然是 Fashion-MNIST 十分类

所以依旧使用：

```python
nn.CrossEntropyLoss
```

### 这里这一章关注的重点不是损失函数

而是：

```text
在相同任务、相似模型下，加入 Dropout 会带来什么变化
```

## 7. 简洁实现

```python
net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Dropout(dropout1),nn.Linear(256,256),nn.ReLU(),nn.Dropout(dropout2),nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

这一部分是在用 PyTorch 官方方式写同样的结构。

### `nn.Dropout(dropout1)`

这是框架已经封装好的 Dropout 层。

它会自动处理：

- 训练时随机丢弃
- 测试时自动关闭

这比手写版更标准，也更安全。

### 这一句非常值得记住

```python
nn.Linear -> nn.ReLU -> nn.Dropout
```

这是非常常见的组合。

### 为什么简洁实现更推荐

因为它：

- 自动区分训练/测试模式
- 不容易自己写错细节
- 更符合实际工程写法

## 8. 你当前 notebook 里最需要注意的一个问题

还是前面提到的这个逻辑：

```python
mask = (torch.rand(X.shape) < dropout).float()
```

如果 `dropout` 表示“丢弃率”，那更合理的实现应该是：

```python
mask = (torch.rand(X.shape) > dropout).float()
```

或者：

```python
mask = (torch.rand(X.shape) < 1 - dropout).float()
```

因为你真正想保留的是：

```text
1 - dropout
```

这点建议你自己在 notebook 里也留意一下。

## 9. Dropout 和权重衰减的区别

这两个都是正则化方法，但思路不同。

### 权重衰减

它做的是：

```text
惩罚大的参数
```

### Dropout

它做的是：

```text
训练时随机让一部分神经元失效
```

所以：

- 权重衰减更像“约束参数幅度”
- Dropout 更像“给网络结构注入随机性”

它们都可以缓解过拟合，但不是同一种方法。

## 10. 这一章最应该彻底弄懂的 5 个点

### 10.1 Dropout 为什么只在训练时开启

因为训练时需要随机扰动提高泛化能力，测试时需要稳定预测。

### 10.2 为什么要对保留下来的神经元做缩放

因为这样可以保持整体输出期望不变。

### 10.3 为什么 Dropout 通常加在隐藏层后面

因为它主要是为了扰动隐藏表示，而不是直接扰动最终输出。

### 10.4 为什么 Dropout 能缓解过拟合

因为它减少了模型对固定局部特征和固定神经元组合的依赖。

### 10.5 为什么更推荐 `nn.Dropout`

因为它自动处理训练/测试差异，写法更标准，出错概率更低。

## 11. 自测问题

你可以试着不看代码回答这些问题：

1. Dropout 的核心思想是什么？
2. 为什么训练时和测试时 Dropout 的行为不同？
3. 如果 dropout = 0 和 dropout = 1，分别意味着什么？
4. 为什么要除以 `1 - dropout`？
5. Dropout 和权重衰减的区别是什么？
6. 你的手写 `dropout_layer` 里，掩码逻辑有没有值得注意的地方？

如果这些问题你能说清楚，说明你已经不只是“会用 Dropout”，而是开始真正理解它了。

## 12. 最后用最简单的话总结

Dropout 这一章最想告诉你的就是：

```text
为了让模型不过度依赖局部细节，可以在训练时随机丢掉一部分神经元
```

它的效果不是让模型更强，而是让模型：

```text
学得更稳、更不容易过拟合
```

当你理解了这一章，你对“正则化”这个概念就不仅仅停留在权重衰减，而是开始接触到了神经网络中另一条非常重要的防过拟合思路。
