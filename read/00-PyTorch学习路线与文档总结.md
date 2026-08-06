# 00 PyTorch 学习路线与文档总结
这份文档的目标不是罗列 API，而是帮你建立一个能长期使用的 PyTorch 学习框架。

如果你现在的感觉是：

- 代码能照着敲
- 但看不懂每一段为什么这么写
- 看到 `tensor`、`backward`、`optimizer`、`DataLoader` 就容易乱

那么这份文档就是给你用的。

它主要基于 PyTorch 官方入门教程整理，并结合你当前学习深度学习的需求，做成中文学习版。

## 1. PyTorch 到底是什么

PyTorch 可以先理解成两层东西：

1. 一个强大的张量计算库
2. 一个深度学习训练框架

也就是说，PyTorch 既能做：

- 数组和矩阵计算
- GPU 加速
- 自动求导

也能做：

- 定义模型
- 训练模型
- 保存模型
- 加载模型
- 推理预测

如果只用一句话概括：

```text
PyTorch = 张量计算 + 自动求导 + 神经网络训练工具
```

## 2. 官方文档最适合你的学习入口

如果你现在还在入门阶段，最推荐的不是直接翻 API 文档，而是先学官方的 `Learn the Basics` 这条线。

官方入门主线是：

1. Tensors
2. Datasets & DataLoaders
3. Build the Neural Network
4. Autograd
5. Optimization
6. Save & Load Model

这条顺序非常合理，因为它刚好对应训练一个神经网络所需要的最小闭环：

```text
数据 -> 模型 -> 损失 -> 梯度 -> 参数更新 -> 保存结果
```

## 3. 你必须先建立的 PyTorch 心智模型

看不懂 PyTorch 代码，往往不是因为语法难，而是因为没有先分清“每个对象在训练里扮演什么角色”。

你可以先把一份典型训练代码拆成 6 个角色：

1. 数据
2. 模型
3. 损失函数
4. 梯度
5. 优化器
6. 训练循环

几乎所有 PyTorch 训练代码，无论是线性回归、MLP、CNN、RNN 还是 Transformer，本质上都在重复这套结构。

你以后看代码时，先不要急着看每一行，而是先问：

1. 数据在哪里？
2. 模型在哪里？
3. 损失函数在哪里？
4. 哪一行在反向传播？
5. 哪一行在更新参数？
6. 哪一行在做验证或测试？

只要这 6 个角色看清了，代码就不会显得那么乱。

## 4. Tensor：PyTorch 的基础数据结构

PyTorch 里最基础的东西是 `Tensor`。

你可以先把它理解成：

- 比 NumPy 数组更适合深度学习的多维数组

它和普通数组最大的不同在于：

1. 它可以方便地放到 GPU 上运行
2. 它能参与自动求导

在深度学习里，下面这些东西通常都是 tensor：

- 输入数据 `X`
- 标签 `y`
- 模型参数 `w`
- 模型输出 `y_hat`
- 损失值 `loss`
- 梯度 `grad`

### 4.1 你最先要看懂的 tensor 属性

一个 tensor 最重要的几个属性是：

- `shape`：形状
- `dtype`：数据类型
- `device`：在 CPU 还是 GPU 上

例如：

```python
import torch

x = torch.rand(3, 4)
print(x.shape)
print(x.dtype)
print(x.device)
```

你以后看到模型代码，第一反应应该是：

```text
这个 tensor 的形状是什么？
```

因为深度学习里大量问题，其实都是“形状没搞清楚”。

### 4.2 常见 tensor 创建方式

```python
torch.tensor([[1, 2], [3, 4]])
torch.zeros(2, 3)
torch.ones(2, 3)
torch.rand(2, 3)
torch.randn(2, 3)
```

常见理解：

- `tensor(...)`：从现有数据创建
- `zeros(...)`：全 0
- `ones(...)`：全 1
- `rand(...)`：0 到 1 的均匀随机数
- `randn(...)`：标准正态分布随机数

### 4.3 Tensor 不是只拿来存数据

这是初学者特别容易忽略的一点。

在 PyTorch 里，tensor 不只是“输入数据容器”，还是：

- 参数容器
- 运算结果容器
- 梯度承载对象

比如：

```python
w = torch.randn(2, 1, requires_grad=True)
```

这个 `w` 既是 tensor，又是一个需要训练的参数。

## 5. Dataset 和 DataLoader：训练数据怎么喂给模型

深度学习训练不是一次把所有数据硬塞给模型，而是通常一批一批地喂。

PyTorch 里，和数据加载最相关的两个对象是：

1. `Dataset`
2. `DataLoader`

### 5.1 Dataset 是什么

`Dataset` 可以理解成：

- 一个“数据集对象”
- 它负责知道“第 i 条样本长什么样”

最简单的自定义 Dataset 通常至少要实现：

```python
__len__()
__getitem__(idx)
```

含义是：

- `__len__`：数据集一共有多少条
- `__getitem__`：给我一个编号，我返回这一条样本

### 5.2 DataLoader 是什么

`DataLoader` 是在 `Dataset` 外面再套一层。

它负责：

- 按 batch 取数据
- 是否打乱顺序
- 是否并行读取

例如：

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

这行的意思是：

- 每次取 32 条数据
- 每个 epoch 前打乱样本顺序

### 5.3 为什么要 mini-batch

因为训练时通常不会：

- 一次只用 1 条数据
- 也不会每次都用全量数据

更常见的是：

- 每次取一小批数据
- 用这批数据计算损失和梯度
- 然后更新一次参数

这就是 mini-batch 训练。

### 5.4 你看训练代码时怎么识别数据部分

最常见的形式是：

```python
for X, y in train_loader:
    ...
```

这通常说明：

- `X` 是输入特征
- `y` 是标签
- `train_loader` 是 DataLoader

你以后看到这个结构，就知道训练循环开始了。

## 6. nn.Module：模型在 PyTorch 里怎么写

PyTorch 里，模型通常是 `nn.Module`。

你可以把 `nn.Module` 理解成：

- 所有神经网络模块的统一父类

这包括：

- 单层线性层 `nn.Linear`
- 卷积层 `nn.Conv2d`
- 激活函数模块
- 整个神经网络本身

### 6.1 最简单的模型定义思路

通常写法是：

```python
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)
```

这里有两个最关键的部分：

1. `__init__`
2. `forward`

### 6.2 `__init__` 是干什么的

`__init__` 里主要做的是：

- 定义模型有哪些层

例如：

```python
self.linear = nn.Linear(2, 1)
```

这表示模型里有一个线性层：

- 输入维度是 2
- 输出维度是 1

### 6.3 `forward` 是干什么的

`forward` 里主要做的是：

- 定义输入数据经过模型时的前向传播过程

例如：

```python
def forward(self, x):
    return self.linear(x)
```

意思就是：

- 输入 `x`
- 经过线性层
- 输出预测结果

### 6.4 为什么模型也是一个 Module

因为 PyTorch 的设计是“模块嵌套模块”。

也就是说：

- 一层是 module
- 一个更大的网络也是 module

这让复杂网络也能统一管理。

### 6.5 `model.parameters()` 是什么

当你写：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

`model.parameters()` 的意思是：

- 取出模型里所有需要训练的参数

这些参数通常包括：

- 权重 `weight`
- 偏置 `bias`

## 7. 前向传播：模型是怎么做预测的

在 PyTorch 里，最常见的一句代码是：

```python
pred = model(X)
```

你要把这句理解为：

- 把输入 `X` 送进模型
- 自动调用 `forward`
- 得到输出 `pred`

这就是前向传播。

你可以把它想成：

```text
输入 -> 经过若干层运算 -> 输出预测值
```

对于线性回归：

```text
X -> Xw+b -> y_hat
```

对于多层感知机：

```text
X -> 线性层 -> 激活函数 -> 线性层 -> 输出
```

## 8. 损失函数：模型到底错了多少

损失函数的作用只有一个：

- 定义“预测错了多少”

例如在线性回归里常见的是：

```python
loss_fn = nn.MSELoss()
```

它衡量的是：

- 预测值和真实值之间的均方误差

在分类问题里，常见的是：

```python
loss_fn = nn.CrossEntropyLoss()
```

它更适合“预测类别”的任务。

### 8.1 为什么必须有损失函数

因为模型训练时必须知道：

- 当前预测好不好
- 哪个方向能让结果变得更好

损失函数不给出“更新规则”，但它提供了“优化目标”。

### 8.2 你可以这样记

```text
损失函数负责定义目标：我要让这个值尽量小
```

## 9. Autograd：PyTorch 为什么会自动求导

这是 PyTorch 最关键的能力之一。

如果没有自动求导，神经网络训练会非常痛苦，因为你得手工推每个参数的导数。

PyTorch 通过 `autograd` 帮你做这件事。

### 9.1 `requires_grad=True` 的含义

例如：

```python
w = torch.randn(2, 1, requires_grad=True)
```

意思是：

- 这个 tensor 以后需要计算梯度

PyTorch 会在后续运算中记录和它相关的计算图。

### 9.2 计算图是什么

你可以粗略理解成：

- PyTorch 会把一连串运算过程记下来

比如：

```python
y_hat = X @ w + b
loss = ((y_hat - y) ** 2).mean()
```

PyTorch 会记住：

```text
w, b -> y_hat -> loss
```

这样当你之后调用：

```python
loss.backward()
```

它就能从 `loss` 反过来一层层求导，最终算出：

- `dLoss/dw`
- `dLoss/db`

### 9.3 `backward()` 做了什么

```python
loss.backward()
```

这句的作用不是更新参数，而是：

- 计算梯度
- 把结果存到参数的 `.grad` 属性里

例如：

- `w.grad`
- `b.grad`

### 9.4 为什么要清零梯度

因为在 PyTorch 里，梯度默认是“累加”的。

也就是说，如果你不清零：

- 这次的梯度会叠加到上次的梯度上

所以训练时几乎总会看到：

```python
optimizer.zero_grad()
```

或者手写参数更新时：

```python
param.grad.zero_()
```

### 9.5 `torch.no_grad()` 是干什么的

它表示：

- 下面这段代码不要构建计算图
- 不需要算梯度

常见用途有两个：

1. 参数更新时
2. 验证和推理时

例如：

```python
with torch.no_grad():
    y_pred = model(X)
```

## 10. 优化器：参数是怎么被改动的

损失函数只负责定义“错了多少”。

真正负责改参数的是优化器。

最常见的是随机梯度下降：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 10.1 优化器和损失函数的关系

它们的关系可以概括成：

```text
损失函数给出优化目标
-> backward() 算出梯度
-> 优化器根据梯度更新参数
```

也就是说：

- 损失函数不更新参数
- 优化器不负责定义“错没错”
- 两者通过“梯度”连接起来

### 10.2 优化的标准三步

PyTorch 训练里最经典的三步是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

逐步理解：

1. `optimizer.zero_grad()`
   - 清空旧梯度

2. `loss.backward()`
   - 根据损失计算各参数梯度

3. `optimizer.step()`
   - 按照优化算法更新参数

### 10.3 `optimizer.step()` 为什么知道梯度

因为梯度已经被 `loss.backward()` 算好，并放进了各个参数的 `.grad` 里。

所以优化器并不是“自己算梯度”，而是：

- 读取 `param.grad`
- 用这些梯度更新参数

### 10.4 学习率是什么

学习率 `lr` 决定：

- 每次参数更新迈多大一步

如果学习率太大：

- 容易震荡
- 甚至发散

如果学习率太小：

- 学得很慢

所以学习率是训练里非常重要的超参数。

## 11. 训练循环：几乎所有 PyTorch 代码的主干

最核心的训练模板通常长这样：

```python
for epoch in range(num_epochs):
    for X, y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

这段代码你以后会反复看到。

你可以把它翻译成人话：

1. 把一批数据送进模型
2. 得到预测结果
3. 计算预测和真实值的差距
4. 清空上次梯度
5. 反向传播，求这次梯度
6. 更新参数

### 11.1 epoch 是什么

`epoch` 的意思是：

- 整个训练集被完整看过一遍

如果你有 1000 条数据，batch_size 是 100：

- 一个 epoch 大约包含 10 次参数更新

### 11.2 batch 是什么

batch 就是：

- 一次送进模型的一小批样本

例如：

```python
batch_size = 32
```

表示每次拿 32 条数据一起训练。

### 11.3 iteration 是什么

每处理一个 batch，通常就算一次 iteration。

所以：

- 一个 epoch 包含多个 iteration

## 12. 训练模式和推理模式

很多人刚开始会忽略这个区别。

PyTorch 里通常有两种模式：

1. 训练模式
2. 推理模式

### 12.1 训练模式

```python
model.train()
```

用于训练时。

对一些层来说，例如：

- Dropout
- BatchNorm

训练模式和测试模式的行为是不一样的。

### 12.2 推理模式

```python
model.eval()
```

用于验证和测试时。

常常和下面配合：

```python
with torch.no_grad():
    pred = model(X)
```

这样做的好处是：

- 不需要梯度
- 节省内存
- 推理更稳定

## 13. 保存和加载模型

训练完模型以后，你通常希望把结果保存下来，不然下次还要重新训练。

PyTorch 最推荐的做法是保存 `state_dict`。

### 13.1 什么是 `state_dict`

它可以理解成：

- 模型内部参数的字典

里面主要存：

- 权重
- 偏置

### 13.2 保存参数

```python
torch.save(model.state_dict(), "model_weights.pth")
```

### 13.3 加载参数

```python
model = MyModel()
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model.eval()
```

为什么要先重新创建模型结构，再加载参数？

因为：

- 参数只告诉你数值是多少
- 模型类本身才定义了网络结构

### 13.4 为什么推荐保存 `state_dict`

因为它更稳定，也更符合官方推荐。

相比直接 `torch.save(model, ...)`：

- `state_dict` 更清晰
- 更便于迁移
- 对类定义依赖更小

## 14. 你现在学深度学习，最该优先掌握的 PyTorch 模块

如果你还在 D2L 前期，不要试图一口气掌握整个 PyTorch。

最应该先掌握的是这些：

1. `torch.tensor`
2. `shape / dtype / device`
3. 张量运算
4. `requires_grad`
5. `loss.backward()`
6. `nn.Module`
7. `nn.Linear`
8. `nn.ReLU`
9. `nn.Sequential`
10. `DataLoader`
11. `nn.MSELoss`
12. `nn.CrossEntropyLoss`
13. `torch.optim.SGD`
14. `model.train()` / `model.eval()`
15. `torch.save()` / `load_state_dict()`

你把这些吃透，已经能覆盖前中期大多数深度学习实验代码。

## 15. 看懂深度学习代码的通用方法

以后你遇到任何 PyTorch 项目，不管是教程代码还是别人仓库代码，都可以按下面顺序读。

### 第一步：先找数据入口

重点找：

- `Dataset`
- `DataLoader`
- `for X, y in ...`

先搞清楚：

- 输入是什么
- 标签是什么
- batch shape 是什么

### 第二步：找模型定义

重点看：

- `class XXX(nn.Module)`
- `forward`
- `nn.Sequential`

先搞清楚：

- 输入维度
- 中间层
- 输出维度

### 第三步：找损失函数

重点看：

- `loss_fn = ...`

先搞清楚：

- 这是回归还是分类
- 用的是哪种损失

### 第四步：找优化器

重点看：

- `optimizer = ...`

先搞清楚：

- 优化的是哪些参数
- 学习率是多少

### 第五步：找训练循环

重点找：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这是训练最核心的三行。

### 第六步：找验证和测试逻辑

重点看：

- `model.eval()`
- `torch.no_grad()`

先搞清楚：

- 哪些代码是在训练
- 哪些代码是在评估

## 16. 结合你当前 D2L 学习阶段的建议

你现在学的是前期基础章节，所以建议把 PyTorch 学习和 D2L 配套起来。

### 16.1 线性回归阶段

重点掌握：

- tensor 基本运算
- `requires_grad`
- 手写损失函数
- 手写 SGD
- `backward()`

### 16.2 softmax 回归阶段

重点掌握：

- 输出维度
- 分类标签形式
- `CrossEntropyLoss`
- 准确率的计算

### 16.3 多层感知机阶段

重点掌握：

- `nn.Linear`
- 激活函数，例如 `ReLU`
- 多层前向传播

### 16.4 卷积神经网络阶段

重点掌握：

- 张量形状变化
- `nn.Conv2d`
- `nn.MaxPool2d`
- 通道数

### 16.5 RNN / Attention / Transformer 阶段

重点掌握：

- 序列维度
- masking
- 嵌入层
- 注意力矩阵 shape

也就是说：

- PyTorch 学习不是和深度学习分开的
- 而是每学一个模型，顺手理解这一章里对应的 PyTorch 工具

## 17. 你现在最容易犯的几个误区

### 17.1 只记 API，不理解训练流程

例如你记住了：

- `nn.Linear`
- `MSELoss`
- `SGD`

但如果你不知道它们在训练链条中的位置，还是很容易看不懂代码。

### 17.2 只看语法，不看 shape

很多初学者最痛苦的问题，其实不是数学，而是 tensor 形状不清楚。

所以你一定要经常问自己：

```text
这个 tensor 的 shape 是什么？
```

### 17.3 看到 `backward()` 就害怕

其实你在当前阶段不需要一开始就把所有链式法则细节都推明白。

你先记住：

- `backward()` 负责把梯度算出来
- 梯度会存进参数的 `.grad`
- 优化器会用这些梯度更新参数

先把流程吃透，再回头补数学细节。

### 17.4 看不懂代码就不断抄代码

抄代码可以帮助熟悉语法，但不能替代理解。

更有效的方式是：

1. 看一小段
2. 用自己的话说出它在干什么
3. 再继续往下看

## 18. 一个你可以长期复用的最小训练模板

下面这份模板，你以后学很多模型都可以拿来对照：

```python
import torch
from torch import nn
from torch.utils.data import DataLoader

model = MyModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

for epoch in range(num_epochs):
    model.train()
    for X, y in train_loader:
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for X, y in valid_loader:
            pred = model(X)
            ...
```

你要练到什么程度？

练到你看到这段代码时，能够直接指出：

- 哪些是数据
- 哪些是模型
- 哪些是损失
- 哪些是梯度
- 哪些是参数更新
- 哪些是验证逻辑

## 19. 给你的推荐学习顺序

如果你的目标是“真正能看懂深度学习代码”，建议按下面顺序学。

### 第 1 阶段：PyTorch 生存基础

先掌握：

1. tensor 创建与 shape
2. 张量运算
3. `requires_grad`
4. `backward`
5. `DataLoader`
6. `nn.Module`
7. `optimizer`

这个阶段的目标是：

- 能看懂线性回归、softmax 回归、MLP 代码

### 第 2 阶段：神经网络常见模块

再掌握：

1. `nn.Linear`
2. `nn.ReLU`
3. `nn.Sequential`
4. `nn.Conv2d`
5. `nn.MaxPool2d`
6. `Dropout`
7. `BatchNorm`

这个阶段的目标是：

- 能看懂常见图像模型和简单全连接网络

### 第 3 阶段：训练工程化

再掌握：

1. `model.train()` / `eval()`
2. checkpoint 保存与加载
3. GPU / device 管理
4. 学习率调整
5. 验证集和测试集流程

这个阶段的目标是：

- 能自己完整训练一个模型

### 第 4 阶段：高级模型阅读

再进入：

1. RNN / LSTM
2. Attention
3. Transformer
4. 自定义模块与复杂 forward

这个阶段的目标是：

- 能看懂主流深度学习论文复现代码

## 20. 学习时最有用的自测问题

每学完一个 notebook，你都可以拿这几个问题问自己：

1. 这个任务的输入和输出分别是什么？
2. 模型结构是什么？
3. 哪个对象在存参数？
4. 损失函数定义了什么目标？
5. 梯度是在哪一步算出来的？
6. 优化器在哪一步更新参数？
7. 训练和测试模式有什么区别？
8. 每个关键 tensor 的 shape 是多少？

如果这些问题答不清，说明你对代码还是“会抄，不会读”。

如果能答清，就说明你已经开始具备独立读代码的能力了。

## 21. 最后给你一个总总结

PyTorch 真正难的，不是 API 数量，而是训练流程的心智模型。

你只要抓住这条主线：

```text
数据 -> 模型 -> 预测 -> 损失 -> backward -> 梯度 -> optimizer.step -> 更好的参数
```

再复杂的深度学习代码，也只是这条主线的不同变体。

所以你现在的学习重点，不应该是“把所有函数背下来”，而应该是：

1. 看懂 tensor 和 shape
2. 看懂 forward
3. 看懂 backward
4. 看懂 optimizer
5. 看懂训练循环

这五件事一旦打通，你后面学 D2L、看论文复现代码、自己写实验代码，都会轻松很多。

## 22. 官方文档参考链接

下面这些是我整理这份笔记时参考的 PyTorch 官方教程入口，你后面可以按顺序继续看：

1. Learn the Basics
   https://docs.pytorch.org/tutorials/beginner/basics/index.html

2. Tensors
   https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

3. Datasets & DataLoaders
   https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

4. Build the Neural Network
   https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

5. Automatic Differentiation with torch.autograd
   https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

6. Optimizing Model Parameters
   https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

7. Save and Load the Model
   https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
