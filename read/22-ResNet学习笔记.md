# 22 ResNet 学习笔记

这份笔记是结合你的文件 [22_ResNet.ipynb](</D:/Learn/d2l-test/Convolutional_Neural_Networks/22_ResNet.ipynb>) 写的，目标是帮助你真正理解：

- 为什么深层网络不是简单“越深越好”
- 残差连接到底在做什么
- `Residual` 块里为什么有时需要 `1 x 1` 卷积
- ResNet 的各个阶段如何组织
- 为什么 ResNet 是现代深度网络中非常核心的结构

如果用一句话概括这一章：

```text
ResNet 的核心思想是：让网络学习残差，并通过跳跃连接让深层网络更容易训练
```

## 1. 先用人话理解 ResNet 为什么出现

按直觉，网络越深，表达能力应该越强。

但实际训练中会遇到一个问题：

```text
层数增加后，训练误差反而可能变差
```

这不一定是过拟合，因为过拟合通常表现为训练误差低、测试误差高。

这里的问题是：

```text
深层网络本身变得难以优化
```

ResNet 的想法是：

```text
如果多加的层暂时学不到更好的变换，至少应该很容易学成“什么都不改变”
```

也就是让网络更容易表示恒等映射。

## 2. 什么是残差连接

普通网络块做的是：

```text
输入 X -> 一堆层 -> 输出 F(X)
```

ResNet 块做的是：

```text
输入 X -> 一堆层 -> F(X)
然后输出 F(X) + X
```

这里的 `X` 直接绕过中间层，加到输出上。

这条路径叫跳跃连接或残差连接。

它的直觉是：

```text
网络不用直接学习目标映射 H(X)，而是学习 H(X) - X 这个残差
```

最后输出：

```text
H(X) = F(X) + X
```

如果某个块暂时不需要改变输入，只要让 `F(X)` 接近 0，就能接近恒等映射。

## 3. `Residual` 类长什么样

notebook 里定义了：

```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
```

一个残差块主路径包含：

1. `3 x 3` 卷积
2. BatchNorm
3. ReLU
4. `3 x 3` 卷积
5. BatchNorm

然后把输入 `X` 加回来，再经过 ReLU。

## 4. `forward` 里真正发生了什么

核心代码是：

```python
Y = F.relu(self.bn1(self.conv1(X)))
Y = self.bn2(self.conv2(Y))
if self.conv3:
    X = self.conv3(X)
Y += X
return F.relu(Y)
```

可以拆成四步：

1. 主路径对 `X` 做两次卷积变换，得到 `Y`
2. 如果形状不匹配，就用 `conv3` 调整 `X`
3. 把主路径输出 `Y` 和跳跃路径输入 `X` 相加
4. 对相加结果做 ReLU

这里最关键的是：

```text
主路径学习变化量，跳跃路径保留原始信息
```

## 5. 什么时候需要 `1 x 1` 卷积

残差相加要求两个张量形状一致。

也就是说：

```text
Y 和 X 的通道数、高、宽都必须能对上
```

如果输入输出通道数相同、空间尺寸也相同，就可以直接相加。

notebook 里验证了：

```python
blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

这时不需要 `1 x 1` 卷积。

但如果要把通道从 3 变成 6，同时空间尺寸减半，就要写：

```python
blk = Residual(3, 6, use_1x1conv=True, strides=2)
```

这个 `1 x 1` 卷积的作用是：

```text
把跳跃路径上的 X 调整成和主路径输出同样的形状
```

所以它不是为了提取复杂空间特征，而是为了让残差相加合法。

## 6. ResNet 第一段 `b1` 在做什么

notebook 里写了：

```python
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

这一段类似很多 CNN 的 stem：

```text
先用较大卷积提取初级特征，再用池化降低空间尺寸
```

输入是灰度图，所以输入通道是 1。

输出变成 64 通道，为后面的残差块做准备。

## 7. `resnet_block` 如何组织多个残差块

代码里定义了：

```python
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

这里的规则是：

- 每个阶段有多个残差块
- 除第一个阶段外，每个新阶段的第一个残差块会下采样
- 下采样时用 `strides=2`
- 通道数变化时用 `1 x 1` 卷积对齐跳跃路径

这体现了 ResNet 的典型设计：

```text
阶段之间降低空间尺寸、增加通道数；阶段内部保持尺寸，继续加深表示
```

## 8. ResNet 主体结构

notebook 里写了：

```python
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

可以看出通道数变化是：

```text
64 -> 64 -> 128 -> 256 -> 512
```

空间尺寸则会逐步减小。

最后：

```python
nn.AdaptiveAvgPool2d((1,1))
nn.Flatten()
nn.Linear(512, 10)
```

说明高层特征被全局平均池化成 512 维向量，再做 10 类分类。

## 9. 为什么 ResNet 特别适合深层网络

残差连接带来的好处可以从两个角度理解：

### 信息流

输入可以通过跳跃连接直接传到后面，信息不容易在层层变换中丢失。

### 梯度流

反向传播时，梯度也能通过跳跃路径更顺畅地传回前面层。

所以 ResNet 能训练比传统 CNN 深得多的网络。

它解决的不是“怎么让网络有更多参数”，而是：

```text
怎么让很深的网络依然容易优化
```

## 10. ResNet 和前面网络的关系

### VGG

```text
通过规则堆叠小卷积加深网络
```

### GoogLeNet

```text
通过并行多尺度路径扩展网络表达
```

### ResNet

```text
通过残差连接让更深网络可以稳定训练
```

ResNet 的思想非常通用，后来不只用于 CNN，也影响了 Transformer 等很多深层模型。

## 11. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. 残差块输出的是 `F(X) + X`
2. 跳跃连接让网络更容易学习恒等映射
3. 形状不一致时，用 `1 x 1` 卷积对齐跳跃路径
4. ResNet 按阶段逐步降低空间尺寸、增加通道数
5. 残差连接改善了深层网络的信息流和梯度流

如果最后只留一句话：

```text
ResNet 的厉害之处不是单纯更深，而是给深层网络加了一条更容易训练下去的路
```
