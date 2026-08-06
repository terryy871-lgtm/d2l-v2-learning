# 23 DenseNet 学习笔记

这份笔记是结合你的文件 [23_DenseNet.ipynb](</D:/Learn/d2l-test/Convolutional_Neural_Networks/23_DenseNet.ipynb>) 写的，目标是帮助你真正理解：

- DenseNet 和 ResNet 的连接方式有什么不同
- 稠密块（DenseBlock）为什么要在通道维度拼接特征
- 增长率（growth rate）是什么意思
- 转换层（transition block）为什么要压缩通道和空间尺寸
- DenseNet 如何复用前面层的特征

如果用一句话概括这一章：

```text
DenseNet 的核心思想是：让每一层都直接使用前面所有层的特征，并把新特征不断拼接到通道维度上
```

## 1. 先用人话理解 DenseNet 想解决什么

ResNet 通过相加实现跳跃连接：

```text
输出 = F(X) + X
```

DenseNet 的思路不一样。

它不是把新特征和旧特征相加，而是直接拼接：

```text
输出 = concat(X, F(X))
```

这样做的结果是：

```text
后面的层可以直接看到前面所有层产生过的特征
```

所以 DenseNet 的名字里有 Dense：

```text
层与层之间连接非常密集
```

## 2. DenseNet 和 ResNet 的最大区别

### ResNet

```text
把输入和输出相加，通道数通常保持一致或通过 1 x 1 卷积对齐
```

### DenseNet

```text
把输入和新输出在通道维度拼接，通道数会不断增加
```

一个简单直觉是：

```text
ResNet 像是在原特征上做修正，DenseNet 像是在保留原特征的同时不断追加新特征
```

这让 DenseNet 特别强调特征复用。

## 3. `conv_block` 在做什么

notebook 里先定义了：

```python
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

这个块的顺序是：

```text
BatchNorm -> ReLU -> Conv
```

它和前面常见的 `Conv -> BatchNorm -> ReLU` 顺序略有不同。

这种先归一化再激活再卷积的写法，常见于一些后续改进网络结构。

对你现在来说，重点是：

```text
每个 conv_block 会根据当前已有特征，生成 num_channels 个新特征通道
```

## 4. DenseBlock 是怎么拼接特征的

核心代码是：

```python
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
```

最关键的是这一句：

```python
X = torch.cat((X, Y), dim=1)
```

`dim=1` 是通道维度。

也就是说，每经过一个小卷积块：

1. 用当前所有已有特征生成新特征 `Y`
2. 把新特征拼到原来的 `X` 后面
3. 下一层再拿拼接后的全部特征作为输入

这就是 DenseBlock 的核心。

## 5. 为什么每一层输入通道数会变多

构造 DenseBlock 时有：

```python
num_channels * i + input_channels
```

假设：

- 初始输入通道数是 3
- 每个卷积块新增 10 个通道
- 稠密块里有 2 个卷积块

那么：

### 第 1 个卷积块

输入通道数是：

```text
3
```

输出新增：

```text
10
```

拼接后通道数变成：

```text
3 + 10 = 13
```

### 第 2 个卷积块

输入通道数是：

```text
13
```

再新增：

```text
10
```

最终通道数变成：

```text
23
```

这正好对应 notebook 里的测试：

```python
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

输出通道数就是 23。

## 6. 什么是增长率 growth rate

DenseNet 里有：

```python
num_channels, growth_rate = 64, 32
```

这里的 `growth_rate` 表示：

```text
DenseBlock 中每个卷积块新增多少个通道
```

如果增长率是 32，那么每经过 DenseBlock 内的一层，通道数就增加 32。

增长率越大：

- 新增特征越多
- 表达能力越强
- 计算和显存开销也越大

所以它是 DenseNet 中很重要的规模控制参数。

## 7. 为什么需要转换层

因为 DenseBlock 会不断拼接特征，通道数会越来越多。

如果一直这么增长，网络会越来越重。

所以 DenseNet 在稠密块之间加入转换层：

```python
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

转换层做两件事：

### `1 x 1` 卷积

```text
压缩通道数
```

### `2 x 2` 平均池化

```text
把空间高宽减半
```

所以转换层像是 DenseBlock 之间的整理步骤：

```text
特征变多后，先压一压，再进入下一个阶段
```

## 8. DenseNet 主体是怎么搭起来的

notebook 里写了：

```python
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

这说明网络中有 4 个 DenseBlock，每个 DenseBlock 里有 4 个卷积块。

每个卷积块新增 32 个通道。

所以一个 DenseBlock 会让通道数增加：

```text
4 * 32 = 128
```

但除了最后一个 DenseBlock，中间都会接转换层，把通道数减半。

这就是 DenseNet 控制模型规模的方式。

## 9. 最终分类部分怎么做

最后网络是：

```python
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

这和 ResNet 类似：

1. 前面提取高层特征
2. BatchNorm + ReLU 做最后整理
3. 全局平均池化把每个通道压成一个值
4. 线性层输出 10 类

这里没有大型全连接层，说明现代 CNN 越来越倾向于：

```text
用卷积主体学习强特征，再用轻量分类头完成分类
```

## 10. DenseNet 的优势应该怎么理解

DenseNet 的优势主要来自特征复用。

因为每一层都可以直接访问前面层的输出，所以：

- 低层细节特征不会轻易丢失
- 后层可以复用已有特征，不必重复学习
- 梯度也更容易传回前面的层

这和 ResNet 有相似目标，但方法不同。

ResNet 是：

```text
通过相加让信息更容易流动
```

DenseNet 是：

```text
通过拼接让所有历史特征都保留下来
```

## 11. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. DenseBlock 会把输入和新特征在通道维度拼接
2. 每个卷积块新增的通道数叫增长率
3. DenseBlock 内部通道数会不断增加
4. transition block 用 `1 x 1` 卷积和平均池化压缩规模
5. DenseNet 强调特征复用，而不是简单加深网络

如果最后只留一句话：

```text
DenseNet 的精髓是把前面学到的特征一路保留下来，让后面的每一层都站在所有已有特征之上继续学习
```
