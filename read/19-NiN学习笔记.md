# 19 NiN 学习笔记

这份笔记是结合你的文件 [19_NiN.ipynb](</D:/Learn/d2l-test/Convolutional_Neural_Networks/19_NiN.ipynb>) 写的，目标是帮助你真正理解：

- NiN 为什么叫 Network in Network
- `1 x 1` 卷积在 NiN 中到底起什么作用
- NiN 块和普通卷积块有什么不同
- 为什么 NiN 可以去掉传统的大型全连接层
- 全局平均池化如何完成分类

如果用一句话概括这一章：

```text
NiN 的核心思想是：在每个空间位置上用小型神经网络增强特征表达，并用全局平均池化替代大规模全连接层
```

## 1. 先用人话理解 NiN 想解决什么

AlexNet 和 VGG 的后半部分都有很大的全连接层。

这些全连接层表达能力强，但问题也明显：

- 参数量大
- 容易过拟合
- 和空间结构的联系被展平后削弱

NiN 的思路是：

```text
既然卷积层已经能提特征，能不能让卷积部分直接输出类别相关的特征图，然后不要再依赖巨大的全连接分类器？
```

这就是 NiN 的重要变化。

它不是只在网络末尾做分类，而是让卷积结构本身更有表达能力。

## 2. 什么是 NiN 块

notebook 里定义了：

```python
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

一个 NiN 块由三层卷积组成：

1. 一个普通卷积，负责看空间邻域
2. 一个 `1 x 1` 卷积，负责通道混合
3. 再一个 `1 x 1` 卷积，继续增强通道表达

你可以把它理解成：

```text
普通卷积负责提局部空间特征，后面的 1 x 1 卷积负责在每个位置上做更复杂的非线性变换
```

这就是 Network in Network 的含义：

```text
在卷积网络的每个局部位置里，再放入一个小型网络
```

## 3. `1 x 1` 卷积在这里为什么重要

前面你已经学过，`1 x 1` 卷积不会看周围邻域。

它只做一件事：

```text
对同一个空间位置上的通道向量做线性组合
```

在 NiN 里，`1 x 1` 卷积的作用更进一步：

```text
把每个像素位置上的特征向量，交给一个小型多层感知机处理
```

因为连续两个 `1 x 1` 卷积之间都有 `ReLU`，所以这不是简单线性变换，而是带非线性的局部特征重组。

这相当于在每个空间位置上都做：

```text
通道特征 -> 隐式小网络 -> 更强的通道特征
```

## 4. NiN 网络结构整体长什么样

notebook 里定义了：

```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())
```

前面几个 NiN 块负责逐步提取特征。

最后一个 NiN 块很特别：

```python
nin_block(384, 10, kernel_size=3, strides=1, padding=1)
```

它直接输出 10 个通道。

这 10 个通道对应 10 个类别。

也就是说，NiN 尝试让卷积层直接产生类别相关特征图。

## 5. 为什么 NiN 可以不用大型全连接层

传统做法是：

```text
卷积特征图 -> Flatten -> 大型全连接层 -> 类别输出
```

NiN 的做法是：

```text
卷积特征图 -> 输出类别通道 -> 全局平均池化 -> 类别输出
```

这里的关键是最后的：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

它会把每个类别通道的整张特征图平均成一个数。

如果最后有 10 个通道，池化后就得到 10 个数。

这 10 个数就可以直接作为分类输出。

## 6. 全局平均池化应该怎么理解

全局平均池化不是在一个小窗口里池化，而是把整张特征图压成一个值。

比如某个类别通道表示“这张图像像不像鞋子”。

那么这个通道在不同空间位置上可能都有响应。

全局平均池化做的是：

```text
把所有位置的响应平均起来，得到这个类别整体上的证据强度
```

这比展平后接全连接层更直接：

```text
每个通道天然对应一个类别，每个通道平均后就是该类别的分数
```

## 7. `AdaptiveAvgPool2d((1, 1))` 的好处

普通平均池化要指定窗口大小。

而自适应平均池化只指定输出大小：

```python
nn.AdaptiveAvgPool2d((1, 1))
```

它的意思是：

```text
不管输入特征图原来多大，都把每个通道压成 1 x 1
```

这让网络对输入尺寸更灵活。

在很多现代网络里，全局平均池化都用这种方式实现。

## 8. Dropout 在 NiN 中放在哪里

NiN 里也用了：

```python
nn.Dropout(0.5)
```

它放在最后分类相关的 NiN 块之前。

这里的作用仍然是正则化：

```text
在最终类别特征生成前，减少模型对某些固定中间特征的过度依赖
```

虽然 NiN 去掉了巨大的全连接层，参数量有所减少，但深层卷积特征仍然可能过拟合，所以 Dropout 依然有价值。

## 9. NiN 和 VGG 的区别

### VGG

```text
用规则的 3 x 3 卷积块加深网络，最后仍然接大规模全连接层
```

### NiN

```text
在卷积块内部加入 1 x 1 卷积增强局部表达，并用全局平均池化替代全连接分类器
```

所以 NiN 更像是在推动 CNN 从：

```text
卷积提特征 + 全连接分类
```

走向：

```text
卷积网络自身直接完成类别特征表达
```

## 10. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. NiN 块由普通卷积加多个 `1 x 1` 卷积组成
2. `1 x 1` 卷积可以看成作用在每个位置上的小型全连接层
3. NiN 用类别通道加全局平均池化替代大型全连接层
4. `AdaptiveAvgPool2d((1, 1))` 会把每个通道压成一个值
5. NiN 的思想影响了后面很多现代 CNN 结构

如果最后只留一句话：

```text
NiN 让你看到，分类不一定非要靠巨大的全连接层，卷积网络本身也可以直接学出类别级表示
```
