# 18 VGG 学习笔记

这份笔记是结合你的文件 [18_VGG.ipynb](</D:/Learn/d2l-test/Convolutional_Neural_Networks/18_VGG.ipynb>) 写的，目标是帮助你真正理解：

- VGG 为什么喜欢反复堆叠 `3 x 3` 卷积
- 什么是 VGG 块
- `conv_arch` 这种配置为什么能让网络结构更清晰
- VGG 和 AlexNet 的设计思想有什么不同
- 为什么训练时使用了通道数缩小版的 VGG

如果用一句话概括这一章：

```text
VGG 的核心思想是：用规则、重复的小卷积块搭出更深、更整齐的卷积网络
```

## 1. 先用人话理解 VGG 在解决什么问题

AlexNet 已经证明深层 CNN 很强，但它的结构看起来有点“手工设计”：

- 第一层是 `11 x 11`
- 第二层是 `5 x 5`
- 后面又是多个 `3 x 3`
- 每层通道数也手动变化

VGG 的思路更规整：

```text
不要每一层都单独发明结构，而是设计一种标准块，然后反复堆叠
```

这让网络有两个明显优点：

1. 结构更容易理解
2. 深度更容易扩展

所以 VGG 这章真正要学的是：

```text
如何用模块化思想设计深层卷积网络
```

## 2. 什么是 VGG 块

notebook 里先定义了：

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

一个 VGG 块包含：

1. 若干个 `3 x 3` 卷积层
2. 每个卷积后接 `ReLU`
3. 最后接一个 `2 x 2` 最大池化层

可以把它理解成：

```text
先在当前分辨率上连续提取特征，再统一把空间尺寸减半
```

这比“卷积一次就池化一次”更强，因为它允许网络在同一尺度上做多次非线性变换。

## 3. 为什么 VGG 偏爱 `3 x 3` 卷积

`3 x 3` 是一个很小的卷积核，但 VGG 大量使用它。

原因是：

### 参数更少

一个 `5 x 5` 卷积核有 25 个空间参数位置。

两个 `3 x 3` 卷积核总共有 18 个空间参数位置。

如果考虑通道数，差异会更明显。

### 非线性更多

两个 `3 x 3` 卷积之间可以插入两个 `ReLU`，而一个 `5 x 5` 卷积通常只接一个激活。

这意味着：

```text
用多个小卷积堆叠，不只是扩大感受野，还增加了非线性表达能力
```

### 结构更统一

统一使用 `3 x 3`，让网络设计变得非常整齐。

你不用每一层都纠结核大小，而是主要控制：

- 每个块里卷积层数量
- 每个块的输出通道数

## 4. `padding=1` 为什么重要

VGG 块里的卷积是：

```python
nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
```

`3 x 3` 卷积配 `padding=1`，会保持高宽不变。

所以在一个 VGG 块内部，卷积层只改变通道数，不改变空间尺寸。

真正让高宽减半的是最后的：

```python
nn.MaxPool2d(kernel_size=2, stride=2)
```

这让尺寸变化更可控：

```text
卷积负责提特征，池化负责降分辨率
```

## 5. `conv_arch` 是什么

notebook 里写了：

```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

每个元组表示一个 VGG 块：

```text
(这个块里有几个卷积层, 这个块输出多少通道)
```

比如：

- `(1, 64)`：1 个卷积层，输出 64 通道
- `(2, 256)`：2 个卷积层，输出 256 通道

这种写法的好处是：

```text
网络结构不再散落在一堆层定义里，而是变成一份清晰的架构配置
```

这就是深度学习代码里常见的“配置驱动结构”。

## 6. `vgg()` 函数在做什么

你定义了：

```python
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
```

这段代码分成两部分：

### 卷积部分

根据 `conv_arch` 一块一块生成 VGG 块。

### 分类部分

展平后接三个全连接层，最后输出 10 类。

这里和 AlexNet 类似，仍然保留了大规模全连接分类器。

## 7. 输出形状为什么最后是 `7 x 7`

输入是：

```text
1 x 224 x 224
```

VGG 有 5 个最大池化层，每个池化层都会让高宽减半。

所以空间尺寸变化大致是：

```text
224 -> 112 -> 56 -> 28 -> 14 -> 7
```

最后一个卷积块输出通道是 512，所以进入全连接层前的形状是：

```text
512 x 7 x 7
```

这就是为什么线性层写成：

```python
nn.Linear(out_channels * 7 * 7, 4096)
```

## 8. 为什么训练时用了缩小版 VGG

notebook 里写了：

```python
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

这是把每个块的通道数缩小到原来的四分之一。

原因很现实：

```text
完整 VGG 参数多、计算重，在 Fashion-MNIST 上训练成本没必要那么高
```

所以这里保留 VGG 的结构思想，但缩小通道数，方便快速实验。

这也是学习模型架构时很常见的做法：

```text
先理解结构，再根据数据和算力调整规模
```

## 9. VGG 和 AlexNet 的核心区别

### AlexNet

```text
更像一次深层 CNN 的工程突破，结构中有多种卷积核大小
```

### VGG

```text
更强调简单规则的重复：小卷积核、固定块、逐步加深
```

VGG 的美感在于：

```text
它把深层 CNN 设计变得像搭积木一样清楚
```

你只要改变 `conv_arch`，就能得到不同深度和宽度的 VGG 网络。

## 10. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. VGG 块由多个 `3 x 3` 卷积加一个最大池化组成
2. `3 x 3` 卷积配 `padding=1` 可以保持空间尺寸
3. 多个小卷积堆叠可以代替大卷积，同时增加非线性
4. `conv_arch` 让网络结构变得配置化、模块化
5. 训练时缩小通道数，是为了保留结构思想同时降低成本

如果最后只留一句话：

```text
VGG 教会你的不是某一个固定网络，而是用统一的小卷积块系统地搭建深层 CNN
```
