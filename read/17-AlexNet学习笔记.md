# 17 AlexNet 学习笔记

这份笔记是结合你的文件 [17_AlexNet.ipynb](</D:/Learn/d2l-test/Convolutional_Neural_Networks/17_AlexNet.ipynb>) 写的，目标是帮助你真正理解：

- AlexNet 相比 LeNet 到底强在哪里
- 为什么 AlexNet 使用更大的输入、更深的网络和更多通道
- `ReLU`、`Dropout`、最大池化在这个网络中分别承担什么角色
- 代码里每一层的形状变化为什么是这样
- 为什么训练 AlexNet 时要把 Fashion-MNIST 放大到 `224 x 224`

如果用一句话概括这一章：

```text
AlexNet 的核心意义是：把卷积神经网络从“小模型能跑”推进到“深层大模型能真正解决复杂视觉任务”
```

## 1. 先用人话理解 AlexNet 为什么重要

前面你已经学过 LeNet。

LeNet 的结构比较小，适合处理 MNIST 这种简单手写数字图像。它的整体思路是：

```text
卷积 -> 池化 -> 卷积 -> 池化 -> 全连接分类
```

AlexNet 仍然沿用了这个基本框架，但它把模型规模明显放大了：

- 输入图片更大
- 卷积通道更多
- 网络层数更深
- 全连接层更宽
- 使用了 `ReLU`
- 使用了 `Dropout`

所以 AlexNet 可以理解成：

```text
LeNet 思想的放大版，也是现代深度卷积网络的起点之一
```

它真正重要的地方不只是“层数多”，而是说明了：

```text
只要数据、算力、训练技巧跟得上，深层卷积网络可以学到非常强的图像特征
```

## 2. 你的 AlexNet 代码整体长什么样

notebook 里定义了：

```python
net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10))
```

这个网络可以分成两大部分：

### 卷积特征提取部分

```text
Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Conv -> Conv -> Conv -> Pool
```

它负责从图像中提取越来越抽象的视觉特征。

### 全连接分类部分

```text
Flatten -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
```

它负责把前面提取出来的特征转换成最终的 10 类分类结果。

## 3. 第一层为什么使用很大的 `11 x 11` 卷积核

AlexNet 第一层是：

```python
nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1)
```

这和 LeNet 里常见的 `5 x 5` 卷积不同，它一上来就使用了比较大的卷积核和比较大的步幅。

可以这样理解：

```text
输入图像很大时，第一层可以先用大窗口快速捕捉粗粒度视觉模式，并迅速降低空间尺寸
```

这里的几个参数分别表示：

- `1`：输入是单通道灰度图
- `96`：输出 96 个通道，也就是学 96 种初级特征
- `kernel_size=11`：每次看一个 `11 x 11` 的局部区域
- `stride=4`：每次移动 4 格，快速下采样
- `padding=1`：边缘补一点空间

所以第一层的作用不是细细扫描，而是：

```text
快速从大图中提取大量低层特征，并把特征图尺寸压下来
```

## 4. 为什么 AlexNet 大量使用 `ReLU`

LeNet 里用的是 `Sigmoid`，而 AlexNet 使用的是 `ReLU`：

```python
nn.ReLU()
```

`ReLU` 的形式很简单：

```text
小于 0 的变成 0，大于 0 的保持原样
```

它相对 `Sigmoid` 的好处是：

- 计算更简单
- 正区间梯度不容易消失
- 深层网络训练更快

在深层网络里，激活函数非常关键。

如果每一层都用容易饱和的 `Sigmoid`，梯度在反向传播中可能越来越小，导致前面的层学得很慢。

AlexNet 使用 `ReLU`，可以理解成是在告诉你：

```text
深层网络不仅要有更多层，还要配合更适合深层训练的激活函数
```

## 5. 最大池化在 AlexNet 里做了什么

AlexNet 使用了多次：

```python
nn.MaxPool2d(kernel_size=3, stride=2)
```

最大池化会在局部窗口里保留最大响应。

在 AlexNet 里，它主要有三个作用：

1. 缩小特征图，降低计算量
2. 保留局部最显著特征
3. 增强一定的局部平移不敏感性

你可以把它理解成：

```text
卷积层负责找特征，最大池化负责把最强的特征响应留下来
```

## 6. 为什么中间连续堆了 3 个卷积层

网络中间有：

```python
nn.Conv2d(256, 384, kernel_size=3, padding=1)
nn.Conv2d(384, 384, kernel_size=3, padding=1)
nn.Conv2d(384, 256, kernel_size=3, padding=1)
```

这三层没有立刻接池化，而是连续做卷积。

这说明 AlexNet 不只是简单重复“卷积一次池化一次”，而是在较高层特征图上继续加深非线性变换。

连续卷积的意义是：

```text
让网络在已经压缩过的空间特征上，组合出更复杂的视觉模式
```

前面的卷积可能更像在看边缘和纹理，后面的卷积则可能开始组合局部形状和更抽象的部件。

## 7. 为什么全连接层这么大

AlexNet 后面用了两个很宽的全连接层：

```python
nn.Linear(6400, 4096)
nn.Linear(4096, 4096)
```

这说明早期 CNN 仍然非常依赖大规模全连接层来做最终分类。

它的好处是表达能力强，但缺点也很明显：

- 参数量巨大
- 容易过拟合
- 训练和存储成本更高

所以 AlexNet 后面紧跟着使用了：

```python
nn.Dropout(p=0.5)
```

这是为了缓解大模型带来的过拟合风险。

## 8. Dropout 在这里为什么重要

在两个大全连接层后面加 Dropout，本质上是在做正则化。

它的含义是：

```text
训练时随机让一部分神经元暂时失效，减少模型对某些固定路径的依赖
```

AlexNet 参数量很大，如果不加控制，容易把训练数据记得太死。

所以这里的 Dropout 可以理解成：

```text
给很宽的全连接层加一个防过拟合刹车
```

## 9. 为什么 Fashion-MNIST 要 resize 到 224

notebook 里读取数据时写了：

```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

Fashion-MNIST 原图是 `28 x 28`，但 AlexNet 原本面向的是更大的图像输入。

如果直接把 `28 x 28` 喂给这个网络，前面的大卷积核和多次池化会让空间尺寸很快被压没。

所以这里把图片放大到 `224 x 224`，目的是：

```text
让输入尺寸适配 AlexNet 的结构设计
```

这不是因为 Fashion-MNIST 本身需要这么大，而是为了复现 AlexNet 这种大输入网络的形状流程。

## 10. 打印输出形状这一步在帮你看什么

代码里写了：

```python
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

这一步非常重要，因为 AlexNet 比 LeNet 深很多，光看定义很容易迷路。

打印每层输出形状可以帮助你检查：

- 通道数如何变化
- 高宽如何逐步缩小
- `Flatten` 之前是不是正好得到 `6400` 个特征
- 全连接层输入维度是否匹配

学习 CNN 时，一个很实用的习惯就是：

```text
每写完一个网络，先用假输入跑一遍形状
```

这能快速发现很多维度错误。

## 11. AlexNet 和 LeNet 的核心区别

可以这样对比：

### LeNet

```text
小输入、小通道、浅层网络、Sigmoid、平均池化
```

### AlexNet

```text
大输入、大通道、更深网络、ReLU、最大池化、Dropout
```

所以 AlexNet 不是简单“多堆几层”，而是多个训练和结构技巧一起发生变化：

- 更大的模型容量
- 更适合深层网络的激活函数
- 更强的正则化
- 更强的特征提取能力

## 12. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. AlexNet 是比 LeNet 大很多的深层 CNN
2. `ReLU` 让深层网络训练更容易
3. 最大池化用于保留强响应并压缩空间尺寸
4. 大全连接层带来强表达能力，也带来过拟合风险
5. Dropout 是 AlexNet 中缓解过拟合的重要手段
6. `resize=224` 是为了让 Fashion-MNIST 适配 AlexNet 的输入结构

如果最后只留一句话：

```text
AlexNet 让你看到，卷积神经网络从经典小模型走向深层大模型时，结构规模、激活函数和正则化必须一起升级
```
