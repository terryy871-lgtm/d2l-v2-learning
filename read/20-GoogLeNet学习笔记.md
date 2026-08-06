# 20 GoogLeNet 学习笔记

这份笔记是结合你的文件 [20_GoogLeNet.ipynb](</D:/Learn/d2l-test/Convolutional_Neural_Networks/20_GoogLeNet.ipynb>) 写的，目标是帮助你真正理解：

- Inception 块为什么要设计多条并行路径
- `1 x 1` 卷积在 GoogLeNet 中为什么这么关键
- GoogLeNet 如何在增加网络宽度的同时控制计算量
- `b1` 到 `b5` 每个阶段在做什么
- 为什么最后使用自适应平均池化而不是大型全连接层

如果用一句话概括这一章：

```text
GoogLeNet 的核心思想是：让网络在同一层同时用多种尺度观察图像，再把这些特征在通道维度上拼起来
```

## 1. 先用人话理解 GoogLeNet 想解决什么

前面的网络大多是在一条主线上不断堆层：

```text
输入 -> 卷积 -> 卷积 -> 池化 -> 卷积 -> 分类
```

GoogLeNet 的想法更大胆：

```text
同一层里，能不能同时用不同大小的卷积核看图像？
```

因为图像里的模式有大有小：

- 小卷积适合看局部细节
- 大卷积适合看更大范围结构
- 池化适合提取稳定的局部概括

GoogLeNet 的 Inception 块就是把这些操作并排放在一起。

## 2. Inception 块长什么样

notebook 里定义了：

```python
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
```

它有四条路径：

1. `1 x 1` 卷积路径
2. `1 x 1` 卷积后接 `3 x 3` 卷积
3. `1 x 1` 卷积后接 `5 x 5` 卷积
4. `3 x 3` 最大池化后接 `1 x 1` 卷积

最后在 `forward` 里：

```python
return torch.cat((p1, p2, p3, p4), dim=1)
```

也就是把四条路径的结果在通道维度上拼接起来。

## 3. 为什么要多条路径

多条路径的本质是：

```text
让同一层同时学习不同感受野下的特征
```

比如：

- `1 x 1` 关注通道组合
- `3 x 3` 关注较小局部邻域
- `5 x 5` 关注更大局部邻域
- 池化路径提供更稳定的局部汇总

这样一来，网络不需要提前决定“这一层到底该用多大的卷积核”。

它可以把多种尺度的结果都算出来，再交给后续网络选择和组合。

这就是 Inception 结构的直觉：

```text
与其猜一个最佳卷积核大小，不如并行尝试多种尺度
```

## 4. `1 x 1` 卷积为什么是关键

如果直接对高通道输入做 `3 x 3` 和 `5 x 5` 卷积，计算量会很大。

GoogLeNet 的做法是在大卷积前先用 `1 x 1` 卷积降维：

```python
self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
```

这相当于：

```text
先把通道数压小，再做更贵的空间卷积
```

所以 `1 x 1` 卷积在这里有两个作用：

1. 融合通道特征
2. 控制后续卷积的计算量

没有 `1 x 1` 降维，Inception 块会非常昂贵。

## 5. 为什么四条路径输出可以拼接

四条路径虽然计算方式不同，但都通过 padding 和 stride 保持了相同的高宽。

比如：

- `3 x 3` 卷积用 `padding=1`
- `5 x 5` 卷积用 `padding=2`
- 池化用 `padding=1` 且 `stride=1`

所以它们输出的空间尺寸一致。

既然高宽一样，就可以在通道维度拼接：

```text
(batch, c1, h, w) + (batch, c2, h, w) + ... -> (batch, c_total, h, w)
```

这说明 Inception 块不是把结果相加，而是：

```text
保留每条路径学到的特征，并把它们作为不同通道交给后面
```

## 6. `b1` 到 `b5` 是怎么组织网络的

notebook 把 GoogLeNet 拆成了几个阶段：

### `b1`

```text
大卷积 + ReLU + 最大池化
```

这一段快速降低输入尺寸，提取初级特征。

### `b2`

```text
1 x 1 卷积 + 3 x 3 卷积 + 最大池化
```

这一段进一步扩展通道，从 64 到 192。

### `b3`

```text
两个 Inception 块 + 最大池化
```

开始正式使用多尺度特征提取。

### `b4`

```text
五个 Inception 块 + 最大池化
```

这是网络的主体部分，反复进行多尺度特征组合。

### `b5`

```text
两个 Inception 块 + 自适应平均池化 + Flatten
```

把高层特征压缩成分类向量。

最后接：

```python
nn.Linear(1024, 10)
```

输出 10 个类别。

## 7. 为什么这里输入 resize 到 96

notebook 里训练时用了：

```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
```

GoogLeNet 原本常用于更大的图像，但在这个教学 notebook 里，为了控制训练成本，把输入缩到 `96 x 96`。

这说明一个很重要的实践点：

```text
学习网络结构时，可以保留核心模块，但调整输入尺寸和通道规模来适配当前数据与算力
```

## 8. GoogLeNet 和 NiN 的关系

GoogLeNet 和 NiN 都大量使用 `1 x 1` 卷积。

但它们的侧重点不同：

### NiN

```text
用 1 x 1 卷积增强每个位置上的通道非线性表达
```

### GoogLeNet

```text
用 1 x 1 卷积做通道融合和降维，让多尺度并行结构可计算
```

所以同一个工具在不同架构中承担的重点会不同。

## 9. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. Inception 块有多条并行路径
2. 不同路径对应不同尺度的特征提取方式
3. `1 x 1` 卷积用于通道融合和降维
4. 多条路径输出在通道维度拼接
5. GoogLeNet 用模块化的 Inception 块搭出了又宽又深的网络

如果最后只留一句话：

```text
GoogLeNet 让网络不再只沿一条路往深处走，而是在每一层并行探索多种尺度的视觉特征
```
