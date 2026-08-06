# 38 Nadaraya-Watson 核回归学习笔记

对应文件：`Attention Mechanism/38_Nadaraya-Watson.ipynb`

这份笔记是结合你的文件 [38_Nadaraya-Watson.ipynb](</D:/Learn/d2l-test/Attention Mechanism/38_Nadaraya-Watson.ipynb>) 写的。它用一个一维函数拟合例子，把注意力机制解释成“根据距离分配权重，然后对 value 做加权平均”。

如果用一句话概括这一章：

```text
Nadaraya-Watson 核回归用 query 和 key 的相似度生成注意力权重，再对 value 加权求和得到预测
```

## 1. 这一章为什么重要

上一章讲了查询、键和值：

```text
query 用来发起查询
key 用来和 query 匹配
value 是最终被汇总的信息
```

这一章用 Nadaraya-Watson 核回归把这个思想具体画出来。

它不是先做机器翻译，也不是先做 Transformer，而是先用一个简单的一维回归问题说明：

```text
注意力机制本质上可以看成一种加权平均
```

不同的是，权重不是固定的，而是根据 query 和 key 的相似度动态计算出来的。

## 2. 先构造一个一维回归任务

notebook 中设置：

```python
n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)
```

这表示随机生成 50 个训练输入，并排序。

输入范围在：

```text
0 到 5
```

真实函数定义为：

```python
def f(x):
    return 2 * torch.sin(x) + x**0.8
```

这个函数不是一条直线，而是带有弯曲变化的非线性函数。

训练标签是：

```python
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
```

也就是在真实函数值上加了噪声。

这更接近真实数据：

```text
观测值 = 真实规律 + 随机扰动
```

测试输入是：

```python
x_test = torch.arange(0, 5, 0.1)
```

它更密集，用来画出模型预测曲线。

## 3. plot_kernel_reg 的作用

函数：

```python
def plot_kernel_reg(y_hat):
```

用来画两类东西：

1. `y_truth`：真实函数曲线
2. `y_hat`：模型预测曲线
3. `x_train, y_train`：带噪声的训练样本点

这能直观看出模型是否拟合到了真实函数走势。

深度学习和机器学习中，画图不是装饰。

在这种低维任务里，画图能非常快地帮你判断：

- 欠拟合
- 过拟合
- 曲线是否太平滑
- 曲线是否被噪声带偏

## 4. 最简单的基线：所有预测都等于平均值

notebook 先做了一个非常粗糙的预测：

```python
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
```

含义是：

```text
不管 x_test 是多少，预测值都等于训练标签的平均值
```

这相当于完全不看 query。

它的预测曲线是一条水平线。

这个方法通常会欠拟合，因为它忽略了输入 `x` 的变化。

但它很适合作为基线，让你看到后面的核回归到底改进在哪里。

## 5. 非参数 Nadaraya-Watson 核回归

接下来 notebook 使用训练输入作为 key，训练输出作为 value。

对于每个测试输入 query，它会计算和所有 key 的距离。

核心代码是：

```python
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
y_hat = torch.matmul(attention_weights, y_train)
```

这里的逻辑可以拆成两步。

### 5.1 根据距离计算权重

`X_repeat - x_train` 表示：

```text
每个测试点 query 和每个训练点 key 的距离
```

距离越小，说明 query 和 key 越接近。

代码中使用：

```text
-(距离^2) / 2
```

距离越小，这个分数越大。

再经过 softmax 后，近的训练点权重大，远的训练点权重小。

### 5.2 对 value 做加权平均

训练标签 `y_train` 就是 value。

预测值是：

```text
所有训练标签的加权平均
```

也就是：

```text
y_hat(query) = sum(attention_weight_i * y_train_i)
```

这就是注意力汇聚。

## 6. query、key、value 在这里分别是谁

在这个一维核回归例子中：

- query：每个 `x_test`
- key：每个 `x_train`
- value：每个 `y_train`

模型问的问题是：

```text
当前测试点 x_test 和哪些训练输入 x_train 更接近？
```

越接近的训练输入，其对应的训练输出 `y_train` 对预测影响越大。

这和翻译中的注意力很像：

```text
当前解码状态和哪些源词位置更相关？
```

相关位置越重要，对当前输出影响越大。

## 7. 注意力权重热力图

代码中：

```python
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

横轴是训练输入，也就是 key。

纵轴是测试输入，也就是 query。

颜色越深，表示当前 query 对某个 key 的注意力权重越大。

因为训练输入和测试输入都按从小到大排序，所以热力图通常会出现一条斜向带状区域。

这说明：

```text
小的测试输入更关注小的训练输入，大的测试输入更关注大的训练输入
```

这正符合距离相近权重更大的直觉。

## 8. torch.bmm 是什么

后面 notebook 展示：

```python
torch.bmm(X, Y)
```

`bmm` 是 batch matrix multiplication，也就是批量矩阵乘法。

普通矩阵乘法一次处理两个二维矩阵。

`bmm` 一次处理一批矩阵。

如果：

```text
X: (batch_size, n, m)
Y: (batch_size, m, p)
```

那么：

```text
torch.bmm(X, Y): (batch_size, n, p)
```

注意力中经常需要对每个样本分别做矩阵乘法，所以 `bmm` 很常用。

## 9. 用 bmm 实现加权求和

代码中：

```python
weights = torch.ones((2, 10)) * 0.1
values = torch.arange(20.0).reshape((2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

这里 `weights` 的每一行都是 10 个 0.1。

也就是对 10 个 value 做平均。

为了用 `bmm`，需要调整形状：

```text
weights.unsqueeze(1): (2, 1, 10)
values.unsqueeze(-1): (2, 10, 1)
```

相乘后得到：

```text
(2, 1, 1)
```

也就是每个样本一组加权平均结果。

这一步是后面注意力层实现的基础。

## 10. 带参数的 Nadaraya-Watson 核回归

前面的核回归没有可学习参数。

它固定使用：

```text
softmax(-(距离^2) / 2)
```

后面 notebook 定义了一个可学习模型：

```python
class NWKernelRegression(nn.Module):
```

其中有一个参数：

```python
self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
```

这个 `w` 控制距离缩放。

注意力权重变成：

```python
softmax(-((queries - keys) * self.w)**2 / 2)
```

如果 `w` 较大，距离差异会被放大，注意力更集中。

如果 `w` 较小，距离差异会被压缩，注意力更分散。

所以 `w` 控制的是核函数的宽窄。

## 11. 为什么训练时要排除自己

训练带参数核回归时，代码构造：

```python
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
```

这一步把每个训练样本自己从自己的 key-value 列表中排除掉。

原因是：

```text
如果允许一个样本直接关注自己，它只要给自己很高权重，就能轻松记住训练标签
```

这会让训练损失看起来很低，但并不代表模型学会了泛化。

排除自己以后，模型必须用邻近样本来预测当前样本。

这更符合核回归想学习的规律。

## 12. 训练过程

训练代码使用：

```python
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
```

因为这是回归任务，所以用均方误差 MSE。

训练只有 5 个 epoch。

每轮步骤是：

1. 清空梯度
2. 用当前 `w` 计算预测值
3. 计算预测和真实 `y_train` 的误差
4. 反向传播
5. 用 SGD 更新 `w`

虽然只有一个参数，但这个例子清楚展示了：

```text
注意力权重也可以由可学习参数控制
```

## 13. 用训练好的模型预测测试集

训练后，代码重新构造测试集的 key 和 value：

```python
keys = x_train.repeat((n_test, 1))
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values)
```

这一次每个测试 query 都会和所有训练 key 比较。

输出是对训练 value 的加权平均。

然后再画出预测曲线。

如果训练合理，预测曲线应该比简单平均更接近真实函数。

## 14. 学习后的注意力热力图

最后再次显示：

```python
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0), ...)
```

这次显示的是训练好的参数 `w` 所产生的注意力权重。

你可以观察：

- 权重是否集中在对角线附近
- 相近输入是否更容易互相关注
- 注意力范围是宽还是窄

如果注意力太窄，模型可能更容易跟着噪声抖动。

如果注意力太宽，模型可能过于平滑，无法捕捉函数细节。

这就是核宽度的偏差-方差权衡。

## 15. 这一章和深度学习注意力的关系

Nadaraya-Watson 核回归看起来像传统机器学习方法，但它和神经网络注意力有同一个骨架：

```text
计算 query-key 相似度 -> softmax 得到权重 -> 对 value 加权求和
```

区别在于：

- 核回归中相似度通常由距离决定
- 神经网络中相似度可以由可学习投影和评分函数决定
- Transformer 中 query、key、value 都来自可学习线性变换

所以这一章是在用最简单的回归例子铺垫后面的注意力评分函数。

## 16. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. Nadaraya-Watson 核回归是注意力汇聚的一个直观例子
2. query 是测试输入，key 是训练输入，value 是训练输出
3. 距离越近，注意力权重越大
4. 预测值是所有 value 的加权平均
5. `torch.bmm` 可以高效实现批量加权求和
6. 可学习参数 `w` 控制注意力分布的宽窄
7. 注意力热力图能直观看出模型关注了哪些 key

如果最后只留一句话：

```text
Nadaraya-Watson 核回归告诉我们：注意力并不是玄学，它就是根据相似度动态分配权重，然后汇总信息
```
