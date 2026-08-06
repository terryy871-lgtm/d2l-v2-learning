# 46 CIFAR-10 图像分类学习笔记

对应文件：`CIFAR-10/CIFAR-10.ipynb`

这份笔记是结合你的文件 [CIFAR-10.ipynb](</D:/Learn/d2l-test/CIFAR-10/CIFAR-10.ipynb>) 写的，目标是帮助你真正理解：

- CIFAR-10 和前面 Fashion-MNIST 有什么不同
- Kaggle 图像分类数据为什么要先整理目录结构
- `train`、`valid`、`train_valid`、`test` 分别有什么作用
- 数据增强为什么只用于训练集
- 为什么这里使用 ResNet18
- 最后 `submission.csv` 是怎么生成的

如果用一句话概括这一章：

```text
CIFAR-10 这一章的核心不是发明新网络，而是把真实图像竞赛的完整流程串起来：整理数据、增强图像、训练模型、验证效果、生成提交文件
```

## 1. 先用人话理解这份 notebook 在做什么

前面你已经学过很多图像分类模型，比如：

- LeNet
- AlexNet
- VGG
- NiN
- GoogLeNet
- ResNet
- DenseNet

那些章节更偏向理解网络结构。

而 CIFAR-10 这一章更像一次完整实战：

```text
拿到一堆图片和标签文件，把它们整理成 PyTorch 能读取的格式，然后训练模型并输出 Kaggle 提交文件
```

所以这章重点不是某一个新层，而是完整工程流程。

## 2. CIFAR-10 是什么

CIFAR-10 是一个经典彩色图像分类数据集。

它有 10 个类别：

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

每张图片是彩色图，所以输入通道数是 3。

这和 Fashion-MNIST 不一样。

Fashion-MNIST 是灰度图，通常输入通道数是 1。

所以 CIFAR-10 更接近真实图像任务。

## 3. 为什么先下载 tiny 数据集

notebook 中：

```python
demo = True
```

如果 `demo=True`，就使用小型数据集：

```python
kaggle_cifar10_tiny.zip
```

这样训练更快，适合学习流程。

如果使用完整 Kaggle 数据集，可以把：

```python
demo = False
```

然后把数据放到对应目录。

这里的重点是：

```text
小数据集用于跑通流程，完整数据集用于正式竞赛训练
```

## 4. `trainLabels.csv` 在做什么

原始训练图片通常是：

```text
1.png
2.png
3.png
...
```

图片文件名本身不包含类别。

类别信息放在：

```text
trainLabels.csv
```

所以代码中定义了：

```python
def read_csv_labels(fname):
    ...
    return dict(((name, label) for name, label in tokens))
```

它会得到一个字典：

```text
图片 id -> 类别名
```

比如：

```text
"1" -> "frog"
```

这一步是后面整理目录的基础。

## 5. 为什么要重组目录

PyTorch 的：

```python
torchvision.datasets.ImageFolder
```

要求数据目录长成这样：

```text
root/
  cat/
    xxx.png
  dog/
    yyy.png
  truck/
    zzz.png
```

也就是说：

```text
文件夹名就是类别名
```

但 Kaggle 原始数据通常是所有图片混在一个目录里，标签在 CSV 文件中。

所以我们要把图片复制到按类别划分的文件夹中。

这就是 `reorg_train_valid` 的作用。

## 6. `copyfile` 函数做了什么

代码中：

```python
def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
```

它做两件事：

1. 如果目标目录不存在，就创建目录
2. 把图片复制过去

这一步看起来简单，但很重要。

因为后面 `ImageFolder` 能否读取数据，完全依赖目录结构是否整理正确。

## 7. 训练集、验证集、完整训练集分别是什么

重组后会生成：

```text
train_valid_test/
  train/
  valid/
  train_valid/
  test/
```

它们的含义不同。

### `train`

用于训练模型。

这是从原始训练集中扣除验证集之后剩下的部分。

### `valid`

用于验证模型。

它帮助我们观察模型在没参与训练的数据上表现如何。

### `train_valid`

包含全部训练数据。

当我们已经确定超参数后，会用它重新训练最终模型。

### `test`

没有真实标签。

它用于最终预测并生成提交文件。

## 8. 为什么验证集要按类别均匀划分

代码中：

```python
n = collections.Counter(labels.values()).most_common()[-1][1]
n_valid_per_label = max(1, math.floor(n * valid_ratio))
```

这里先找出样本最少的类别数量，再按比例决定每个类别拿多少张做验证。

这样做是为了避免验证集类别严重不平衡。

如果某些类别验证样本太多，某些类别太少，验证准确率就不够可靠。

所以这里的思路是：

```text
每个类别都拿出相同数量的验证样本
```

## 9. 为什么测试集放进 `unknown` 文件夹

测试集没有标签。

但 `ImageFolder` 仍然要求图片必须在某个类别文件夹下。

所以代码中：

```python
train_valid_test/test/unknown
```

把所有测试图片都放进 `unknown`。

这不是说它们真实类别是 unknown。

只是为了让 `ImageFolder` 能顺利读取。

最终预测时，我们只关心图片和模型输出，不关心这个临时文件夹名。

## 10. 训练集数据增强做了什么

训练集变换是：

```python
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(...)
])
```

这里有几步。

### Resize(40)

先把图片放大到 `40 x 40`。

### RandomResizedCrop(32)

随机裁剪出一个区域，再缩放回 `32 x 32`。

这让模型看到同一张图片的不同局部变化。

### RandomHorizontalFlip

随机水平翻转。

这对很多自然图像类别有帮助，比如汽车、船、马左右翻转后类别不变。

### ToTensor

把 PIL 图片转成 PyTorch 张量。

### Normalize

按 CIFAR-10 的通道均值和标准差做标准化。

这能让输入分布更稳定。

## 11. 为什么验证集和测试集不做随机增强

验证集和测试集只做：

```python
ToTensor()
Normalize(...)
```

它们不做随机裁剪、随机翻转。

原因是：

```text
验证和测试应该稳定评估模型，而不是每次输入都随机变化
```

训练时加随机性是为了增强泛化能力。

验证和测试时保持确定性，是为了让评估结果可比较。

## 12. `ImageFolder` 如何读取数据

代码中：

```python
torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train
)
```

`ImageFolder` 会自动：

1. 扫描类别文件夹
2. 给每个类别分配编号
3. 读取图片路径
4. 加载图片并应用 transform

这就是为什么前面要把目录整理成类别文件夹。

## 13. DataLoader 的几个参数怎么看

训练集：

```python
shuffle=True
drop_last=True
```

表示：

- 每个 epoch 打乱训练样本
- 丢掉最后不足一个 batch 的样本

验证集：

```python
shuffle=False
drop_last=True
```

测试集：

```python
shuffle=False
drop_last=False
```

测试集不能丢样本。

因为每一张测试图片都要生成预测结果。

## 14. 为什么使用 ResNet18

代码中：

```python
net = d2l.resnet18(num_classes, 3)
```

这里 `num_classes=10`，输入通道数是 `3`。

ResNet18 适合 CIFAR-10 的原因是：

- 比 LeNet 更强
- 比非常深的网络更轻
- 残差连接让训练更稳定
- 对彩色图像分类表现不错

这章不是重新讲 ResNet 结构，而是把 ResNet 用在真实图像分类任务上。

## 15. 损失函数为什么用交叉熵

代码中：

```python
loss = nn.CrossEntropyLoss(reduction="none")
```

CIFAR-10 是 10 类分类任务。

模型输出的是每个类别的分数。

交叉熵适合这种多分类问题。

`reduction="none"` 表示先保留每个样本的损失，后面的训练函数再自己汇总。

## 16. 训练函数做了哪些事

`train` 函数里有几个关键点：

### SGD + momentum

```python
torch.optim.SGD(..., momentum=0.9, weight_decay=wd)
```

动量可以让优化方向更稳定。

权重衰减用于正则化，缓解过拟合。

### 学习率调度

```python
torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
```

每隔一段 epoch 降低学习率。

这可以让训练前期走得快，后期收敛更细。

### 多 GPU 支持

```python
nn.DataParallel(net, device_ids=devices)
```

如果有多个 GPU，可以并行训练。

如果只有一个 GPU 或 CPU，也能按当前设备运行。

### 验证集评估

如果传入 `valid_iter`，每个 epoch 后会计算验证准确率。

这样可以观察模型是否过拟合。

## 17. 为什么先用 train + valid 分开训练

代码先做：

```python
train(net, train_iter, valid_iter, ...)
```

这是为了评估当前超参数是否合适。

比如：

- 学习率是否太大
- epoch 是否够
- weight decay 是否合适
- 数据增强是否有效

如果验证集准确率不好，就要调参。

这个阶段的目标是：

```text
用验证集判断训练方案是否靠谱
```

## 18. 为什么最后用 train_valid 重新训练

确定训练方案后，代码又做：

```python
train(net, train_valid_iter, None, ...)
```

这次不再留验证集，而是使用全部有标签训练数据。

原因是：

```text
最终提交前，希望模型利用所有带标签数据学习
```

验证集已经完成了选择超参数的任务。

最终模型可以把验证集也纳入训练。

## 19. `submission.csv` 是怎么生成的

预测测试集时：

```python
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
```

这里做的是：

1. 对测试图片前向预测
2. 取分数最高的类别编号
3. 转回 CPU 和 numpy
4. 收集所有预测

然后：

```python
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
```

把类别编号转成类别名，保存为 CSV。

这就是 Kaggle 提交文件。

## 20. 为什么要排序测试图片 id

代码中：

```python
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
```

这是为了让输出顺序和测试图片读取顺序对应。

Kaggle 提交文件通常要求：

```text
id,label
1,cat
2,ship
...
```

如果顺序错了，即使模型预测本身不错，提交结果也会乱。

所以生成提交文件时，样本顺序非常重要。

## 21. 这份 notebook 的完整流程

可以把它压缩成下面这条线：

```text
下载数据 -> 读取标签 -> 重组目录 -> 数据增强 -> 构建 DataLoader -> 训练 ResNet18 -> 验证调参 -> 全量训练 -> 预测测试集 -> 生成 submission.csv
```

这就是一次完整图像分类竞赛的基本流程。

## 22. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. CIFAR-10 是 10 类彩色图像分类任务
2. Kaggle 原始图片需要按类别重组目录，才能用 `ImageFolder`
3. 验证集用于调参，最终提交前可用 `train_valid` 全量训练
4. 训练集使用随机增强，验证集和测试集保持确定性
5. ResNet18 是这里的分类主干网络
6. `submission.csv` 需要按测试 id 顺序写出类别名

如果最后只留一句话：

```text
CIFAR-10 这一章真正训练的是你的完整项目能力：不只是搭模型，还要把数据、验证、训练和提交结果全部串成闭环
```
