# 44 ImageNet Dogs 狗品种识别学习笔记

对应文件：`ImageNet Dogs/ImageNet Dogs.ipynb`

这份笔记是结合你的文件 [ImageNet Dogs.ipynb](</D:/Learn/d2l-test/ImageNet Dogs/ImageNet Dogs.ipynb>) 写的。它对应的是一个 Kaggle 风格的图像分类任务：给一张狗的图片，预测它属于 120 个狗品种中的哪一个。

如果用一句话概括这一章：

```text
ImageNet Dogs 任务的核心是用预训练 ResNet34 提取图像特征，再训练一个新的分类头完成 120 类狗品种识别
```

## 1. 这一章在学什么

前面的 CNN 章节主要是在 Fashion-MNIST、CIFAR-10 等较小数据集上训练模型。

这一章更接近真实图像竞赛。

它涉及：

1. 下载并整理 Kaggle 图像数据
2. 划分训练集、验证集和测试集
3. 对训练图像做数据增强
4. 使用 ImageNet 预训练的 ResNet34
5. 冻结预训练特征层，只训练新的分类头
6. 在验证集上观察损失
7. 使用全部训练数据重新训练
8. 生成 `submission.csv`

这章最重要的思想是迁移学习。

也就是：

```text
不要从零开始学所有图像特征，而是借用大数据集上已经学好的视觉能力
```

## 2. 为什么要用迁移学习

狗品种识别比 CIFAR-10 难得多。

CIFAR-10 只有 10 类，图片也比较小。

狗品种识别有 120 类，而且很多狗品种外形非常接近。

例如：

```text
不同梗犬、不同猎犬、不同牧羊犬之间可能只差耳朵、毛色、体型或脸部细节
```

如果从零训练一个深层 CNN，需要大量数据和算力。

但 ImageNet 上训练好的模型已经学会了很多通用视觉特征，比如：

- 边缘
- 纹理
- 颜色组合
- 局部形状
- 物体部件
- 高层语义特征

所以这章直接使用预训练 `resnet34`。

模型前面的特征提取部分保持不动，只重新训练最后的分类部分。

这就是迁移学习中常见的 feature extractor 用法。

## 3. 下载数据集

notebook 中：

```python
d2l.DATA_HUB['dog_tiny'] = (...)
demo = True
```

这里使用的是一个小型演示数据集 `dog_tiny`。

如果 `demo=True`，会下载小数据集，方便快速跑通流程。

如果要参加完整 Kaggle 比赛，需要把：

```python
demo = False
```

并把完整数据放到：

```text
../data/dog-breed-identification
```

这一点很重要。

小数据集只是用于学习代码流程，不代表最终竞赛效果。

## 4. 重组数据目录

原始 Kaggle 数据通常是：

```text
train/
test/
labels.csv
```

其中 `labels.csv` 记录每张训练图片的类别。

但 PyTorch 的 `ImageFolder` 期望目录结构是：

```text
类别名/
  图片1.jpg
  图片2.jpg
```

所以 notebook 定义：

```python
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)
```

它主要做三件事：

1. 读取 `labels.csv`
2. 按类别整理训练图片
3. 划分验证集并整理测试集

整理后会得到类似：

```text
train_valid_test/
  train/
  valid/
  train_valid/
  test/
```

其中：

- `train`：训练时使用
- `valid`：验证模型效果
- `train_valid`：最终提交前使用训练集加验证集重新训练
- `test`：没有标签，用于生成提交文件

## 5. valid_ratio 的意义

代码中：

```python
valid_ratio = 0.1
```

表示从训练数据中划出 10% 作为验证集。

验证集不参与参数更新，只用来观察模型泛化能力。

如果训练损失下降但验证损失上升，通常说明模型开始过拟合。

在竞赛流程中，验证集像是本地的小型排行榜。

它帮助我们在提交前判断模型是否真的变好。

## 6. 训练集数据增强

训练变换 `transform_train` 包含：

```python
RandomResizedCrop
RandomHorizontalFlip
ColorJitter
ToTensor
Normalize
```

这些操作的目的不是改变标签，而是让模型看到同一张图片的不同合理版本。

### 6.1 RandomResizedCrop

随机裁剪图像的一部分，再缩放到 `224 x 224`。

这样模型不会只记住狗在图片中的固定位置。

它要学会：

```text
无论狗在中间、偏左、偏右，只要关键特征存在，都应该识别出来
```

### 6.2 RandomHorizontalFlip

随机水平翻转图片。

狗朝左或朝右不应该改变品种。

这能增加训练样本的变化。

### 6.3 ColorJitter

随机改变亮度、对比度和饱和度。

真实照片可能来自不同光照环境。

这个增强能减少模型对固定颜色和光照的依赖。

### 6.4 Normalize

标准化使用：

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

这是 ImageNet 预训练模型常用的均值和标准差。

因为 ResNet34 是在 ImageNet 上预训练的，所以输入图片也应该按 ImageNet 的方式标准化。

如果这里标准化方式不匹配，预训练特征的效果会下降。

## 7. 测试集和验证集为什么不用随机增强

`transform_test` 使用：

```python
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(...)
```

验证和测试时不做随机裁剪、随机翻转、随机变色。

原因是：

```text
评估时希望结果稳定，不能同一张图每次预测都不一样
```

所以测试阶段使用确定性的中心裁剪。

这能让验证损失和提交结果更稳定。

## 8. ImageFolder 数据集

notebook 使用：

```python
torchvision.datasets.ImageFolder(...)
```

`ImageFolder` 会自动把子文件夹名当成类别名。

例如目录：

```text
train/beagle/xxx.jpg
train/pug/yyy.jpg
```

会被识别为两个类别：

```text
beagle
pug
```

最终类别顺序保存在：

```python
train_valid_ds.classes
```

生成提交文件时，也会使用这个类别顺序作为 CSV 表头。

## 9. DataLoader 的作用

数据集定义好以后，还要用 `DataLoader` 批量读取：

```python
torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
```

训练集使用 `shuffle=True`。

这可以打乱样本顺序，减少模型记住固定顺序的可能性。

`drop_last=True` 表示如果最后一个 batch 不够完整，就丢掉。

这样每个 batch 形状一致，训练更稳定。

测试集使用：

```python
drop_last=False
```

因为测试集每张图片都要预测，不能丢掉最后不足一个 batch 的图片。

## 10. get_net：构建迁移学习模型

核心函数是：

```python
def get_net(devices):
```

它构建了一个新模型：

```python
finetune_net = nn.Sequential()
finetune_net.features = torchvision.models.resnet34(pretrained=True)
finetune_net.output_new = nn.Sequential(
    nn.Linear(1000, 256),
    nn.ReLU(),
    nn.Linear(256, 120)
)
```

这里可以理解为两段：

```text
预训练 ResNet34 特征提取器 -> 新的 120 类分类头
```

ResNet34 原本输出 1000 类 ImageNet 分数。

这里没有直接使用原来的 1000 类作为最终结果，而是在后面接了一个新的小网络：

```text
1000 -> 256 -> 120
```

最终的 `120` 对应 120 个狗品种。

## 11. 为什么冻结 features

代码中：

```python
for param in finetune_net.features.parameters():
    param.requires_grad = False
```

这表示不更新 ResNet34 的预训练参数。

只训练新加的 `output_new`。

这样做有几个好处：

1. 训练更快
2. 小数据集上不容易过拟合
3. 可以保留 ImageNet 学到的通用视觉特征
4. 显存和计算压力更小

它也有一个限制：

```text
如果狗品种任务和 ImageNet 特征差异较大，只训练分类头可能不够
```

完整竞赛中，有时会解冻部分高层参数进行 fine-tuning。

但入门流程先冻结特征层更稳。

## 12. evaluate_loss：验证损失

验证函数：

```python
def evaluate_loss(data_iter, net, devices):
```

它遍历验证集，计算平均交叉熵损失。

这里使用：

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

先保留每个样本的损失，再手动求平均：

```python
l_sum += l.sum()
n += labels.numel()
return l_sum / n
```

这样可以准确按样本数量计算平均损失。

验证损失越低，说明模型对未见过图片的预测越可靠。

## 13. train：训练函数

训练函数中先使用：

```python
nn.DataParallel(net, device_ids=devices)
```

如果有多块 GPU，可以并行训练。

如果只有一块 GPU 或 CPU，也能按可用设备运行。

优化器是 SGD：

```python
torch.optim.SGD(..., lr=lr, momentum=0.9, weight_decay=wd)
```

其中：

- `lr`：学习率
- `momentum=0.9`：让参数更新带有惯性，训练更平滑
- `weight_decay=wd`：权重衰减，用来缓解过拟合

## 14. 学习率调度器

代码使用：

```python
scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
```

含义是每隔 `lr_period` 个 epoch，把学习率乘以 `lr_decay`。

当前设置：

```python
lr_period = 2
lr_decay = 0.9
```

也就是每 2 轮学习率变成原来的 90%。

训练前期学习率稍大，方便快速下降。

训练后期学习率逐渐变小，方便更细致地靠近较优解。

## 15. 训练时观察什么

训练时会画出：

```text
train loss
valid loss
```

如果二者都下降，说明模型在正常学习。

如果训练损失下降但验证损失不降，说明可能过拟合。

如果训练损失和验证损失都不降，可能是学习率、数据、模型连接或冻结策略有问题。

对于狗品种识别这种细粒度分类任务，验证损失很重要。

因为模型很容易记住训练集中某些背景或姿势，而不是真正识别品种。

## 16. 使用 train_valid 重新训练

完成验证后，notebook 又执行：

```python
net = get_net(devices)
train(net, train_valid_iter, None, ...)
```

这次不再使用验证集，而是把训练集和验证集合并成 `train_valid`。

原因是：

```text
提交前希望利用所有有标签数据来训练最终模型
```

验证集在调参阶段有用。

当参数已经确定后，就可以把它也加入训练，让最终模型看到更多数据。

## 17. 生成 submission.csv

最终模型对测试集预测：

```python
output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
```

`softmax` 会把模型输出分数变成概率。

每张测试图片都会得到 120 个类别概率。

然后写入：

```text
submission.csv
```

CSV 格式是：

```text
id,breed1,breed2,...,breed120
图片id,概率1,概率2,...,概率120
```

这正是 Kaggle 多分类提交常见格式。

注意：这里写入的不是预测类别名，而是每个类别的概率。

## 18. 这一章和 CIFAR-10 的区别

CIFAR-10 章节通常是从数据增强、训练卷积网络、生成提交文件入手。

ImageNet Dogs 这一章的重点更偏向迁移学习。

主要区别是：

1. 类别数从 10 类变成 120 类
2. 图片更真实、更复杂
3. 类别之间差异更细微
4. 使用 ImageNet 预训练模型
5. 训练时只更新新的分类头
6. 输入尺寸使用 ResNet 常见的 `224 x 224`

所以这章不是简单重复 CIFAR-10，而是进一步学习真实任务中常用的预训练模型套路。

## 19. 代码中容易踩坑的地方

### 类别顺序必须一致

提交文件表头使用：

```python
train_valid_ds.classes
```

预测概率的列顺序必须和这个类别顺序一致。

如果类别顺序错了，即使模型预测对了，提交文件也会错位。

### 测试集不能 drop_last

测试集必须：

```python
drop_last=False
```

否则最后不足一个 batch 的测试图片会丢失，提交文件行数就不完整。

### 预训练模型输入要标准化

ResNet34 预训练时使用 ImageNet 标准化。

所以这里也必须使用相同均值和标准差。

### 冻结参数后优化器只训练 requires_grad=True 的参数

代码中：

```python
param for param in net.parameters() if param.requires_grad
```

这保证优化器只更新新分类头。

## 20. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. 狗品种识别是 120 类细粒度图像分类任务
2. 小数据集上从零训练深层 CNN 很难，迁移学习更实用
3. 预训练 ResNet34 负责提取图像特征
4. 新分类头负责把特征映射到 120 个狗品种
5. 训练集使用随机增强，验证集和测试集使用确定性预处理
6. 验证阶段用 `valid` 调参，最终提交前用 `train_valid` 重新训练
7. `submission.csv` 需要写入每个测试样本属于每个类别的概率

如果最后只留一句话：

```text
ImageNet Dogs 这章真正要掌握的是迁移学习流程：整理数据，复用预训练特征，训练新分类头，再把测试集概率写成提交文件
```
