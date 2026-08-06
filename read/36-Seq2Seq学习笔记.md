# 36 Seq2Seq 学习笔记

对应文件：`Recurrent_Neural_Network/36_seq2seq.ipynb`

这份笔记是结合你的文件 [36_seq2seq.ipynb](</D:/Learn/d2l-test/Recurrent_Neural_Network/36_seq2seq.ipynb>) 写的，目标是把机器翻译数据集、编码器-解码器架构和真正可训练的翻译模型连起来。

如果用一句话概括这一章：

```text
Seq2Seq 是用一个编码器读入源语言序列，再用一个解码器一步步生成目标语言序列的模型
```

## 1. 这一章在学什么

前面已经完成了两件事：

1. 机器翻译数据集章节准备好了英文、法文句子，并把它们转成词元编号
2. 编码器-解码器章节定义好了 `Encoder`、`Decoder`、`EncoderDecoder` 的抽象骨架

这一章真正开始把模型写出来。

它实现的是一个基于 GRU 的 Seq2Seq 翻译模型。

输入类似：

```text
go .
```

输出类似：

```text
va !
```

模型要学习的不是简单分类，而是：

```text
一个长度可变的源序列 -> 一个长度可变的目标序列
```

这也是 Seq2Seq 名字的来源：sequence to sequence。

## 2. 导入依赖

notebook 开头导入：

```python
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
```

这些库分别用于：

- `collections`：在 BLEU 中统计 n-gram 出现次数
- `math`：计算 BLEU 中的指数和幂
- `torch`：张量计算
- `nn`：构建神经网络层
- `d2l`：使用教材封装的数据加载、训练绘图和设备选择函数

这章的代码比较完整，会涉及模型定义、损失函数、训练、预测和评价。

## 3. Seq2SeqEncoder：编码器

编码器类继承自 `d2l.Encoder`：

```python
class Seq2SeqEncoder(d2l.Encoder):
```

它的作用是读取源语言句子。

例如英文句子：

```text
i lost .
```

经过词表后会变成编号序列：

```text
[词元1, 词元2, 词元3, <eos>, <pad>, ...]
```

编码器要把这些离散编号变成连续向量，并交给 GRU 处理。

### 3.1 嵌入层

代码里先定义嵌入层：

```python
self.embedding = nn.Embedding(vocab_size, embed_size)
```

它的作用是：

```text
词元编号 -> 词向量
```

如果输入形状是：

```text
(batch_size, num_steps)
```

经过嵌入层后形状变为：

```text
(batch_size, num_steps, embed_size)
```

也就是每个词元都被表示成一个长度为 `embed_size` 的向量。

### 3.2 GRU 层

编码器使用 GRU：

```python
self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)
```

GRU 会沿着时间步读取句子。

它能把前面词元的信息保存在隐藏状态里，因此适合处理序列。

### 3.3 为什么要 permute

PyTorch 的 RNN 默认输入形状是：

```text
(num_steps, batch_size, embed_size)
```

但数据加载出来通常是：

```text
(batch_size, num_steps)
```

所以嵌入后要执行：

```python
X = X.permute(1, 0, 2)
```

这一步是把批量维和时间步维交换：

```text
(batch_size, num_steps, embed_size)
-> (num_steps, batch_size, embed_size)
```

### 3.4 编码器输出什么

编码器返回：

```python
return output, state
```

其中：

- `output`：每个时间步的隐藏状态，形状是 `(num_steps, batch_size, num_hiddens)`
- `state`：最后时间步的多层隐藏状态，形状是 `(num_layers, batch_size, num_hiddens)`

在这个基础 Seq2Seq 模型中，最关键的是 `state`，因为它会被用来初始化解码器。

## 4. 编码器形状测试

notebook 构造了一个小编码器：

```python
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
```

然后输入：

```python
X = torch.zeros((4, 7), dtype=torch.long)
```

含义是：

```text
batch_size = 4
num_steps = 7
```

输出形状应为：

```text
output: (7, 4, 16)
state:  (2, 4, 16)
```

这个测试不是为了训练，而是为了确认编码器的维度流动正确。

深度学习代码里，形状检查非常重要。很多错误不是公式错，而是维度没对齐。

## 5. Seq2SeqDecoder：解码器

解码器继承自 `d2l.Decoder`：

```python
class Seq2SeqDecoder(d2l.Decoder):
```

它的作用是根据编码器提供的上下文，生成目标语言。

例如输入英文：

```text
i lost .
```

解码器要生成：

```text
j'ai perdu .
```

### 5.1 解码器也需要嵌入层

目标语言词元同样是编号，所以也需要：

```python
self.embedding = nn.Embedding(vocab_size, embed_size)
```

这里的 `vocab_size` 是目标语言词表大小。

编码器的词表是源语言词表，解码器的词表是目标语言词表，二者通常不同。

### 5.2 解码器 GRU 的输入为什么更宽

解码器的 GRU 定义为：

```python
self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
```

注意输入大小不是 `embed_size`，而是：

```text
embed_size + num_hiddens
```

这是因为解码器每一步不只看当前目标词向量，还会拼接编码器给出的上下文 `context`。

也就是：

```text
当前目标词向量 + 源句子的上下文表示
```

这样解码器生成每个词时，都能参考源语言句子的信息。

### 5.3 init_state 的作用

解码器有：

```python
def init_state(self, enc_outputs, *args):
    return enc_outputs[1]
```

这里直接把编码器最后的隐藏状态 `state` 作为解码器初始状态。

直观理解：

```text
编码器读完整个源句子后，把“读懂的结果”交给解码器作为起点
```

这就是基础 Seq2Seq 的核心连接方式。

### 5.4 context 是什么

在 `forward` 中：

```python
context = state[-1].repeat(X.shape[0], 1, 1)
```

`state[-1]` 取的是最后一层 GRU 的隐藏状态。

它可以看作源句子的整体语义表示。

因为解码器要生成多个时间步，所以用 `repeat` 把它复制到每个时间步：

```text
(batch_size, num_hiddens)
-> (num_steps, batch_size, num_hiddens)
```

然后和目标词嵌入拼接：

```python
X_and_context = torch.cat((X, context), 2)
```

这一步是把目标词信息和源句子信息合在一起。

### 5.5 输出层

GRU 输出后，还需要映射到目标语言词表：

```python
self.dense = nn.Linear(num_hiddens, vocab_size)
```

最终输出的每个位置都是一个长度为 `vocab_size` 的向量。

它表示：

```text
当前位置预测为目标词表中每个词的分数
```

训练时会把这些分数送入交叉熵损失。

## 6. Teacher Forcing：训练时为什么喂真实目标词

训练 Seq2Seq 时，解码器输入不是模型自己上一步预测出来的词，而是目标句子右移后的真实词元。

代码中：

```python
bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
dec_input = torch.cat([bos, Y[:, :-1]], 1)
```

假设目标句子是：

```text
je suis chez moi <eos>
```

那么解码器输入是：

```text
<bos> je suis chez moi
```

标签是：

```text
je suis chez moi <eos>
```

这样模型每一步都在学习：

```text
给定前面的正确词，下一个词应该是什么
```

这种训练方式叫 teacher forcing。

它能让训练更稳定，因为早期模型预测很差，如果总把错误预测喂回去，后面的时间步会越错越远。

## 7. sequence_mask：屏蔽 padding

机器翻译数据会把句子补齐到固定长度。

短句子后面会有 `<pad>`。

例如：

```text
va ! <eos> <pad> <pad> <pad>
```

这些 `<pad>` 只是为了凑形状，不是真正要学习的词。

所以计算损失时要忽略它们。

`sequence_mask` 的作用就是把无效位置盖掉：

```python
mask = torch.arange(maxlen)[None, :] < valid_len[:, None]
```

如果有效长度是：

```text
[1, 2]
```

那么 mask 大致表示：

```text
第一行只保留第 1 个位置
第二行保留前 2 个位置
```

无效位置会被替换成指定值，比如 `0` 或 `-1`。

## 8. MaskedSoftmaxCELoss：带遮蔽的交叉熵

普通交叉熵会把所有时间步都算进去。

但翻译任务中，`<pad>` 位置不应该参与损失。

所以 notebook 自定义了：

```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
```

核心逻辑是：

```python
weights = torch.ones_like(label)
weights = sequence_mask(weights, valid_len)
```

`weights` 中真实词元位置为 `1`，padding 位置为 `0`。

然后：

```python
weighted_loss = (unweighted_loss * weights).mean(dim=1)
```

这样 padding 位置的损失会乘以 `0`，不会影响模型更新。

这一点非常重要。

如果不 mask，模型会花很多精力学习预测 `<pad>`，翻译质量会变差。

## 9. train_seq2seq：训练流程

训练函数 `train_seq2seq` 完成了完整训练循环。

### 9.1 初始化权重

代码使用 Xavier 初始化：

```python
nn.init.xavier_uniform_(m.weight)
```

它适合线性层和循环网络权重，能让训练初期的信号更稳定。

### 9.2 优化器和损失函数

使用：

```python
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = MaskedSoftmaxCELoss()
```

Adam 对学习率不太敏感，适合这个入门实现。

损失函数使用带 mask 的版本，是因为句子里有 padding。

### 9.3 每个 batch 的训练步骤

每个小批量中，数据包括：

```text
X, X_valid_len, Y, Y_valid_len
```

含义是：

- `X`：源语言输入
- `X_valid_len`：源语言有效长度
- `Y`：目标语言标签
- `Y_valid_len`：目标语言有效长度

训练步骤是：

1. 清空梯度
2. 构造解码器输入 `dec_input`
3. 前向传播得到预测 `Y_hat`
4. 计算 masked loss
5. 反向传播
6. 梯度裁剪
7. 更新参数

其中梯度裁剪：

```python
d2l.grad_clipping(net, 1)
```

是为了避免 RNN 训练中常见的梯度爆炸。

## 10. 超参数和训练模型

notebook 使用：

```python
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs = 0.005, 300
```

这些参数都比较小，适合教学演示。

其中：

- `embed_size=32`：每个词向量长度为 32
- `num_hiddens=32`：GRU 隐藏状态长度为 32
- `num_layers=2`：使用两层 GRU
- `dropout=0.1`：在多层 RNN 中做轻微正则化
- `num_steps=10`：每个句子最多保留 10 个词元
- `num_epochs=300`：小数据集上训练较多轮

然后加载机器翻译数据：

```python
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
```

再组合模型：

```python
net = d2l.EncoderDecoder(encoder, decoder)
```

这正好用上了前一章的编码器-解码器架构。

## 11. predict_seq2seq：如何翻译一句话

训练完成后，预测函数会把英文句子翻译成法文。

预测流程和训练不同。

训练时使用 teacher forcing，会把真实目标词喂给解码器。

预测时没有真实目标词，所以只能：

```text
把上一步预测出的词，作为下一步的输入
```

流程是：

1. 把源句子分词并加 `<eos>`
2. 截断或填充到 `num_steps`
3. 编码器处理源句子
4. 用编码器输出初始化解码器状态
5. 解码器从 `<bos>` 开始生成
6. 每一步选分数最高的词
7. 如果预测到 `<eos>` 就停止

这是一种贪心搜索。

它每一步都选当前最可能的词，不会同时保留多个候选翻译。

后面更高级的机器翻译会使用 beam search，但这章先用最简单直接的方法。

## 12. BLEU：评价翻译质量

BLEU 用来评价机器翻译结果。

它不是只看单词是否完全相同，而是比较预测句子和参考句子的 n-gram 重合。

例如参考译文是：

```text
j'ai perdu .
```

预测译文如果也是：

```text
j'ai perdu .
```

BLEU 会很高。

如果预测成：

```text
je perdu .
```

虽然有部分词重合，但短语结构不完全一致，分数会下降。

代码中：

```python
score = math.exp(min(0, 1 - len_label / len_pred))
```

这是短句惩罚。

如果预测译文太短，只覆盖了参考译文的一小部分，就会被扣分。

后面的循环会统计 1-gram、2-gram 等短语匹配。

`k=2` 表示最多比较到 2-gram。

## 13. 例子翻译

notebook 最后测试：

```python
engs = ['go .', "i lost .", "he's calm .", "i'm home ."]
```

这些都是短句。

模型会输出翻译结果和 BLEU 分数：

```text
go . => va !, bleu ...
```

这个结果用来检查模型是否真的学会了简单翻译。

如果 BLEU 很低，可能说明训练不充分、模型太小、学习率不合适，或者数据处理出了问题。

## 14. 这一章容易混淆的点

### 编码器 output 和 state 的区别

`output` 是所有时间步的隐藏状态。

`state` 是最后时间步的隐藏状态。

基础 Seq2Seq 主要使用 `state` 作为上下文。

后面注意力机制会更重视 `output`，因为注意力需要查看源序列每个位置的信息。

### 训练和预测的解码器输入不同

训练时：

```text
输入真实目标词的前缀
```

预测时：

```text
输入模型上一步预测出来的词
```

这是 Seq2Seq 非常重要的区别。

### padding 不等于真实词

`<pad>` 只是补齐长度。

它不能参与损失。

所以 `MaskedSoftmaxCELoss` 是这章非常关键的代码。

## 15. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. Seq2Seq 用编码器读取源语言，用解码器生成目标语言
2. 编码器最后的隐藏状态可以作为解码器的初始状态
3. 解码器每一步会结合目标词嵌入和源句子上下文
4. 训练时使用 teacher forcing，预测时使用上一步预测词
5. padding 位置必须用 mask 从损失中排除
6. BLEU 可以衡量预测译文和参考译文的 n-gram 重合程度

如果最后只留一句话：

```text
Seq2Seq 的关键不是某一层代码，而是把“读完整个输入序列”和“逐步生成输出序列”连接成一个可训练的翻译系统
```
