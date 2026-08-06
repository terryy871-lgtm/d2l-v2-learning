# 43 Transformer 学习笔记

对应文件：`Attention Mechanism/43_Transformer.ipynb`

这份笔记是结合你的文件 [43_Transformer.ipynb](</D:/Learn/d2l-test/Attention Mechanism/43_Transformer.ipynb>) 写的。它把前面学过的多头注意力、自注意力、位置编码、残差连接和前馈网络组合起来，实现完整的 Transformer 编码器-解码器翻译模型。

如果用一句话概括这一章：

```text
Transformer 用多头注意力代替循环结构，通过编码器和解码器堆叠完成序列到序列建模
```

## 1. Transformer 为什么重要

前面的 RNN、GRU、LSTM 都按时间步处理序列。

它们适合序列数据，但有两个问题：

1. 长距离依赖需要逐步传递
2. 时间步之间难以完全并行

Transformer 的核心变化是：

```text
不再依赖循环结构，而是用自注意力直接建模序列中任意位置之间的关系
```

这让模型更容易并行训练，也更擅长捕捉长距离依赖。

## 2. Transformer 的整体结构

Transformer 仍然是编码器-解码器架构。

编码器负责读取源语言序列。

解码器负责生成目标语言序列。

不同的是，编码器和解码器内部不再使用 GRU，而是使用：

- 多头注意力
- 残差连接
- 层规范化
- 逐位置前馈网络
- 位置编码

这些模块组合成一个个 block，再堆叠多层。

## 3. PositionWiseFFN：逐位置前馈网络

notebook 中定义：

```python
class PositionWiseFFN(nn.Module):
```

它由两个线性层和一个 ReLU 组成：

```python
dense1 -> ReLU -> dense2
```

它叫“逐位置”，是因为它对每个时间步的位置独立应用同一个前馈网络。

输入形状：

```text
(batch_size, num_steps, ffn_num_input)
```

输出形状：

```text
(batch_size, num_steps, ffn_num_outputs)
```

它不会混合不同时间步的信息。

不同位置的信息交流主要由注意力完成。

## 4. LayerNorm 和 BatchNorm 的区别

notebook 对比了：

```python
nn.LayerNorm
nn.BatchNorm1d
```

批量规范化通常沿 batch 维统计均值和方差。

层规范化则对每个样本内部的特征维度做规范化。

序列模型中 batch 内句子长度、padding、时间步结构都比较复杂。

所以 Transformer 更常使用 LayerNorm。

它对每个样本独立处理，不依赖 batch 内其他样本的统计。

## 5. AddNorm：残差连接加层规范化

代码定义：

```python
class AddNorm(nn.Module):
```

核心是：

```python
return self.ln(self.dropout(Y) + X)
```

这里包含三个动作：

1. 对子层输出 `Y` 做 dropout
2. 和原输入 `X` 做残差相加
3. 做 LayerNorm

残差连接可以让梯度更容易传播。

LayerNorm 可以稳定每层输入分布。

这两个结构是 Transformer 能堆叠多层的重要原因。

## 6. EncoderBlock：编码器块

编码器块包含：

```python
self.attention = d2l.MultiHeadAttention(...)
self.addnorm1 = AddNorm(...)
self.ffn = PositionWiseFFN(...)
self.addnorm2 = AddNorm(...)
```

前向传播：

```python
Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
return self.addnorm2(Y, self.ffn(Y))
```

第一步是自注意力：

```text
Q = K = V = X
```

每个源词都可以关注源序列中的其他词。

第二步是逐位置前馈网络。

整体结构是：

```text
多头自注意力 -> AddNorm -> FFN -> AddNorm
```

## 7. TransformerEncoder：完整编码器

完整编码器包含：

1. 词元嵌入
2. 位置编码
3. 多个 EncoderBlock

代码中：

```python
X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
```

嵌入乘以：

```text
sqrt(num_hiddens)
```

是为了调整词嵌入的数值尺度，让它和位置编码更匹配。

然后编码器逐层处理：

```python
for i, blk in enumerate(self.blks):
    X = blk(X, valid_lens)
```

每层都会更新序列中每个位置的表示。

## 8. 编码器注意力权重

编码器保存：

```python
self.attention_weights[i] = blk.attention.attention.attention_weights
```

这些权重可以用来可视化每层、每个头的注意力分布。

在编码器自注意力中，横轴和纵轴都对应源序列位置。

热力图可以观察：

```text
每个源词在编码时关注了哪些源词
```

## 9. DecoderBlock：解码器块

解码器块比编码器块复杂。

它包含三部分：

1. 掩蔽多头自注意力
2. 编码器-解码器多头注意力
3. 逐位置前馈网络

每部分后面都有 AddNorm。

整体结构是：

```text
掩蔽自注意力 -> AddNorm -> 编码器-解码器注意力 -> AddNorm -> FFN -> AddNorm
```

## 10. 为什么解码器自注意力要 mask

训练时目标句子是一次性输入的。

如果不做 mask，第 1 个位置可能看到第 2、3、4 个真实词。

这就相当于考试时偷看答案。

所以训练时构造：

```python
dec_valid_lens = torch.arange(1, num_steps + 1).repeat(batch_size, 1)
```

含义是：

```text
第 1 个 query 只能看 1 个 key
第 2 个 query 只能看 2 个 key
...
```

这样解码器只能关注当前及之前的位置。

## 11. 编码器-解码器注意力

解码器第二层注意力是：

```python
Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
```

这里：

- query 来自解码器当前表示
- key 和 value 来自编码器输出

它的作用是：

```text
生成目标词时，回头查看源语言序列中相关位置
```

这和 Bahdanau 注意力的思想一致，只是这里使用多头注意力。

## 12. 预测阶段为什么要缓存 key_values

代码中：

```python
state[2][self.i] = key_values
```

解码器预测时是一个词一个词生成。

如果每一步都重新计算之前所有位置，会浪费计算。

缓存历史 key-value 后，下一步只需要追加新位置。

这提高了自回归预测效率。

## 13. TransformerDecoder：完整解码器

完整解码器包含：

1. 目标词嵌入
2. 位置编码
3. 多个 DecoderBlock
4. 输出线性层

最后：

```python
return self.dense(X), state
```

输出形状是：

```text
(batch_size, num_steps, tgt_vocab_size)
```

每个目标位置都会得到一个词表预测分数。

## 14. 训练 Transformer 翻译模型

notebook 设置：

```python
num_hiddens = 32
num_layers = 2
num_heads = 4
num_epochs = 200
```

这是教学规模的 Transformer。

训练流程仍然是 Seq2Seq：

1. 加载机器翻译数据
2. 构造编码器和解码器
3. 使用 teacher forcing 构造解码器输入
4. 计算交叉熵损失
5. 忽略 `<pad>`
6. 梯度裁剪
7. 更新参数

虽然模型结构换成 Transformer，但训练目标仍然是：

```text
给定源语言和目标前缀，预测目标序列下一个词
```

## 15. 翻译和 BLEU

训练后，notebook 用几个短句测试：

```python
go .
i lost .
he's calm .
i'm home .
```

预测函数逐步生成目标词。

BLEU 分数衡量预测译文和参考译文的 n-gram 重合。

这和前面的 Seq2Seq、Bahdanau 注意力章节保持一致。

## 16. 注意力可视化

notebook 最后可视化三类注意力：

1. 编码器自注意力
2. 解码器掩蔽自注意力
3. 编码器-解码器交叉注意力

编码器自注意力显示源序列内部如何互相关注。

解码器自注意力显示目标序列生成时如何关注已生成位置。

交叉注意力显示目标词如何关注源词位置。

这些热力图能帮助理解 Transformer 的内部行为。

## 17. Transformer 和 RNN Seq2Seq 的区别

RNN Seq2Seq：

```text
按时间步逐个处理，隐藏状态传递历史信息
```

Transformer：

```text
通过注意力直接建立任意位置之间的联系
```

RNN 的顺序性强，但并行能力弱。

Transformer 并行能力强，但需要位置编码补充顺序信息。

## 18. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. Transformer 仍然是编码器-解码器架构
2. 编码器块由多头自注意力和 FFN 组成
3. 解码器块多了掩蔽自注意力和交叉注意力
4. AddNorm 是残差连接加层规范化
5. 位置编码为自注意力补充顺序信息
6. 解码器训练时必须 mask 未来位置
7. 注意力权重可以用热力图解释模型关注位置

如果最后只留一句话：

```text
Transformer 的核心是用多头注意力在序列位置之间直接传递信息，再通过残差、规范化和前馈网络堆叠成强大的序列模型
```

