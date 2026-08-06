# 40 Bahdanau 注意力学习笔记

对应文件：`Attention Mechanism/40_Bahdanau.ipynb`

这份笔记是结合你的文件 [40_Bahdanau.ipynb](</D:/Learn/d2l-test/Attention Mechanism/40_Bahdanau.ipynb>) 写的。它把前面学到的加性注意力接入 Seq2Seq 翻译模型，让解码器在生成每个目标词时动态关注源句子的不同位置。

如果用一句话概括这一章：

```text
Bahdanau 注意力让 Seq2Seq 解码器不再只依赖一个固定上下文，而是在每个解码时间步重新对源序列做注意力汇聚
```

## 1. 为什么基础 Seq2Seq 还不够

基础 Seq2Seq 中，编码器读完整个源句子后，把最后隐藏状态交给解码器。

这种方法相当于把整句话压缩成一个向量。

短句子问题不大。

但句子变长后，一个向量很难保存所有细节。

例如源句子中有主语、动词、宾语、地点、时间，解码器生成不同目标词时应该关注不同部分。

所以 Bahdanau 注意力引入了一个更灵活的机制：

```text
每生成一个词，都根据当前解码器状态重新查看编码器所有时间步输出
```

## 2. Bahdanau 注意力的核心思想

在每个解码时间步：

1. 用解码器当前隐藏状态作为 query
2. 用编码器每个时间步输出作为 key 和 value
3. 用加性注意力计算 query 对每个源词位置的权重
4. 对编码器输出做加权求和，得到 context
5. 把 context 和当前目标词嵌入拼接后送入 GRU

可以写成：

```text
当前解码状态 -> 查询源序列 -> 得到上下文 -> 生成下一个词
```

这比基础 Seq2Seq 的固定上下文更强。

## 3. AttentionDecoder 接口

notebook 先定义：

```python
class AttentionDecoder(d2l.Decoder):
```

它只额外要求解码器提供：

```python
attention_weights
```

这个属性用来保存每个解码步骤的注意力权重。

为什么要保存？

因为注意力权重不仅参与计算，还可以拿来可视化。

翻译后我们可以画热力图，看看目标词分别关注了源句子哪些位置。

## 4. Seq2SeqAttentionDecoder 的结构

核心类是：

```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
```

它包含：

```python
self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
self.embedding = nn.Embedding(vocab_size, embed_size)
self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers)
self.dense = nn.Linear(num_hiddens, vocab_size)
```

各部分作用是：

- `attention`：根据当前解码状态从源序列中取上下文
- `embedding`：把目标词编号转为词向量
- `rnn`：逐步生成目标序列隐藏状态
- `dense`：把隐藏状态映射到目标词表分数

注意 GRU 输入维度是：

```text
embed_size + num_hiddens
```

因为每一步输入都由目标词嵌入和上下文向量拼接而成。

## 5. init_state 做了什么

解码器初始化状态：

```python
outputs, hidden_state = enc_outputs
return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
```

编码器输出原本形状是：

```text
(num_steps, batch_size, num_hiddens)
```

注意力计算更习惯使用：

```text
(batch_size, num_steps, num_hiddens)
```

所以要 `permute(1, 0, 2)`。

返回的状态包含三部分：

- `enc_outputs`：源序列每个位置的编码结果
- `hidden_state`：解码器 GRU 的隐藏状态
- `enc_valid_lens`：源序列有效长度，用来 mask padding

## 6. forward 中每一步如何生成

解码器会遍历目标序列的每个时间步：

```python
for x in X:
```

每一步取最后一层隐藏状态作为 query：

```python
query = torch.unsqueeze(hidden_state[-1], dim=1)
```

然后计算上下文：

```python
context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
```

这里：

- query：当前解码状态
- keys：编码器所有时间步输出
- values：编码器所有时间步输出
- valid_lens：源句子有效长度

得到的 `context` 表示当前要生成目标词时，源句子中最相关的信息汇总。

## 7. context 和目标词嵌入为什么要拼接

代码中：

```python
x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
```

当前目标词嵌入告诉模型：

```text
目标句子已经生成到哪里
```

context 告诉模型：

```text
当前应该重点参考源句子的哪些信息
```

二者拼接后，GRU 才能综合“目标端历史”和“源端上下文”生成下一步隐藏状态。

## 8. 和基础 Seq2Seq 的区别

基础 Seq2Seq：

```text
编码器最后隐藏状态 -> 解码器初始状态
```

Bahdanau 注意力：

```text
编码器所有时间步输出 -> 每个解码时间步动态注意力汇聚
```

核心差别是：

```text
上下文从固定向量变成动态向量
```

所以它更适合长序列。

## 9. 形状测试为什么重要

notebook 用假数据测试：

```python
X = torch.zeros((4, 7), dtype=torch.long)
output, state = decoder(X, state)
```

它检查输出形状和状态结构是否正确。

注意力模型的维度比较复杂，尤其是：

- 时间步维度
- batch 维度
- 隐藏维度
- query/key/value 维度

先用小数据测试形状，可以避免训练时才爆出难查的维度错误。

## 10. 训练函数做了什么

notebook 中写了 `train_seq2seq_fixed`。

它整体流程和前面的 Seq2Seq 类似：

1. Xavier 初始化参数
2. 使用 Adam 优化器
3. 使用交叉熵损失并忽略 `<pad>`
4. 构造解码器输入
5. 前向传播得到预测
6. 反向传播
7. 梯度裁剪
8. 更新参数

其中解码器输入仍然使用 teacher forcing：

```python
dec_input = torch.cat([bos, Y[:, :-1]], 1)
```

也就是：

```text
<bos> + 真实目标句子去掉最后一个词
```

## 11. 为什么使用 ignore_index

损失函数是：

```python
nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
```

这表示 `<pad>` 位置不参与损失。

它和前面自定义 masked loss 的目标一致：

```text
不要让模型为了 padding 学习无意义模式
```

## 12. 翻译和 BLEU

训练后，notebook 测试：

```python
engs = ['go .', "i lost .", "he's calm .", "i'm home ."]
```

模型会输出翻译和 BLEU 分数。

BLEU 用来衡量预测译文和参考译文的 n-gram 重合程度。

如果翻译结果和参考译文越接近，BLEU 通常越高。

## 13. 注意力热力图怎么看

最后 notebook 整理：

```python
attention_weights
```

并画热力图。

横轴是源句子的 key positions。

纵轴是目标句子的 query positions。

可以理解为：

```text
生成第 i 个目标词时，模型主要看源句子的哪些位置
```

理想情况下，相关词之间会有更深颜色。

这使 Bahdanau 注意力比基础 Seq2Seq 更可解释。

## 14. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. Bahdanau 注意力使用加性注意力
2. 解码器当前隐藏状态是 query
3. 编码器所有时间步输出是 key 和 value
4. 每个解码时间步都会重新计算 context
5. context 和目标词嵌入拼接后送入 GRU
6. 注意力热力图能解释目标词关注了源句子哪些位置

如果最后只留一句话：

```text
Bahdanau 注意力让翻译模型从“读完一句话只记一个向量”升级为“生成每个词时都能回头看源句子”
```

