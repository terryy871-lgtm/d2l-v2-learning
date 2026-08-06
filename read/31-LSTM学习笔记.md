# 31 LSTM 学习笔记

这份笔记是结合你的文件 [31_Long_Short-Term_Memory.ipynb](</D:/Learn/d2l-test/Recurrent_Neural_Network/31_Long_Short-Term_Memory.ipynb>) 写的，目标是帮助你真正理解：

- 为什么普通 RNN 和 GRU 之后还要学习 LSTM
- LSTM 的输入门、遗忘门、输出门分别控制什么
- 记忆元 `C` 和隐藏状态 `H` 有什么区别
- 从零实现 LSTM 的代码如何对应公式
- `nn.LSTM` 简洁实现和手写实现是什么关系

如果用一句话概括这一章：

```text
LSTM 的核心思想是：用独立的记忆元保存长期信息，再用三个门控制信息的写入、保留和输出
```

## 1. 先用人话理解 LSTM 为什么出现

普通 RNN 会不断用当前输入和上一隐藏状态更新新隐藏状态。

它的问题是：

```text
历史信息每一步都被重新混合，长距离信息很容易被冲淡
```

GRU 用更新门和重置门缓解了这个问题。

LSTM 更进一步，把“记忆”这件事单独拎出来，设计了一个记忆元：

```text
C_t
```

你可以把它理解成一条更稳定的信息通道。

隐藏状态 `H_t` 负责对外输出，记忆元 `C_t` 负责长期保存。

## 2. LSTM 有哪些核心变量

LSTM 每个时间步主要有：

- 输入 `X_t`
- 上一隐藏状态 `H_{t-1}`
- 上一记忆元 `C_{t-1}`
- 当前隐藏状态 `H_t`
- 当前记忆元 `C_t`

普通 RNN 通常只有隐藏状态。

LSTM 多了一个记忆元。

这就是它和普通 RNN 的关键区别。

## 3. 三个门分别控制什么

LSTM 中有三个门。

### 输入门

```text
控制有多少新信息写入记忆元
```

代码里是：

```python
I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
```

### 遗忘门

```text
控制旧记忆保留多少
```

代码里是：

```python
F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
```

### 输出门

```text
控制记忆元中有多少内容输出成隐藏状态
```

代码里是：

```python
O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
```

这些门都经过 `sigmoid`，所以值通常在 0 到 1 之间。

这让它们可以像比例开关一样控制信息流动。

## 4. 候选记忆元是什么

代码中：

```python
C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
```

候选记忆元可以理解成：

```text
当前时间步准备写入的新内容
```

但它不会全部写入。

到底写多少，要由输入门控制。

所以 LSTM 不是直接改写记忆，而是先生成候选内容，再决定写入比例。

## 5. 记忆元如何更新

核心代码是：

```python
C = F * C + I * C_tilda
```

这行非常重要。

它分成两部分：

### `F * C`

```text
旧记忆保留多少
```

### `I * C_tilda`

```text
新候选内容写入多少
```

所以新的记忆元是：

```text
保留的旧记忆 + 写入的新记忆
```

这就是 LSTM 处理长期依赖的核心。

## 6. 隐藏状态如何生成

代码中：

```python
H = O * torch.tanh(C)
```

这里先把记忆元通过 `tanh` 压到合适范围，再用输出门控制输出多少。

也就是说：

```text
记忆元不一定全部暴露给下一层或输出层
```

输出门决定当前时间步应该展示多少记忆内容。

## 7. `get_lstm_params` 在初始化什么

LSTM 要为每个门准备一套参数：

- 输入门参数：`W_xi, W_hi, b_i`
- 遗忘门参数：`W_xf, W_hf, b_f`
- 输出门参数：`W_xo, W_ho, b_o`
- 候选记忆元参数：`W_xc, W_hc, b_c`

最后还需要输出层参数：

- `W_hq`
- `b_q`

所以 LSTM 参数明显比普通 RNN 多。

但这也换来了更强的记忆控制能力。

## 8. `init_lstm_state` 为什么返回两个张量

代码中：

```python
return (
    torch.zeros((batch_size, num_hiddens), device=device),
    torch.zeros((batch_size, num_hiddens), device=device)
)
```

这两个分别是：

- `H`：隐藏状态
- `C`：记忆元

一开始没有历史信息，所以都初始化为 0。

这和 GRU 不一样。

GRU 通常只有隐藏状态，LSTM 有隐藏状态和记忆元两条状态线。

## 9. 从零实现和简洁实现的关系

notebook 后面用了：

```python
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNLM(lstm_layer, len(vocab))
```

`nn.LSTM` 会自动帮你处理：

- 输入门
- 遗忘门
- 输出门
- 记忆元更新
- 多时间步循环
- 参数注册

从零实现是为了看懂原理。

简洁实现是为了实际训练。

两者做的是同一件事，只是抽象层级不同。

## 10. 这份 notebook 你最应该记住什么

学完这一章，最重要的是记住：

1. LSTM 有隐藏状态 `H` 和记忆元 `C`
2. 输入门控制新信息写入
3. 遗忘门控制旧记忆保留
4. 输出门控制记忆如何变成隐藏状态
5. LSTM 比普通 RNN 和 GRU 更复杂，但长期记忆能力更强

如果最后只留一句话：

```text
LSTM 的强大之处在于，它不是每一步都粗暴覆盖记忆，而是细致地决定什么该忘、什么该写、什么该输出
```
