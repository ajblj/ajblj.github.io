# 2 预备知识


## 数据操作

访问元素，子区间访问为开区间，如 ***1: 3*** 的含义是[1, 3)左闭右开，***:: 3*** 的含义是从头到尾跳 3 个访问

![Figure 1-1 数据操作](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231009112555278.png)

使用 `arange` 创建一个行向量 `x`

```python
x = torch.arange(12)
output: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
```

可以通过张量的 `shape` 属性来访问张量（沿每个轴的长度）的形状

```python
x.shape
output: torch.Size([12])
```

如果只想知道张量中元素的总数，即形状的所有元素乘积，可以检查它的大小（size）。因为这里在处理的是一个向量，所以它的 `shape` 与它的 `size` 相同

```python
x.numel()
output: 12
```

要想改变一个张量的形状而不改变元素数量和元素值，可以调用`reshape`函数，此外我们可以通过`-1`来调用此自动计算出维度的功能。 即可以用`x.reshape(-1,4)`或`x.reshape(3,-1)`来取代`x.reshape(3,4)`，因为知道宽度或者高度后，另一个维度会被自动计算得到

```python
X = x.reshape(3, 4)
output:tensor([[ 0,  1,  2,  3],
        			[ 4,  5,  6,  7],
        			[ 8,  9, 10, 11]])
```

使用全0、全1、其他常量，或者从特定分布中随机采样的数字，来初始化矩阵

```python
torch.zeros((2, 3, 4))
output:tensor([[[0., 0., 0., 0.],
         				[0., 0., 0., 0.],
         				[0., 0., 0., 0.]],
        			[[0., 0., 0., 0.],
         				[0., 0., 0., 0.],
         				[0., 0., 0., 0.]]])

torch.ones((2, 3, 4))
output:tensor([[[1., 1., 1., 1.],
         				[1., 1., 1., 1.],
         				[1., 1., 1., 1.]],
        			[[1., 1., 1., 1.],
         				[1., 1., 1., 1.],
         				[1., 1., 1., 1.]]])

torch.randn(3, 4)
output:tensor([[-0.0135,  0.0665,  0.0912,  0.3212],
        			[ 1.4653,  0.1843, -1.6995, -0.3036],
        			[ 1.7646,  1.0450,  0.2457, -0.7732]])
```

通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值

```python
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
output: tensor([[2, 1, 4, 3],
        				[1, 2, 3, 4],
        				[4, 3, 2, 1]])
```

### 运算符

对于任意具有相同形状的张量， 常见的标准算术运算符（`+`、`-`、`*`、`/`和`**`）都可以被升级为按元素运算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
output:(tensor([ 3.,  4.,  6., 10.]),
 				tensor([-1.,  0.,  2.,  6.]),
 				tensor([ 2.,  4.,  8., 16.]),
 				tensor([0.5000, 1.0000, 2.0000, 4.0000]),
 				tensor([ 1.,  4., 16., 64.]))


torch.exp(x)
output: tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])
```

将多个张量连结（concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量。我们只需要提供张量列表，并给出沿哪个轴连结。

```python
X = torch.arange(12, dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
output:(tensor([[ 0.,  1.,  2.,  3.],
         				[ 4.,  5.,  6.,  7.],
         				[ 8.,  9., 10., 11.],
         				[ 2.,  1.,  4.,  3.],
         				[ 1.,  2.,  3.,  4.],
         				[ 4.,  3.,  2.,  1.]]),
 				tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
         				[ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
         				[ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))
```

通过*逻辑运算符*构建二元张量，以及对张量中所有元素求和

```python
X == Y
output:tensor([[False,  True, False,  True],
        			[False, False, False, False],
        			[False, False, False, False]])

X.sum()
output:tensor(66.)
```

### 广播机制

即使张量形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作，这种机制的工作方式如下：

- 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
- 对生成的数组执行按元素操作。

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
output:(tensor([[0],
                [1],
                [2]]),
        tensor([[0, 1]]))

a + b
output:tensor([[0, 1],
               [1, 2],
               [2, 3]])
```

### 索引和切片

可以用`[-1]`选择最后一个元素，可以用`[1:3]`选择第二个和第三个元素

```python
X = torch.arange(12, dtype=torch.float32).reshape(3,4)
X[-1], X[1:3]
output:(tensor([ 8.,  9., 10., 11.]),
				tensor([[ 4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11.]]))
```

我们想为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值。

```python
X[0:2, :] = 12
X
output:tensor([[12., 12., 12., 12.],
               [12., 12., 12., 12.],
               [ 8.,  9., 10., 11.]])
```

### 节省内存

运行一些操作可能会导致为新结果分配内存，用Python的`id()`函数演示了这一点

```python
before = id(Y)
Y = Y + X
id(Y) == before
output: False
```

执行原地操作非常简单，可以使用切片表示法将操作的结果分配给先前分配的数组，也可以用`+=`

```python
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
output:id(Z): 135358378815744
			 id(Z): 135358378815744
  
X += Y
```

### 转换为其他Python对象

将深度学习框架定义的张量转换为NumPy张量`ndarray`很容易，使用`numpy()`；反之也同样容易，使用`torch.tensor()`

```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
output:(numpy.ndarray, torch.Tensor)
```

将大小为1的张量转换为Python标量，可以调用`item`函数或Python的内置函数

```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
output:(tensor([3.5000]), 3.5, 3.5, 3)
```

## 数据预处理

### 读取数据集

从CSV文件中加载原始数据集

```python
import pandas as pd

data = pd.read_csv(data_file)
print(data)
output:NumRooms Alley   Price
		0       NaN  Pave  127500
		1       2.0   NaN  106000
		2       4.0   NaN  178100
		3       NaN   NaN  140000
```

### 处理缺失值

为了处理缺失的数据，典型的方法包括插值法和删除法， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

在这里，我们将考虑插值法，通过位置索引`iloc`，我们将`data`分成`inputs`和`outputs`， 其中前者为`data`的前两列，而后者为`data`的最后一列。对于`inputs`中缺少的数值，我们用同一列的**均值**替换“NaN”项。

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
output:NumRooms Alley
	0       3.0  Pave
	1       2.0   NaN
	2       4.0   NaN
	3       3.0   NaN
```

对于`inputs`中不能取均值的类别值或离散值，我们将“NaN”视为一个类别。由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， `pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。

> dummy_na : bool, default False
> Add a column to indicate NaNs, if False NaNs are ignored.

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
output:NumRooms  Alley_Pave  Alley_nan
0       3.0           1          0
1       2.0           0          1
2       4.0           0          1
3       3.0           0          1
```

### 转换为张量格式

现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式

```python
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y
output:(tensor([[3., 1., 0.],
                [2., 0., 1.],
                [4., 0., 1.],
                [3., 0., 1.]], dtype=torch.float64),
        tensor([127500., 106000., 178100., 140000.], dtype=torch.float64))
```

## 线性代数

### 标量

![Figure 3-1 标量](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010105731009.png)

标量由只有一个元素的张量表示

```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
output: (tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```

### 向量

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010110116335.png" alt="image-20231010110116335" style="zoom:33%;" />

点乘（若两向量正交，则下面式子等于0）：
$$
a^Tb=\sum_{i}{a_i}{b_i}
$$

向量可以被视为标量值组成的列表，可以通过索引来访问任一元素

```python
x = torch.arange(4)
output: tensor([0, 1, 2, 3])

x[3]
output: tensor(3)
```

调用Python的内置`len()`函数来访问张量的长度，当用张量表示一个向量（只有一个轴）时，我们也可以通过`.shape`属性访问向量的长度。 形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）

```python
len(x)
output: 4

x.shape
output: torch.Size([4])
```

### 矩阵

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010110818322.png" alt="image-20231010110818322" style="zoom:33%;" />

一般采用F范数进行矩阵范数的计算：

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010111412545.png" alt="image-20231010111412545" style="zoom:33%;" />

对称矩阵$A_{ij}=A_{ji}$，反对称矩阵$A_{ij}=-A_{ji}$

正交矩阵：

- 所有行都相互正交
- 所有行都有单位长度 $U \, with\, \sum_{j}{U_{ij}U_{kj}}$
- 可以写成$UU^T=1$

置换矩阵（是正交矩阵）：$P \; where\; P_{ij}=1 \;if \;and \;only \;if \;j=\pi(i)$

特征向量和特征值：$Ax=\lambda x$

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010113708779.png" alt="image-20231010113708779" style="zoom:33%;" />

- 不被矩阵改变方向的向量$x$是特征向量，对称矩阵总是可以找到特征向量
- 特征值是$\lambda$



通过指定两个分量m和n来创建一个形状为m×n的矩阵

```python
A = torch.arange(20).reshape(5, 4)
output: tensor([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11],
                [12, 13, 14, 15],
                [16, 17, 18, 19]])
```

矩阵的转置

```python
A.T
output: tensor([[ 0,  4,  8, 12, 16],
                [ 1,  5,  9, 13, 17],
                [ 2,  6, 10, 14, 18],
                [ 3,  7, 11, 15, 19]])
```

### 张量

与矩阵相似构建张量

```python
X = torch.arange(24).reshape(2, 3, 4)
output: tensor([[[ 0,  1,  2,  3],
                 [ 4,  5,  6,  7],
                 [ 8,  9, 10, 11]],
                
                [[12, 13, 14, 15],
                 [16, 17, 18, 19],
                 [20, 21, 22, 23]]])
```

给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
output: (tensor([[ 0.,  1.,  2.,  3.],
                 [ 4.,  5.,  6.,  7.],
                 [ 8.,  9., 10., 11.],
                 [12., 13., 14., 15.],
                 [16., 17., 18., 19.]]),
         tensor([[ 0.,  2.,  4.,  6.],
                 [ 8., 10., 12., 14.],
                 [16., 18., 20., 22.],
                 [24., 26., 28., 30.],
                 [32., 34., 36., 38.]]))
```

两个矩阵的按元素乘法称为Hadamard积（Hadamard product）（数学符号⊙）
$$
\mathbf{A} \odot \mathbf{B} =
\left[
\begin{matrix}
  a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n}  \\\
  a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n}  \\\
  \vdots & \vdots & \ddots & \vdots  \\\
  a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn} 
\end{matrix}
\right]
$$

```python
A * B
output: tensor([[  0.,   1.,   4.,   9.],
                [ 16.,  25.,  36.,  49.],
                [ 64.,  81., 100., 121.],
                [144., 169., 196., 225.],
                [256., 289., 324., 361.]])
```

将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘

```python
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
output: (tensor([[[ 2,  3,  4,  5],
                  [ 6,  7,  8,  9],
                  [10, 11, 12, 13]],
                 [[14, 15, 16, 17],
                  [18, 19, 20, 21],
                  [22, 23, 24, 25]]]),
         torch.Size([2, 3, 4]))
```

### 降维

可以指定张量沿哪一个轴来通过求和降低维度

```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
output: tensor([[ 0.,  1.,  2.,  3.],
                [ 4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11.],
                [12., 13., 14., 15.],
                [16., 17., 18., 19.]]

A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
output: (tensor([40., 45., 50., 55.]), torch.Size([4]))

A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
output: (tensor([6., 22., 38., 54., 70.]), torch.Size([5]))
```

沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和

```python
A.sum(axis=[0, 1])  # 结果和A.sum()相同
output: tensor(190.)
```

一个与求和相关的量是平均值（mean或average），同样，计算平均值的函数也可以沿指定轴降低张量的维度

```python
A.mean(), A.sum() / A.numel()
output: (tensor(9.5000), tensor(9.5000))

A.mean(axis=0), A.mean(axis=1), A.sum(axis=0) / A.shape[0]
output: (tensor([ 8.,  9., 10., 11.]),
         tensor([ 1.5000,  5.5000,  9.5000, 13.5000, 17.5000]),
         tensor([ 8.,  9., 10., 11.]))
```

有时在调用函数来计算总和或均值时保持轴数不变会很有用

```python
sum_A = A.sum(axis=1, keepdims=True)
output: tensor([[ 6.],
                [22.],
                [38.],
                [54.],
                [70.]])
```

由于`sum_A`在对每行进行求和后仍保持两个轴，我们可以通过广播将`A`除以`sum_A`

```python
A / sum_A
output: tensor([[0.0000, 0.1667, 0.3333, 0.5000],
                [0.1818, 0.2273, 0.2727, 0.3182],
                [0.2105, 0.2368, 0.2632, 0.2895],
                [0.2222, 0.2407, 0.2593, 0.2778],
                [0.2286, 0.2429, 0.2571, 0.2714]])
```

沿某个轴计算`A`元素的累积总和

```python
A.cumsum(axis=0)
output: tensor([[ 0.,  1.,  2.,  3.],
                [ 4.,  6.,  8., 10.],
                [12., 15., 18., 21.],
                [24., 28., 32., 36.],
                [40., 45., 50., 55.]])
```

### 点积

给定两个向量$\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，它们的点积（dot product）$\mathbf{x}^\top\mathbf{y}$（或$\langle\mathbf{x},\mathbf{y}\rangle$）是相同位置的按元素乘积的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。也可以通过执行按元素乘法，然后进行求和来表示两个向量的点积

```python
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
x, y, torch.dot(x, y)
output: (tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))

torch.sum(x * y)
output: tensor(6.)
```

### 矩阵-向量积和乘法

#### 矩阵向量积

在代码中使用张量表示矩阵-向量积，我们使用`mv`函数。 当我们为矩阵`A`和向量`x`调用`torch.mv(A, x)`时，会执行矩阵-向量积。`A`的列维数（沿轴1的长度）必须与`x`的维数（其长度）相同

```python
A.shape, x.shape, torch.mv(A, x)
output: (torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```

#### 矩阵乘法

使用`mm`函数对矩阵做矩阵乘法

```python
B = torch.ones(4, 3)
torch.mm(A, B)
output: tensor([[ 6.,  6.,  6.],
                [22., 22., 22.],
                [38., 38., 38.],
                [54., 54., 54.],
                [70., 70., 70.]])
```

### 范数

在线性代数中，向量范数是将向量映射到标量的函数$f$。 给定任意向量$x$，向量范数要满足一些属性：

- 如果我们按常数因子$\alpha$缩放向量的所有元素， 其范数也会按相同常数因子的***绝对值***缩放：
  $$
  f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x})
  $$

- 熟悉的三角不等式：
  $$
  f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y})
  $$

- 范数必须是非负的，且范数最小为0，当且仅当向量全由0组成：
  $$
  f(\mathbf{x}) \geq 0
  $$

**$L_2$范数**是向量元素平方和的平方根：$||\mathbf{x}||_ 2 = \sqrt{\sum_{i=1}^n x_i^2}$，其中，在$L_2$范数中常常省略下标$2$，也就是说$||\mathbf{x}||$等同于$||\mathbf{x}||_2$。 在代码中，我们可以按如下方式计算向量的$L_2$范数：

```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)
output: tensor(5.)
```

**$L_1$范数**它表示为向量元素的绝对值之和：$||\mathbf{x}||_ 1 = \sum_{i=1}^n \left|x_i \right|$，与$L_2$范数相比，$L_1$范数受异常值的影响较小。为了计算$L_1$范数，将绝对值函数和按元素求和组合起来：

```python
torch.abs(u).sum()
output: tensor(7.)
```

$L_2$范数和$L_1$范数都是更一般的$L_p$范数的特例：$||\mathbf{x}||_ p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}$。

类似于向量的$L_2$范数，矩阵$\mathbf{X} \in \mathbb{R}^{m \times n}$的**Frobenius范数**是矩阵元素平方和的平方根：$||\mathbf{X}||_ F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}$，Frobenius范数满足向量范数的所有性质，它就像是矩阵形向量的$L_2$范数。调用以下函数将计算矩阵的Frobenius范数：

```python
torch.norm(torch.ones((4, 9)))
output: tensor(6.)
```

## 微积分

### 导数和微分

可以将拟合模型的任务分解为两个关键问题：

- *优化*（optimization）：用模型拟合观测数据的过程
- *泛化*（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型

给定$y=f(x)$，其中$x$和$y$分别是函数$f$的自变量和因变量。以下表达式是等价的：
$$
f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x)
$$
其中符号$\frac{d}{dx}$和$D$是**微分运算符**，表示**微分**操作，可以使用以下规则来对常见函数求微分：

* $DC = 0$（$C$是一个常数）

* $Dx^n = nx^{n-1}$

* $De^x = e^x$

* $D\ln(x) = 1/x$

假设函数$f$和$g$都是可微的，$C$是一个常数，则：

- 常数相乘法则：$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$
- 加法法则：$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$
- 乘法法则：$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$
- 除法法则：$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010202158099.png" alt="image-20231010202158099" style="zoom:33%;" />

### 偏导数

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010204549829.png" alt="image-20231010204549829" style="zoom:33%;" />

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010205217466.png" alt="image-20231010205217466" style="zoom:33%;" />

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010205010424.png" alt="image-20231010205010424" style="zoom:33%;" />

在深度学习中，函数通常依赖于许多变量。因此，我们需要将微分的思想推广到**多元函数**上。为了计算$\frac{\partial y}{\partial x_i}$，我们可以简单地将$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$看作常数，并计算$y$关于$x_i$的导数。对于偏导数的表示，以下是等价的：
$$
\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f
$$

### 梯度

可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。具体而言，设函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$的输入是一个$n$维向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量：
$$
\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top
$$
其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义时被$\nabla f(\mathbf{x})$取代。

假设$\mathbf{x}$为$n$维向量，在微分多元函数时经常使用以下规则:

* 对于所有$\mathbf{A} \in \mathbb{R}^{m \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times m}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

同样，对于任何矩阵$\mathbf{X}$，都有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。

### 链式法则

然而，上面方法可能很难找到梯度，因为在深度学习中，多元函数通常是*复合*（composite）的， 所以难以应用上述任何规则来微分这些函数。 幸运的是，链式法则可以被用来微分复合函数。

先考虑单变量函数。假设函数$y=f(u)$和$u=g(x)$都是可微的，根据链式法则：
$$
\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}
$$
当函数具有任意数量的变量的情况时，假设可微分函数$y$有变量$u_1, u_2, \ldots, u_m$，其中每个可微分函数$u_i$都有变量$x_1, x_2, \ldots, x_n$。对于任意$i = 1, 2, \ldots, n$，链式法则给出：
$$
\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial u_1} \frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2} \frac{\partial u_2}{\partial x_i} + \cdots + \frac{\partial y}{\partial u_m} \frac{\partial u_m}{\partial x_i}
$$

## 自动微分

自动求导的两种模式：正向累积和反向累积（反向传播）

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231010214757564.png" alt="image-20231010214757564" style="zoom:33%;" />

正向累积：计算复杂度是O(n)，用来计算一个变量的梯度；内存复杂度是O(1)

反向累积：计算复杂度是O(n)，n是操作子个数；内存复杂度是O(n)，因为需要存储正向的所有中间结果



假设我们想对函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于列向量$\mathbf{x}$求导

```python
import torch

x = torch.arange(4.0)
x
output: tensor([0., 1., 2., 3.])
```

在我们计算$y$关于$\mathbf{x}$的梯度之前，需要一个地方来存储梯度，使用`requires_grad_(True)`。一个标量函数关于向量$\mathbf{x}$的梯度是向量，并且与$\mathbf{x}$具有相同的形状，而标量函数关于向量$\mathbf{x}$的导数是向量，且其形状是向量$\mathbf{x}$的转置

```python
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None

y = 2 * torch.dot(x, x)
y
output: tensor(28., grad_fn=<MulBackward0>)
```

通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度

```python
y.backward()
x.grad
output: tensor([2., 2., 2., 2.])
```

根据手算可得，函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于$\mathbf{x}$的梯度应为$4\mathbf{x}$

如果要计算$\mathbf{x}$的另一个函数，需要首先清楚梯度的值

```python
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
output: tensor([1., 1., 1., 1.])
```

### 非标量变量的反向传播

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵，但我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和

```python
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
output: tensor([0., 2., 4., 6.])
```

### 分离计算

有时希望**将某些计算移动到记录的计算图之外**，例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。若想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数，并且只考虑到`x`在`y`被计算后发挥的作用。

```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
output: tensor([True, True, True, True])
```

由于记录了`y`的计算结果，可以随后在`y`上调用反向传播， 得到`y=x*x`关于的`x`的导数，即`2*x`。

```python
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
output: tensor([True, True, True, True])
```

### Python控制流的梯度计算

使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），仍然可以计算得到的变量的梯度

```python
def f(a):
  b = a * 2
  while b.norm() < 1000:
    b = b * 2
    if b.sum() > 0:
      c = b
    else:
      c = 100 * b
    return c
  
# 随机标量
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

a.grad == d / a
output: tensor(True)
```

## 概率



---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/2-%E9%A2%84%E5%A4%87%E7%9F%A5%E8%AF%86/  

