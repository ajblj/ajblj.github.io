# 6 卷积神经网络


## 从全连接层到卷积

*卷积神经网络*（convolutional neural networks，CNN）是机器学习利用自然图像中一些已知结构的创造性方法。

### 不变性

卷积神经网络正是将*空间不变性*（spatial invariance）的这一概念系统化，从而基于这个模型使用较少的参数来学习有用的表示。适合于计算机视觉的神经网络架构的特点：

- **平移不变性**（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。

-  **局部性**（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

### 多层感知机的限制

若用多层感知机处理图像信息，首先，多层感知机的输入是二维图像$\mathbf{X}$，其隐藏表示$\mathbf{H}$在数学上是一个矩阵，在代码中表示为二维张量。其中$\mathbf{X}$和$\mathbf{H}$具有相同的形状。为了方便理解，我们可以认为，无论是输入还是隐藏表示都拥有空间结构。使用$[\mathbf{X}]_ {i, j}$和$[\mathbf{H}]_ {i, j}$分别表示输入图像和隐藏表示中位置（$i$,$j$）处的像素。为了使每个隐藏神经元都能接收到每个输入像素的信息，我们将参数从权重矩阵（如同我们先前在多层感知机中所做的那样）替换为四阶权重张量$\mathsf{W}$。假设$\mathbf{U}$包含偏置参数，可以将全连接层形式化地表示为：
$$
\begin{aligned} 
\left[\mathbf{H}\right]_ {i, j} &= [\mathbf{U}]_ {i, j} + \sum_k \sum_l[\mathsf{W}]_ {i, j, k, l}  [\mathbf{X}]_ {k, l}\\\
&= [\mathbf{U}]_ {i, j} + \sum_ a \sum_ b [\mathsf{V}]_ {i, j, a, b}  [\mathbf{X}]_ {i+a, j+b}
\end{aligned}
$$
其中，从$\mathsf{W}$到$\mathsf{V}$的转换只是形式上的转换，因为在这两个四阶张量的元素之间存在一一对应的关系。我们只需重新索引下标$(k, l)$，使$k = i+a$、$l = j+b$，由此可得$[\mathsf{V}]_ {i, j, a, b} = [\mathsf{W}]_ {i, j, i+a, j+b}$。索引$a$和$b$通过在正偏移和负偏移之间移动覆盖了整个图像。对于隐藏表示中任意给定位置（$i$,$j$）处的像素值$[\mathbf{H}]_ {i, j}$，可以通过在$x$中以$(i, j)$为中心对像素进行加权求和得到，加权使用的权重为$[\mathsf{V}]_{i, j, a, b}$。

#### 平移不变性

现在引用上述的第一个原则：平移不变性。这意味着检测对象在输入$\mathbf{X}$中的平移，应该仅导致隐藏表示$\mathbf{H}$中的平移。也就是说，$\mathsf{V}$和$\mathbf{U}$实际上不依赖于$(i, j)$的值，即$[\mathsf{V}]_ {i, j, a, b} = [\mathbf{V}]_ {a, b}$。并且$\mathbf{U}$是一个常数，比如$u$。因此，我们可以简化$\mathbf{H}$定义为：
$$
[\mathbf{H}]_ {i, j} = u + \sum_ a\sum_ b [\mathbf{V}]_ {a, b} [\mathbf{X}]_ {i+a, j+b}
$$
这就是**卷积**（convolution）。我们是在使用系数$[\mathbf{V}]_ {a, b}$对位置$(i, j)$附近的像素$(i+a, j+b)$进行加权得到$[\mathbf{H}]_ {i, j}$。注意，$[\mathbf{V}]_ {a, b}$的系数比$[\mathsf{V}]_ {i, j, a, b}$少很多，因为前者不再依赖于图像中的位置。

#### 局部性

现在引用上述的第二个原则：局部性。如上所述，为了收集用来训练参数$[\mathbf{H}]_ {i, j}$的相关信息，我们不应偏离到距$(i, j)$很远的地方。这意味着在$|a|> \Delta$或$|b| > \Delta$的范围之外，我们可以设置$[\mathbf{V}]_ {a, b} = 0$。因此可以将$[\mathbf{H}]_ {i, j}$重写为
$$
[\mathbf{H}]_ {i, j} = u + \sum_ {a = -\Delta}^{\Delta} \sum_ {b = -\Delta}^{\Delta} [\mathbf{V}]_ {a, b}  [\mathbf{X}]_ {i+a, j+b}
$$
简而言之，上述公式是一个**卷积层**（convolutional layer），而卷积神经网络是包含卷积层的一类特殊的神经网络。在深度学习研究社区中，$\mathbf{V}$被称为**卷积核**（convolution kernel）或者**滤波器**（filter），亦或简单地称之为该卷积层的**权重**，通常该权重是可学习的参数。当图像处理的局部区域很小时，卷积神经网络与多层感知机的训练差异可能是巨大的：以前，多层感知机可能需要数十亿个参数来表示网络中的一层，而现在卷积神经网络通常只需要几百个参数，而且不需要改变输入或隐藏表示的维数。参数大幅减少的代价是，我们的特征现在是平移不变的，并且当确定每个隐藏活性值时，每一层只包含局部的信息。以上所有的权重学习都将依赖于归纳偏置。当这种偏置与现实相符时，我们就能得到样本有效的模型，并且这些模型能很好地泛化到未知数据中。但如果这偏置与现实不符时，比如当图像不满足平移不变时，我们的模型可能难以拟合我们的训练数据。

### 卷积

在进一步讨论之前，我们先简要回顾一下为什么上面的操作被称为卷积。在数学中，两个函数（比如$f, g: \mathbb{R}^d \to \mathbb{R}$）之间的“卷积”被定义为
$$
(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}
$$
也就是说，卷积是当把一个函数“翻转”并移位$\mathbf{x}$时，测量$f$和$g$之间的重叠。当为离散对象时，积分就变成求和。例如，对于从定义域为$\mathbb{Z}$的、平方可和的、无限维向量集合中抽取的向量，我们得到以下定义：


$$
(f * g)(i) = \sum_a f(a) g(i-a)
$$
对于二维张量，则为函数$f$的自变量$(a, b)$和函数$g$的自变量$(i-a, j-b)$上的对应加和：
$$
(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b)
$$
这看起来类似于公式3，但有一个主要区别：这里不是使用$(i+a, j+b)$，而是使用差值。然而，这种区别是表面的，因为我们总是可以匹配公式3和公式6之间的符号。我们在公式3中的原始定义更正确地描述了**互相关**（cross-correlation）。

### 通道

然而这种方法有一个问题：我们忽略了图像一般包含三个通道/三种原色（红色、绿色和蓝色）。实际上，图像不是二维张量，而是一个由高度、宽度和颜色组成的**三维张量**，比如包含$1024 \times 1024 \times 3$个像素。前两个轴与像素的空间位置有关，而第三个轴可以看作每个像素的多维表示。因此，我们将$\mathsf{X}$索引为$[\mathsf{X}]_ {i, j, k}$。由此卷积相应地调整为$[\mathsf{V}]_ {a,b,c}$，而不是$[\mathbf{V}]_ {a,b}$。

此外，由于输入图像是三维的，我们的隐藏表示$\mathsf{H}$也最好采用三维张量。换句话说，对于每一个空间位置，我们想要采用一组而不是一个隐藏表示。这样一组隐藏表示可以想象成一些互相堆叠的二维网格。因此，我们可以把隐藏表示想象为一系列具有二维张量的**通道**（channel）。这些通道有时也被称为**特征映射**（feature maps），因为每个通道都向后续层提供一组空间化的学习特征。直观上可以想象在靠近输入的底层，一些通道专门识别边缘，而一些通道专门识别纹理。

为了支持输入$\mathsf{X}$和隐藏表示$\mathsf{H}$中的多个通道，我们可以在$\mathsf{V}$中添加第四个坐标，即$[\mathsf{V}]_ {a, b, c, d}$。综上所述，
$$
[\mathsf{H}]_ {i,j,d} = u + \sum_ {a = -\Delta}^{\Delta} \sum_ {b = -\Delta}^{\Delta} \sum_ c [\mathsf{V}]_ {a, b, c, d} [\mathsf{X}]_ {i+a, j+b, c}
$$
其中隐藏表示$\mathsf{H}$中的索引$d$表示输出通道，而随后的输出将继续以三维张量$\mathsf{H}$作为输入进入下一个卷积层。所以，上述公式可以定义具有多个通道的卷积层，而其中$\mathsf{V}$是该卷积层的权重。

然而，仍有许多问题亟待解决。例如，图像中是否到处都有存在目标物体的可能？如何有效地计算输出层？如何选择适当的激活函数？为了训练有效的网络，如何做出合理的网络设计选择？我们将在本章的其它部分讨论这些问题。

## 图像卷积

### 互相关运算

严格来说，卷积层是个错误的叫法，因为它所表达的运算其实是**互相关运算**（cross-correlation），而不是卷积运算。根据上一节中的描述，在卷积层中，输入张量和核张量通过互相关运算产生输出张量。

首先，暂时忽略通道（第三维）这一情况，看看如何处理二维图像数据和隐藏表示。在下图中，输入是高度为$3$、宽度为$3$的二维张量（即形状为$3 \times 3$）。卷积核的高度和宽度都是$2$，而卷积核窗口（或卷积窗口）的形状由内核的高度和宽度决定（即$2 \times 2$），Input中的蓝色框表示卷积窗口。

![Figure 2-1 二维互相关运算](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/correlation.svg)

在二维互相关运算中，卷积窗口从输入张量的左上角开始，从左到右、从上到下滑动。当卷积窗口滑动到新一个位置时，包含在该窗口中的部分张量与卷积核张量进行按元素相乘，得到的张量再求和得到一个单一的标量值，由此得出了这一位置的输出张量值。在如上例子中，输出张量的四个元素由二维互相关运算得到，这个输出高度为$2$、宽度为$2$，如下所示：
$$
0\times0+1\times1+3\times2+4\times3=19,\\\
1\times0+2\times1+4\times2+5\times3=25,\\\
3\times0+4\times1+6\times2+7\times3=37,\\\
4\times0+5\times1+7\times2+8\times3=43.
$$
注意，输出大小略小于输入大小。这是因为卷积核的宽度和高度大于1，而卷积核只与图像中每个大小完全适合的位置进行互相关运算。所以，输出大小等于输入大小$n_h \times n_w$减去卷积核大小$k_h \times k_w$，即：$(n_h-k_h+1) \times (n_w-k_w+1)$

这是因为我们需要足够的空间在图像上“移动”卷积核。稍后，我们将看到如何通过在图像边界周围填充零来保证有足够的空间移动卷积核，从而保持输出大小不变。接下来，我们在`corr2d`函数中实现如上过程，该函数接受输入张量`X`和卷积核张量`K`，并返回输出张量`Y`。

```python
import torch
from torch import nn
from d2l import torch as d2l

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape # 卷积核的高和宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
  
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)

output: tensor([[19., 25.],
                [37., 43.]])
```

### 卷积层

卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。就像之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。

基于上面定义的`corr2d`函数**实现二维卷积层**。在`__init__`构造函数中，将`weight`和`bias`声明为两个模型参数。前向传播函数调用`corr2d`函数并添加偏置。

```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

高度和宽度分别为$h$和$w$的卷积核可以被称为$h \times w$卷积或$h \times w$卷积核，同样也将带有$h \times w$卷积核的卷积层称为$h \times w$卷积层。

### 图像中目标的边缘监测

下面时卷积层的一个简单应用：通过找到像素变化的位置，来检测图像中不同颜色的边缘。首先，我们一个$6\times 8$像素的黑白图像，中间四列为黑色（$0$），其余像素为白色（$1$）。

```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
output: tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
                [1., 1., 0., 0., 0., 0., 1., 1.],
                [1., 1., 0., 0., 0., 0., 1., 1.],
                [1., 1., 0., 0., 0., 0., 1., 1.],
                [1., 1., 0., 0., 0., 0., 1., 1.],
                [1., 1., 0., 0., 0., 0., 1., 1.]])
```

接下来，构造一个高度为$1$、宽度为$2$的卷积核`K`。当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零。

```python
K = torch.tensor([[1.0, -1.0]])
```

现在，对参数`X`（输入）和`K`（卷积核）执行互相关运算。如下所示，输出`Y`中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为$0$。

```python
Y = corr2d(X, K)
Y
output: tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
                [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
```

现在将输入的二维图像转置，再进行如上的互相关运算。其输出如下，之前检测到的垂直边缘消失了。不出所料，这个卷积核`K`只可以检测垂直边缘，无法检测水平边缘。

```python
corr2d(X.t(), K)
output: tensor([[0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.]])
```

### 卷积核

如果我们只需寻找黑白边缘，那么以上`[1, -1]`的边缘检测器足以。然而，当有了更复杂数值的卷积核，或者连续的卷积层时，我们不可能手动设计滤波器。那么就需要学习由`X`生成`Y`的卷积核。

下面将检查是否可以通过仅查看*“输入-输出”对*来学习由`X`生成`Y`的卷积核。首先构造一个卷积层，并将其卷积核初始化为随机张量。接下来，在每次迭代中，我们比较`Y`与卷积层输出的平方误差，然后计算梯度来更新卷积核。为了简单起见，我们在此使用内置的二维卷积层，并忽略偏置。

```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False) # 前两个参数为输入通道数和输出通道数

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
        
output: 
epoch 2, loss 7.253
epoch 4, loss 1.225
epoch 6, loss 0.209
epoch 8, loss 0.036
epoch 10, loss 0.007
```

在$10$次迭代之后，误差已经降到足够低，下面来看看我们所学的卷积核的权重张量。

```python
conv2d.weight.data.reshape((1, 2))
output: tensor([[ 0.9829, -0.9892]])
```

可以发现上面学习到的卷积核权重非常接近我们之前定义的卷积核`K`。

### 互相关和卷积

回想一下我们在上一节中观察到的互相关和卷积运算之间的对应关系。为了得到正式的**卷积**运算输出，我们需要执行上一节中严格定义的卷积运算，而不是互相关运算。幸运的是，它们差别不大，我们只需水平和垂直翻转二维卷积核张量，然后对输入张量执行**互相关**运算。

值得注意的是，由于卷积核是从数据中学习到的，因此无论这些层执行严格的卷积运算还是互相关运算，卷积层的输出都不会受到影响。为了说明这一点，假设卷积层执行互相关运算并学习Figure 2-1中的卷积核，该卷积核在这里由矩阵$\mathbf{K}$表示。假设其他条件不变，当这个层执行严格的卷积时，学习的卷积核$\mathbf{K}'$在水平和垂直翻转之后将与$\mathbf{K}$相同。也就是说，当卷积层对Figure 2-1中的输入和$\mathbf{K}'$执行严格卷积运算时，将得到与互相关运算中Figure 2-1相同的输出。

为了与深度学习文献中的标准术语保持一致，我们将继续把“互相关运算”称为卷积运算，尽管严格地说，它们略有不同。此外，对于卷积核张量上的权重，我们称其为**元素**。

### 特征映射和感受野

如在上一节中所述，Figure 2-1中输出的卷积层有时被称为**特征映射**（feature map），因为它可以被视为一个输入映射到下一层的空间维度的转换器。在卷积神经网络中，对于某一层的任意元素$x$，其**感受野**（receptive field）是指在前向传播期间可能影响$x$计算的所有元素（来自所有先前层）。

请注意，感受野可能大于输入的实际大小。用Figure 2-1为例来解释感受野：给定$2 \times 2$卷积核，阴影输出元素值$19$的感受野是输入阴影部分的四个元素。假设之前输出为$\mathbf{Y}$，其大小为$2 \times 2$，现在我们在其后附加一个卷积层，该卷积层以$\mathbf{Y}$为输入，输出单个元素$z$。在这种情况下，$\mathbf{Y}$上的$z$的感受野包括$\mathbf{Y}$的所有四个元素，而输入的感受野包括最初所有九个输入元素。因此，当一个特征图中的任意元素需要检测更广区域的输入特征时，我们可以构建一个更深的网络。

## 填充和步幅

### 填充

在应用多层卷积时，常常丢失边缘像素。由于我们通常使用小卷积核，因此对于任何单个卷积，我们可能只会丢失几个像素。但随着我们应用许多连续卷积层，累积丢失的像素数就多了。解决这个问题的简单方法即为*填充*（padding）：在输入图像的边界填充元素（通常填充元素是$0$）。例如，在下图中，我们将$3 \times 3$输入填充到$5 \times 5$，那么它的输出就增加为$4 \times 4$。阴影部分是第一个输出元素以及用于输出计算的输入和核张量元素：$0\times0+0\times1+0\times2+0\times3=0$。

![Figure 3-1 带填充的二维互相关](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/conv-pad.svg)

通常，如果我们添加$p_h$行填充（大约一半在顶部，一半在底部）和$p_w$列填充（左侧大约一半，右侧一半），则输出形状将为

$$
(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1)
$$
这意味着输出的高度和宽度将分别增加$p_h$和$p_w$。在许多情况下，我们需要设置$p_h=k_h-1$和$p_w=k_w-1$，使输入和输出具有相同的高度和宽度。这样可以在构建网络时更容易地预测每个图层的输出形状。假设$k_h$是奇数，我们将在高度的两侧填充$p_h/2$行。如果$k_h$是偶数，则一种可能性是在输入顶部填充$\lceil p_h/2\rceil$行，在底部填充$\lfloor p_h/2\rfloor$行。同理，我们填充宽度的两侧。

卷积神经网络中卷积核的高度和宽度通常为奇数，例如1、3、5或7。选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。

此外，使用奇数的核大小和填充大小也提供了书写上的便利。对于任何二维张量`X`，当满足：
1. 卷积核的大小是奇数；
2. 所有边的填充行数和列数相同；
3. 输出与输入具有相同高度和宽度
则可以得出：输出`Y[i, j]`是通过以输入`X[i, j]`为中心，与卷积核进行互相关计算得到的。

比如，在下面的例子中，我们创建一个高度和宽度为3的二维卷积层，并在所有侧边填充1个像素。给定高度和宽度为8的输入，则输出的高度和宽度也是8。

```python
import torch
from torch import nn


# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape

output: torch.Size([8, 8])
```

当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度。在如下示例中，我们使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

output: torch.Size([8, 8])
```

### 步幅

在计算互相关时，卷积窗口从输入张量的左上角开始，向下、向右滑动。在前面的例子中默认每次滑动一个元素。但是，有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。

这样每次滑动元素的数量称为**步幅**（stride），如下图是垂直步幅为$3$，水平步幅为$2$的二维互相关运算。着色部分是输出元素以及用于输出计算的输入和内核张量元素：$0\times0+0\times1+1\times2+2\times3=8$、$0\times0+6\times1+0\times2+0\times3=6$。可以看到，为了计算输出中第一列的第二个元素和第一行的第二个元素，卷积窗口分别向下滑动3行和向右滑动2列。但是，当卷积窗口继续向右滑动两列时，没有输出，因为输入元素无法填充窗口（除非我们添加另一列填充）。

![Figure 3-2 垂直步幅为，水平步幅为的二维互相关运算](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/conv-stride.svg)

通常，当垂直步幅为$s_h$、水平步幅为$s_w$时，输出形状为
$$
\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor
$$
如果我们设置了$p_h=k_h-1$和$p_w=k_w-1$，则输出形状将简化为$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$。更进一步，如果输入的高度和宽度可以被垂直和水平步幅整除，则输出形状将为$(n_h/s_h) \times (n_w/s_w)$。

下面，将高度和宽度的步幅设置为2，从而将输入的高度和宽度减半。

```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
output: torch.Size([4, 4])
```

若行和列的填充和步幅都不同：

```python
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
output: torch.Size([2, 2])
```

为了简洁起见，当输入高度和宽度两侧的填充数量分别为$p_h$和$p_w$时，称之为填充$(p_h, p_w)$，当$p_h = p_w = p$时，填充是$p$。同理，当高度和宽度上的步幅分别为$s_h$和$s_w$时，称之为步幅$(s_h, s_w)$，当$s_h = s_w = s$时，称步幅为$s$。默认情况下，填充为0，步幅为1。在实践中很少使用不一致的步幅或填充，也就是说通常有$p_h = p_w$和$s_h = s_w$。

## 多输入输出通道

### 多输入通道

当输入包含多个通道时，需要构造一个与输入数据具有相同输入通道数的卷积核，以便与输入数据进行互相关运算。假设输入的通道数为$c_i$，那么卷积核的输入通道数也需要为$c_i$。如果卷积核的窗口形状是$k_h\times k_w$，那么当$c_i=1$时，我们可以把卷积核看作形状为$k_h\times k_w$的二维张量。然而，当$c_i>1$时，我们卷积核的每个输入通道将包含形状为$k_h\times k_w$的张量。将这些张量$c_i$连结在一起可以得到形状为$c_i\times k_h\times k_w$的卷积核。由于输入和卷积核都有$c_i$个通道，我们可以对每个通道输入的二维张量和卷积核的二维张量进行互相关运算，再**对通道求和**（将$c_i$的结果相加）得到二维张量。这是多通道输入和多输入通道卷积核之间进行二维互相关运算的结果。

下图演示了一个具有两个输入通道的二维互相关运算的示例。阴影部分是第一个输出元素以及用于计算这个输出的输入和核张量元素：$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$。

![Figure 4-1 两个输入通道的互相关计算](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/conv-multi-in.svg)

下面将实现一下多输入通道互相关运算，简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。

```python
import torch
from d2l import torch as d2l

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

我们可以构造与上面图片中的值相对应的输入张量`X`和核张量`K`，以验证互相关运算的输出。

```python
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)

output: tensor([[ 56.,  72.],
                [104., 120.]])
```

### 多输出通道

到目前为止，不论有多少输入通道，还只实现了一个输出通道。然而，正如我们在之前所讨论的，每一层有多个输出通道是至关重要的。在最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度。直观地说，我们可以将每个通道看作对**不同特征**的响应。而现实可能更为复杂一些，因为每个通道不是独立学习的，而是为了共同使用而优化的。因此，多输出通道并不仅是学习多个单通道的检测器。

用$c_i$和$c_o$分别表示输入和输出通道的数目，并让$k_h$和$k_w$为卷积核的高度和宽度。为了获得多个通道的输出，我们可以为每个输出通道创建一个形状为$c_i\times k_h\times k_w$的卷积核张量，这样卷积核的形状是$c_o\times c_i\times k_h\times k_w$。在互相关运算中，每个输出通道先获取所有输入通道，再以对应该输出通道的卷积核计算出结果。

如下所示，实现一个计算多个通道的输出的互相关函数：

```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

通过将核张量`K`与`K+1`（`K`中每个元素加$1$）和`K+2`（`K`中每个元素加$2$）连接起来，构造了一个具有$3$个输出通道的卷积核。

```python
K = torch.stack((K, K + 1, K + 2), 0)
K.shape
output: torch.Size([3, 2, 2, 2])
```

下面，我们对输入张量`X`与卷积核张量`K`执行互相关运算。现在的输出包含$3$个通道，第一个通道的结果与先前输入张量`X`和多输入单输出通道的结果一致。

```python
corr2d_multi_in_out(X, K)
output:
tensor([[[ 56.,  72.],
         [104., 120.]],

        [[ 76., 100.],
         [148., 172.]],

        [[ 96., 128.],
         [192., 224.]]])
```

### $1\times 1$ 卷积层

$1 \times 1$卷积层，即$k_h = k_w = 1$，看起来似乎没有多大意义。毕竟，卷积的本质是有效提取相邻像素间的相关特征，而$1 \times 1$卷积显然没有此作用。尽管如此，$1 \times 1$仍然十分流行，经常包含在复杂深层网络的设计中。因为使用了最小窗口，$1\times 1$卷积失去了卷积层的特有能力——在高度和宽度维度上，识别相邻元素间相互作用的能力。其实$1\times 1$卷积的唯一计算发生在通道上，通常用于**调整网络层的通道数量和控制模型复杂性**。

下图展示了使用$1\times 1$卷积核与$3$个输入通道和$2$个输出通道的互相关计算。这里输入和输出具有相同的高度和宽度，输出中的每个元素都是从输入图像中同一位置的元素的线性组合。我们可以将$1\times 1$卷积层看作在每个像素位置应用的全连接层，以$c_i$个输入值转换为$c_o$个输出值。因为这仍然是一个卷积层，所以跨像素的权重是一致的。同时，$1\times 1$卷积层需要的权重维度为$c_o\times c_i$，再额外加上一个偏置。

![Figure 4-2 使用具有3个输入通道和2个输出通道的1×1卷积核](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/conv-1x1.svg)

下面，使用全连接层实现$1 \times 1$卷积：

```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```

当执行$1\times 1$卷积运算时，上述函数相当于先前实现的互相关函数`corr2d_multi_in_out`。下面用一些样本数据来验证这一点。

```python
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```

## 池化层

通常处理图像时，我们希望逐渐降低隐藏表示的空间分辨率、聚集信息，这样随着神经网络中层叠的上升，每个神经元对其敏感的感受野（输入）就越大。而机器学习任务通常会跟全局图像的问题有关（例如，“图像是否包含一只猫呢？”），所以最后一层的神经元应该对整个输入全局敏感。通过逐渐聚合信息，生成越来越粗糙的映射，最终实现学习全局表示的目标，同时将卷积图层的所有优势保留在中间层。

此外，当检测较底层的特征时（例如之前所讨论的边缘），我们通常希望这些特征保持某种程度上的平移不变性。例如，如果我们拍摄黑白之间轮廓清晰的图像`X`，并将整个图像向右移动一个像素，即`Z[i, j] = X[i, j + 1]`，则新图像`Z`的输出可能大不相同。而在现实中，随着拍摄角度的移动，任何物体几乎不可能发生在同一像素上。即使用三脚架拍摄一个静止的物体，由于快门的移动而引起的相机振动，可能会使所有物体左右移动一个像素。

下面将介绍**池化**（pooling）层，它具有双重目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。

### 最大池化层和平均池化层

与卷积层类似，池化层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动，为固定形状窗口（有时称为**池化窗口**）遍历的每个位置计算一个输出。然而，不同于卷积层中的输入与卷积核之间的互相关计算，池化层不包含参数。相反，池运算是确定性的，我们通常计算池化窗口中所有元素的最大值或平均值。这些操作分别称为**最大池化**（maximum pooling）和**平均池化**（average pooling）。

在这两种情况下，与互相关运算符一样，池化窗口从输入张量的左上角开始，从左往右、从上往下的在输入张量内滑动。在池化窗口到达的每个位置，它计算该窗口中输入子张量的最大值或平均值。计算最大值或平均值是取决于使用了最大池化层还是平均池化层。

![Figure5-1 最大池化层](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/pooling.svg)

上图中输出张量的高度为$2$，宽度为$2$。这四个元素为每个池化窗口中的最大值：
$$
\max(0, 1, 3, 4)=4,\\\
\max(1, 2, 4, 5)=5,\\\
\max(3, 4, 6, 7)=7,\\\
\max(4, 5, 7, 8)=8.\\
$$
池化窗口形状为$p \times q$的池化层称为$p \times q$池化层，池化操作称为$p \times q$池化。回到本节开头提到的对象边缘检测示例，现在我们将使用卷积层的输出作为$2\times 2$最大池化的输入。设置卷积层输入为`X`，池化层输出为`Y`。无论`X[i, j]`和`X[i, j + 1]`的值相同与否，或`X[i, j + 1]`和`X[i, j + 2]`的值相同与否，池化层始终输出`Y[i, j] = 1`。也就是说，使用$2\times 2$最大池化层，即使在高度或宽度上移动一个元素，卷积层仍然可以识别到模式。

在下面的代码中的`pool2d`函数，实现池化层的前向传播。这类似于第2节中的`corr2d`函数。然而，这里没有卷积核，输出为输入中每个区域的最大值或平均值。

```python
import torch
from torch import nn
from d2l import torch as d2l

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

构建输入张量`X`，验证二维最大池化层的输出：

```python
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
output: tensor([[4., 5.],
                [7., 8.]])
```

此外还可以验证平均池化层：

```python
pool2d(X, (2, 2), 'avg')
output: tensor([[2., 3.],
                [5., 6.]])
```

### 填充和步幅

与卷积层一样，池化层也可以改变输出形状。和以前一样，我们可以通过填充和步幅以获得所需的输出形状。下面用深度学习框架中内置的二维最大池化层，来演示池化层中填充和步幅的使用。

首先构造了一个输入张量`X`，它有四个维度，其中样本数和通道数都是1：

```python
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X
output:
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]]]])
```

默认情况下，深度学习框架中的步幅与池化窗口的大小相同。因此，如果我们使用形状为`(3, 3)`的池化窗口，那么默认情况下，我们得到的步幅形状为`(3, 3)`。

```python
pool2d = nn.MaxPool2d(3)
pool2d(X)
output: tensor([[[[10.]]]])
```

填充和步幅可以手动设定：

```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
output: tensor([[[[ 5.,  7.],
                  [13., 15.]]]])
```

当然，也可以设定一个任意大小的矩形池化窗口，并分别设定填充和步幅的高度和宽度：

```python
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
output: tensor([[[[ 5.,  7.],
                  [13., 15.]]]])
```

### 多通道

在处理多通道输入数据时，池化层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。这意味着池化层的输出通道数与输入通道数相同。下面，将在通道维度上连结张量`X`和`X + 1`，以构建具有2个通道的输入。

```python
X = torch.cat((X, X + 1), 1)
X
output:
tensor([[[[ 0.,  1.,  2.,  3.],
          [ 4.,  5.,  6.,  7.],
          [ 8.,  9., 10., 11.],
          [12., 13., 14., 15.]],

         [[ 1.,  2.,  3.,  4.],
          [ 5.,  6.,  7.,  8.],
          [ 9., 10., 11., 12.],
          [13., 14., 15., 16.]]]])
```

如下所示，池化后输出通道的数量仍然是2。

```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
output: 
tensor([[[[ 5.,  7.],
          [13., 15.]],

         [[ 6.,  8.],
          [14., 16.]]]])
```

## 卷积神经网络(LeNet)

### LeNet

总体来看，LeNet（LeNet-5）由两个部分组成：

- 卷积编码器：由两个卷积层组成

- 全连接层密集块：由三个全连接层组成

该架构如下图所示：

![Figure 6-1 LeNet中的数据流](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/lenet.svg)

每个卷积块中的基本单元是一个卷积层、一个sigmoid激活函数和平均池化层。请注意，虽然ReLU和最大池化层更有效，但它们在20世纪90年代还没有出现。每个卷积层使用$5\times 5$卷积核和一个sigmoid激活函数。这些层将输入映射到多个二维特征输出，同时增加通道的数量。第一卷积层有6个输出通道，而第二个卷积层有16个输出通道。每个$2\times2$池操作（步幅2）通过空间下采样将维数减少4倍。卷积的输出形状由批量大小、通道数、高度、宽度决定。

为了将卷积块的输出传递给稠密块，我们必须在小批量中展平每个样本。换言之，我们将这个四维输入转换成全连接层所期望的二维输入。这里的二维表示的第一个维度是小批量的样本数，第二个维度给出每个样本的平面向量表示。LeNet的稠密块有三个全连接层，分别有120、84和10个输出。因为在执行分类任务，所以输出层的10维对应于最后输出结果的数量。

通过下面的LeNet代码，可以看出用深度学习框架实现此类模型非常简单。我们只需要实例化一个`Sequential`块并将需要的层连接在一起。

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

我们对原始模型做了一点小改动，去掉了最后一层的高斯激活，网络其他部分与最初的LeNet-5一致。下面，我们将一个大小为$28 \times 28$的单通道（黑白）图像输入LeNet。通过在每一层打印输出的形状，我们可以检查模型，以确保其操作与我们期望的一致。

![Figure 6-2 LeNet的简化版](https://zh-v2.d2l.ai/_images/lenet-vert.svg)

```python
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)
    
output:
Conv2d output shape: 	 torch.Size([1, 6, 28, 28])
Sigmoid output shape: 	 torch.Size([1, 6, 28, 28])
AvgPool2d output shape: 	 torch.Size([1, 6, 14, 14])
Conv2d output shape: 	 torch.Size([1, 16, 10, 10])
Sigmoid output shape: 	 torch.Size([1, 16, 10, 10])
AvgPool2d output shape: 	 torch.Size([1, 16, 5, 5])
Flatten output shape: 	 torch.Size([1, 400])
Linear output shape: 	 torch.Size([1, 120])
Sigmoid output shape: 	 torch.Size([1, 120])
Linear output shape: 	 torch.Size([1, 84])
Sigmoid output shape: 	 torch.Size([1, 84])
Linear output shape: 	 torch.Size([1, 10])
```

请注意，在整个卷积块中，与上一层相比，每一层特征的高度和宽度都减小了。第一个卷积层使用2个像素的填充，来补偿$5 \times 5$卷积核导致的特征减少。相反，第二个卷积层没有填充，因此高度和宽度都减少了4个像素。随着层叠的上升，通道的数量从输入时的1个，增加到第一个卷积层之后的6个，再到第二个卷积层之后的16个。同时，每个池化层的高度和宽度都减半。最后，每个全连接层减少维数，最终输出一个维数与结果分类数相匹配的输出。

### 模型训练

现在已经实现了LeNet，让我们看看LeNet在Fashion-MNIST数据集上的表现。

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

虽然卷积神经网络的参数较少，但与深度的多层感知机相比，它们的计算成本仍然很高，因为每个参数都参与更多的乘法。通过使用GPU，可以用它加快训练。

为了进行评估，我们需要对 [3.6节]中描述的`evaluate_accuracy`函数进行轻微的修改。由于完整的数据集位于内存中，因此在模型使用GPU计算数据集之前，我们需要将其复制到显存中。

```python
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

为了使用GPU，还需要一点小改动。与 [3.6节]中定义的`train_epoch_ch3`不同，在进行正向和反向传播之前，我们需要将每一小批量数据移动到我们指定的设备（例如GPU）上。

如下所示，训练函数`train_ch6`也类似于 [3.6节]中定义的`train_ch3`。由于我们将实现多层神经网络，因此我们将主要使用高级API。以下训练函数假定从高级API创建的模型作为输入，并进行相应的优化。我们使用Xavier随机初始化模型参数。与全连接层一样，我们使用交叉熵损失函数和小批量随机梯度下降。

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

下面训练和评估LeNet-5模型：

```python
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

output: 
loss 0.466, train acc 0.824, test acc 0.800
34835.7 examples/sec on cuda:0
```

![Figure 6-3 运行结果](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231103163251831.png)

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/6-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/  

