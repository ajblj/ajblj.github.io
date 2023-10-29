# 4 多层感知机


## 多层感知机

### 隐藏层

如果我们的标签通过仿射变换后确实与我们的输入数据相关，那么这种方法确实足够了。 但是，仿射变换中的*线性*是一个很强的假设。线性意味着单调假设： 任何特征的增大都会导致模型输出的增大（如果对应的权重为正）， 或者导致模型输出的减小（如果对应的权重为负）。对线性模型的依赖对应于一个隐含的假设， 即区分猫和狗的唯一要求是评估单个像素的强度。 对于深度神经网络，需要使用观测数据来联合学习隐藏层表示和应用于该表示的线性预测器。

可以通过在网络中加入一个或多个隐藏层来克服线性模型的限制， 使其能处理更普遍的函数关系类型。要做到这一点，最简单的方法是将许多全连接层堆叠在一起。每一层都输出到上面的层，直到生成最后的输出。可以把前$L-1$层看作**表示**，把最后一层看作**线性预测器**。这种架构通常称为**多层感知机**（multilayer perceptron），通常缩写为**MLP**。如下图：

<img src="http://d2l.ai/_images/mlp.svg" alt="一个单隐藏层的多层感知机，具有5个隐藏单元" style="zoom:100%;" />

这个多层感知机有4个输入，3个输出，其隐藏层包含5个隐藏单元。输入层不涉及任何计算，因此使用此网络产生输出只需要实现隐藏层和输出层的计算。因此，这个多层感知机中的层数为2。这两个层都是全连接的，每个输入都会影响隐藏层中的每个神经元，而隐藏层中的每个神经元又会影响输出层中的每个神经元。

然而，具有全连接层的多层感知机的参数开销可能会高得令人望而却步。即使在不改变输入或输出大小的情况下，可能在参数节约和模型有效性之间进行权衡

我们通过矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$来表示$n$个样本的小批量，其中每个样本具有$d$个输入特征。对于具有$h$个隐藏单元的单隐藏层多层感知机，用$\mathbf{H} \in \mathbb{R}^{n \times h}$表示隐藏层的输出，称为**隐藏表示**（hidden representations）。在数学或代码中，$\mathbf{H}$也被称为**隐藏层变量**（hidden-layer variable）或**隐藏变量**（hidden variable）。因为隐藏层和输出层都是全连接的，所以有隐藏层权重$\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$和隐藏层偏置$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$以及输出层权重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$和输出层偏置$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$。形式上，我们按如下方式计算单隐藏层多层感知机的输出$\mathbf{O} \in \mathbb{R}^{n \times q}$：
$$
\begin{aligned}
  \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
  \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$
在添加隐藏层之后，模型现在需要跟踪和更新额外的参数。可我们能从中得到什么好处呢？在上面定义的模型里，我们没有好处！原因很简单：上面的隐藏单元由输入的仿射函数给出，而输出（softmax操作前）只是隐藏单元的仿射函数。仿射函数的仿射函数本身就是仿射函数，但是我们之前的线性模型已经能够表示任何仿射函数。

可以证明这一等价性，即对于任意权重值，我们只需合并隐藏层，便可产生具有参数$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$和$\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$的等价单层模型：
$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$
为了发挥多层架构的潜力，我们还需要一个额外的关键要素：在仿射变换之后对每个隐藏单元应用非线性的**激活函数**（activation function）$\sigma$。激活函数的输出（例如，$\sigma(\cdot)$）被称为**活性值**（activations）。一般来说，有了激活函数，就不可能再将多层感知机退化成线性模型：
$$
\begin{aligned}
  \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\\
  \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\\
\end{aligned}
$$
由于$\mathbf{X}$中的每一行对应于小批量中的一个样本，出于记号习惯的考量，我们定义非线性函数$\sigma$也以按行的方式作用于其输入，即一次计算一个样本。但是本节应用于隐藏层的激活函数通常不仅按行操作，也按元素操作。这意味着在计算每一层的线性部分之后，可以计算每个活性值，而不需要查看其他隐藏单元所取的值。对于大多数激活函数都是这样。

为了构建更通用的多层感知机，可以继续堆叠这样的隐藏层，例如$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$和$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$，一层叠一层，从而产生更有表达能力的模型。

虽然一个单隐层网络能学习任何函数， 但并不意味着应该尝试使用单隐藏层网络来解决所有问题。 事实上，通过使用更深（而不是更广）的网络，可以更容易地逼近许多函数。

### 激活函数

*激活函数*（activation function）通过**计算加权和并加上偏置**来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算。 大多数激活函数都是非线性的。

#### ReLU函数

最受欢迎的激活函数是*修正线性单元*（Rectified linear unit，*ReLU*）， 因为它实现简单，同时在各种预测任务中表现良好。 ReLU提供了一种非常简单的非线性变换。 给定元素$0$，ReLU函数被定义为该元素与0的最大值：
$$
\operatorname{ReLU}(x) = \max(x, 0)
$$
ReLU函数通过将相应的活性值设为0，仅保留正元素并丢弃所有负元素。

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_mlp_76f463_21_0.svg" alt="ReLu函数" style="zoom:100%;" />

当输入为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。

```python
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_mlp_76f463_36_0.svg" alt="ReLu函数导数" style="zoom:100%;" />

使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。 这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题。

ReLU函数有许多变体，包括*参数化ReLU*（Parameterized ReLU，*pReLU*）函数。 该变体为ReLU添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过：
$$
\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x)
$$

#### sigmoid函数

对于一个定义域在$\mathbb{R}$中的输入，**sigmoid函数**将输入变换为区间(0, 1)上的输出。因此，sigmoid通常称为**挤压函数**（squashing function）：它将范围（-inf, inf）中的任意输入压缩到区间（0, 1）中的某个值：
$$
\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}
$$
当人们逐渐关注到到基于梯度的学习时，sigmoid函数是一个自然的选择，因为它是一个平滑的、可微的阈值单元近似。当想要将输出视作二元分类问题的概率时，sigmoid仍然被广泛用作**输出单元**上的激活函数，然而，sigmoid在隐藏层中已经较少使用，它在大部分时候被更简单、更容易训练的ReLU所取代。

下面绘制sigmoid函数，输入接近0时，sigmoid函数接近线性变换。

```python
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_mlp_76f463_51_0.svg" alt="sigmoid函数" style="zoom:100%;" />

sigmoid函数的导数为下面的公式：
$$
\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right)
$$
sigmoid函数的导数图像如下所示。 当输入为0时，sigmoid函数的导数达到最大值0.25； 而输入在任一方向上越远离0点时，导数越接近0。

```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_mlp_76f463_66_0.svg" alt="sigmoid函数导数" style="zoom:100%;" />

#### tanh函数

与sigmoid函数类似， tanh(双曲正切)函数也能将其输入压缩转换到区间(-1, 1)上。 tanh函数的公式如下：
$$
\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}
$$
当输入在0附近时，tanh函数接近线性变换。函数的形状类似于sigmoid函数，不同的是tanh函数关于坐标系原点中心对称。

```python
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_mlp_76f463_81_0.svg" alt="tanh函数" style="zoom:100%;" />

tanh函数的导数是：
$$
\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x)
$$
当输入接近0时，tanh函数的导数接近最大值1。 与我们在sigmoid函数图像中看到的类似， 输入在任一方向上越远离0点，导数越接近0。

```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_mlp_76f463_96_0.svg" alt="tanh函数导数" style="zoom:100%;" />

 

## 实现多层感知机

### 读取数据

```python
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

### 初始化模型参数

Fashion-MNIST中的每个图像由$28 \times 28 = 784$个灰度像素值组成。所有图像共分为10个类别。忽略像素之间的空间结构，我们可以将每个图像视为具有784个输入特征和10个类的简单分类数据集。首先，将实现一个具有单隐藏层的多层感知机，它包含256个隐藏单元。通常，我们选择2的若干次幂作为层的宽度。因为内存在硬件中的分配和寻址方式，这么做往往可以在计算上更高效。

```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

### 激活函数

这里实现ReLU激活函数， 而不是直接调用内置的`relu`函数。

```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

### 模型

由于忽略了空间结构， 所以使用`reshape`将每个二维图像转换为一个长度为`num_inputs`的向量。

```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
```

### 损失函数

这里使用高级API中的内置函数来计算softmax和交叉熵损失。

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

### 训练

多层感知机的训练过程与softmax回归的训练过程完全相同。 可以直接调用`d2l`包的`train_ch3`函数， 将迭代周期数设置为10，并将学习率设置为0.1。

```python
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

<img src="https://zh.d2l.ai/_images/output_mlp-scratch_106d07_81_0.svg" alt="多层感知机模型训练结果" style="zoom:100%;" />

## 使用深度学习框架简洁实现多层感知机

与softmax回归的简洁实现相比， 唯一的区别是添加了2个全连接层（之前只添加了1个全连接层）。 第一层是隐藏层，它包含256个隐藏单元，并使用了ReLU激活函数，第二层是输出层。

```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

训练过程的实现与实现softmax回归时完全相同。

```python
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231016181236034.png" alt="image-20231016181236034" style="zoom: 60%;" />

## 模型选择、欠拟合和过拟合

我们的目标是发现某些模式， 这些模式捕捉到了我们训练集潜在总体的规律。 如果成功做到了这点，即使是对以前从未遇到过的个体， 模型也可以成功地评估风险。 如何发现可以泛化的模式是机器学习的根本问题。将模型在训练数据上拟合的比在潜在分布中更接近的现象称为*过拟合*（overfitting）， 用于对抗过拟合的技术称为*正则化*（regularization）。

### 训练误差和泛化误差

*训练误差*（training error）是指， 模型在**训练数据集**上计算得到的误差。 *泛化误差*（generalization error）是指， 模型应用在同样从原始样本的分布中抽取的无限多数据样本时，**模型误差的期望**。在实际中，我们只能通过将模型应用于一个独立的测试集来估计泛化误差， 该测试集由随机选取的、未曾在训练集中出现的数据样本构成。

#### 模型复杂性

本节为了给出一些直观的印象，将重点介绍几个倾向于影响模型泛化的因素。

1. 可调整参数的数量。当可调整参数的数量（有时称为*自由度*）很大时，模型往往更容易过拟合。
2. 参数采用的值。当权重的取值范围较大时，模型可能更容易过拟合。
3. 训练样本的数量。即使模型很简单，也很容易过拟合只包含一两个样本的数据集。而过拟合一个有数百万个样本的数据集则需要一个极其灵活的模型。

### 模型选择

在机器学习中，我们通常在评估几个候选模型后选择最终的模型，这个过程叫做*模型选择*。 有时需要进行比较的模型在本质上是完全不同的（比如，决策树与线性模型）；又有时，我们需要比较不同的超参数设置下的同一类模型，例如，训练多层感知机模型时，我们可能希望比较具有不同数量的隐藏层、不同数量的隐藏单元以及不同的激活函数组合的模型。为了确定候选模型中的最佳模型，我们通常会使用验证集。

#### 验证集

原则上，在确定所有的超参数之前，我们不希望用到测试集。 如果我们在模型选择过程中使用测试数据，可能会有过拟合测试数据的风险。如果过拟合了训练数据，还可以在测试数据上的评估来判断过拟合。 但是如果过拟合了测试数据，就无法知道了。因此，决不能依靠测试数据进行模型选择。 然而，也不能仅仅依靠训练数据来选择模型，因为我们无法估计训练数据的泛化误差。

解决此问题的常见做法是将数据分成三份，除了训练和测试数据集之外，还增加一个*验证数据集*（validation dataset），也叫*验证集*（validation set）。

#### K折交叉验证

当训练数据稀缺时，甚至可能无法提供足够的数据来构成一个合适的验证集。这个问题的一个流行的解决方案是采用$K$**折交叉验证**。这里，原始训练数据被分成$K$个不重叠的子集。然后执行$K$次模型训练和验证，每次在$K-1$个子集上进行训练，并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证。最后，通过对$K$次实验的结果取平均来估计训练和验证误差。

### 欠拟合和过拟合

当比较训练和验证误差时，要注意两种常见的情况。 ①训练误差和验证误差都很严重， 但它们之间仅有一点差距。 如果模型不能降低训练误差，这可能意味着模型过于简单（即表达能力不足），无法捕获试图学习的模式。此外，由于训练和验证误差之间的*泛化误差*很小， 我们有理由相信可以用一个更复杂的模型降低训练误差。 这种现象被称为*欠拟合*（underfitting）；②当训练误差明显低于验证误差时表明严重的*过拟合*（overfitting）。 

注意，*过拟合*并不总是一件坏事。 特别是在深度学习领域，众所周知， 最好的预测模型在训练数据上的表现往往比在保留（验证）数据上好得多。 最终通常更关心验证误差，而不是训练误差和验证误差之间的差距。

是否过拟合或欠拟合可能取决于模型复杂性和可用训练数据集的大小。

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231017202333566.png" alt="image-20231017202333566" style="zoom:50%;" />

#### 模型复杂度

例子：给定由单个特征$x$和对应实数标签$y$组成的训练数据，试图找到下面的$d$阶多项式来估计标签$y$。
$$
\hat{y}= \sum_{i=0}^d x^i w_i
$$
这只是一个线性回归问题，特征是$x$的幂给出的，模型的权重是$w_i$给出的，偏置是$w_0$给出的（因为对于所有的$x$都有$x^0 = 1$）。

由于这只是一个线性回归问题，我们可以使用平方误差作为我们的损失函数。高阶多项式函数比低阶多项式函数复杂得多。高阶多项式的参数较多，模型函数的选择范围较广。因此在固定训练数据集的情况下，高阶多项式函数相对于低阶多项式的训练误差应该始终更低（最坏也是相等）。事实上，当数据样本包含了$x$的不同值时，函数阶数等于数据样本数量的多项式函数可以完美拟合训练集。

如下图直观地描述了多项式的阶数和欠拟合与过拟合之间的关系。

<img src="https://zh.d2l.ai/_images/capacity-vs-error.svg" alt="模型复杂度对欠拟合和过拟合的影响" style="zoom:100%;" />

#### 数据集大小

另一个重要因素是数据集的大小。 训练数据集中的样本越少，我们就越有可能（且更严重地）过拟合。 随着训练数据量的增加，泛化误差通常会减小。一般来说，更多的数据不会有什么坏处。对于固定的任务和数据分布，模型复杂性和数据集大小之间通常存在关系。给出更多的数据，可能会尝试拟合一个更复杂的模型。能够拟合更复杂的模型可能是有益的。如果没有足够的数据，简单的模型可能更有用。对于许多任务，深度学习只有在有数千个训练样本时才优于线性模型。

### 多项式回归

#### 生成数据集

给定$x$，下面将使用以下三阶多项式来生成训练和测试数据的标签：
$$
y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2)
$$
噪声项$\epsilon$服从均值为0且标准差为0.1的正态分布。在优化的过程中，通常希望避免非常大的梯度值或损失值。这就是将特征从$x^i$调整为$\frac{x^i}{i!}$的原因，这样可以避免很大的$i$带来的特别大的指数值。为训练集和测试集各生成100个样本。

```python
max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) # 求x的幂
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

存储在`poly_features`中的单项式由gamma函数重新缩放，其中$\Gamma(n)=(n-1)!$。

```python
# NumPy ndarray转换为tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=
    torch.float32) for x in [true_w, features, poly_features, labels]]

features[:2], poly_features[:2, :], labels[:2]
output: (tensor([[0.2680],
                 [0.0072]]),
         tensor([[1.0000e+00, 2.6800e-01, 3.5911e-02, 3.2080e-03, 2.1493e-04, 1.1520e-05,
                  5.1456e-07, 1.9700e-08, 6.5994e-10, 1.9651e-11, 5.2664e-13, 1.2831e-14,
                  2.8655e-16, 5.9072e-18, 1.1308e-19, 2.0203e-21, 3.3840e-23, 5.3346e-25,
                  7.9426e-27, 1.1203e-28],
                 [1.0000e+00, 7.1886e-03, 2.5838e-05, 6.1913e-08, 1.1127e-10, 1.5997e-13,
                  1.9166e-16, 1.9682e-19, 1.7686e-22, 1.4126e-25, 1.0155e-28, 6.6363e-32,
                  3.9755e-35, 2.1983e-38, 1.1287e-41, 5.6052e-45, 0.0000e+00, 0.0000e+00,
                  0.0000e+00, 0.0000e+00]]),
         tensor([5.1112, 4.9042]))
```

#### 训练和测试

实现一个函数来评估模型在给定数据集上的损失。

```python
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]
```

定义训练函数

```python
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

#### 三阶多项式函数-正常拟合

首先使用三阶多项式函数，它与数据生成函数的阶数相同。结果表明，该模型能有效降低训练损失和测试损失。学习到的模型参数也接近真实值$w = [5, 1.2, -3.4, 5.6]$。

```python
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
output: weight: [[ 4.994724   1.2322158 -3.36956    5.5164495]]
```

<img src="https://zh.d2l.ai/_images/output_underfit-overfit_ec26bd_81_1.svg" alt="正常拟合损失" style="zoom:100%;" />

#### 线性函数-欠拟合

再看线性函数拟合，减少该模型的训练损失相对困难。 在最后一个迭代周期完成后，训练损失仍然很高。 当用来拟合非线性模式（如这里的三阶多项式函数）时，线性模型容易欠拟合。

```python
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
output: weight: [[3.3262606 3.4666014]]
```

<img src="https://zh.d2l.ai/_images/output_underfit-overfit_ec26bd_96_1.svg" alt="欠拟合损失" style="zoom:100%;" />

#### 高阶多项式函数-过拟合

尝试使用一个阶数过高的多项式来训练模型。在这种情况下，没有足够的数据用于学到高阶系数应该具有接近于零的值。因此，这个过于复杂的模型会轻易受到训练数据中噪声的影响。虽然训练损失可以有效地降低，但测试损失仍然很高。结果表明，复杂模型对数据造成了过拟合。

```python
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)
output: weight: [[ 4.9849787   1.2896876  -3.2996354   5.145749   -0.34205326  1.2237961
                  0.20393135  0.3027379  -0.20079008 -0.16337848  0.11026663  0.21135856
                  -0.00940325  0.11873583 -0.15114897 -0.05347819  0.17096086  0.1863975
                  -0.09107699 -0.02123026]]
```

<img src="https://zh.d2l.ai/_images/output_underfit-overfit_ec26bd_111_1.svg" alt="过拟合损失" style="zoom:100%;" />

## 权重衰减

总是可以通过去收集更多的训练数据来缓解过拟合，但这可能成本很高，耗时颇多，或者完全超出控制，因而在短期内不可能做到。假设我们已经拥有尽可能多的高质量数据，我们便可以将重点放在**正则化**技术上。

即使是阶数上的微小变化，比如从2到3，也会显著增加模型的复杂性。仅仅通过简单的限制特征数量（在多项式回归中体现为限制阶数），可能仍然使模型在过简单和过复杂中徘徊， 我们需要一个更细粒度的工具来调整函数的复杂性，使其达到一个合适的平衡位置。

### 范数与权重衰减

在训练参数化机器学习模型时，权重衰减（weight decay）是最广泛使用的正则化的技术之一，它通常也被称为$L_2$**正则化**。一种简单的方法是通过线性函数$f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$中的权重向量的某个范数来度量其复杂性，例如$\| \mathbf{w} \|^2$。要保证权重向量比较小，最常用方法是将其范数作为惩罚项加到最小化损失的问题中。将原来的训练目标**最小化训练标签上的预测损失**，调整为**最小化预测损失和惩罚项之和**。但如果我们的权重向量增长的太大，学习算法可能会更集中于最小化权重范数$\| \mathbf{w} \|^2$。我们回顾一下线性回归中的损失函数：
$$
L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2
$$
$\mathbf{x}^{(i)}$是样本$i$的特征，$y^{(i)}$是样本$i$的标签，$(\mathbf{w}, b)$是权重和偏置参数。为了惩罚权重向量的大小，必须以某种方式在损失函数中添加$\| \mathbf{w} \|^2$。就通过**正则化常数**$\lambda$来描述这种额外惩罚损失的平衡，这是一个非负超参数，我们使用验证数据拟合：
$$
L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2
$$
对于$\lambda = 0$，就恢复了原来的损失函数；对于$\lambda > 0$，就限制了$\| \mathbf{w} \|$的大小。这里仍然除以$2$：因为取一个二次函数的导数时，$2$和$1/2$会抵消，以确保更新表达式看起来既漂亮又简单。这里使用平方范数而不是标准范数（即欧几里得距离）的原因：便于计算，通过平方$L_2$范数，去掉平方根，留下权重向量每个分量的平方和，这使得惩罚的导数很容易计算，导数的和就等于和的导数。

另外使用$L_2$范数，而不是$L_1$范数的原因：事实上，这个选择在整个统计领域中都是有效的和受欢迎的。$L_2$正则化线性模型构成经典的**岭回归**（ridge regression）算法，$L_1$正则化线性回归是统计学中类似的基本模型，通常被称为**套索回归**（lasso regression）。使用$L_2$范数的一个原因是它对权重向量的大分量施加了巨大的惩罚。这使得学习算法偏向于在大量特征上均匀分布权重的模型。在实践中，这可能使它们对单个变量中的观测误差更为稳定。相比之下，$L_1$惩罚会导致模型将权重集中在一小部分特征上，而将其他权重清除为零。这称为**特征选择**（feature selection），可能是其他场景下需要的。

$L_2$正则化回归的小批量随机梯度下降更新如下式：
$$
\begin{aligned}
\mathbf{w} 
& \leftarrow \mathbf{w} - \eta\left[\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right) + \lambda\mathbf{w} \right] \\\
& \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$
上述式子通常$\eta\lambda < 1$，也就是**先把$\mathbf{w}$变小了一点，沿着梯度方向走一点**。

这里根据估计值与观测值之间的差异来更新$\mathbf{w}$，同时也在试图将$\mathbf{w}$的大小缩小到零。这就是为什么这种方法有时被称为**权重衰减**。我们仅考虑惩罚项，优化算法在训练的每一步**衰减**权重。与特征选择相比，权重衰减为我们提供了一种连续的机制来调整函数的复杂度。较小的$\lambda$值对应较少约束的$\mathbf{w}$，而较大的$\lambda$值对$\mathbf{w}$的约束更大。

是否对相应的偏置$b^2$进行惩罚在不同的实践中会有所不同，在神经网络的不同层中也会有所不同。通常，网络输出层的偏置项不会被正则化。

### 实现权重衰减

#### 生成数据

通过一个简单的例子来演示权重衰减，首先，像以前一样生成一些数据，生成公式如下：
$$
y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2)
$$
我们选择标签是关于输入的线性函数。标签同时被均值为0，标准差为0.01高斯噪声破坏。为了使过拟合的效果更加明显，我们可以将问题的维数增加到$d = 200$，并使用一个只包含20个样本的小训练集。

```python
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
```

#### 初始化模型参数

定义一个函数来随机初始化模型参数

```python
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
```

#### 定义$L_2$范数惩罚

实现这一惩罚最方便的方法是对所有项求平方后并将它们求和。

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
```

#### 训练代码实现

线性网络和平方损失没有变化， 所以我们通过`d2l.linreg`和`d2l.squared_loss`导入它们。 唯一的变化是损失现在包括了惩罚项。

```python
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())
```

#### 忽略正则化训练

用`lambd = 0`禁用权重衰减后运行这个代码。注意，这里训练误差有了减少，但测试误差没有减少，这意味着出现了严重的过拟合。

```python
train(lambd=0)
output: w的L2范数是： 13.156189918518066
```

<img src="https://zh.d2l.ai/_images/output_weight-decay_ec9cc0_81_1.svg" alt="损失1" style="zoom:100%;" />

#### 使用权重衰减

下面，使用权重衰减来运行代码。注意，在这里训练误差增大，但测试误差减小。这正是我们期望从正则化中得到的效果。

```python
train(lambd=3)
output: w的L2范数是： 0.3738817870616913
```

<img src="https://zh.d2l.ai/_images/output_weight-decay_ec9cc0_96_1.svg" alt="损失2" style="zoom:100%;" />

### 简洁实现

由于权重衰减在神经网络优化中很常用，深度学习框架为了便于我们使用权重衰减，将权重衰减集成到优化算法中，以便与任何损失函数结合使用。此外，这种集成还有计算上的好处，允许在不增加任何额外的计算开销的情况下向算法中添加权重衰减。由于更新的权重衰减部分仅依赖于每个参数的当前值，因此优化器必须至少接触每个参数一次。

在下面的代码中，我们在实例化优化器时直接通过`weight_decay`指定weight decay超参数。默认情况下PyTorch同时衰减权重和偏移。这里我们只为权重设置了`weight_decay`，所以偏置参数$b$不会衰减。

```python
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
```

上述代码不用太纠结loss函数是否有加上所说的$\frac{\lambda}{2} \|\mathbf{w}\|^2$，这个罚主要是为了限制$\mathbf{w}$学得太大。

```python
train_concise(0)
output: w的L2范数： 13.727912902832031
```

<img src="https://zh.d2l.ai/_images/output_weight-decay_ec9cc0_130_1.svg" alt="简洁实现损失1" style="zoom:100%;" />

```python
train_concise(3)
output: w的L2范数： 0.3890590965747833
```

<img src="https://zh.d2l.ai/_images/output_weight-decay_ec9cc0_131_1.svg" alt="简洁实现损失2" style="zoom:100%;" />

## 暂退法(Dropout)

暂退法的动机：一个好的模型需要对输入数据的扰动鲁棒。解决方法：具有输入噪声的训练等价于Tikhonov正则化，暂退法，即在层之间加入噪声。

暂退法在前向传播过程中，计算每一内部层的同时注入噪声，这已经成为训练神经网络的常用技术。暂退法从表面上看是在训练过程中丢弃（drop out）一些神经元，在整个训练过程的每一次迭代中，标准暂退法包括在计算下一层之前将当前层中的一些节点置零。

在每次训练迭代中，将从均值为零的分布$\epsilon \sim \mathcal{N}(0,\sigma^2)$采样噪声添加到输入$\mathbf{x}$，从而产生扰动点$\mathbf{x}' = \mathbf{x} + \epsilon$，预期是$E[\mathbf{x}'] = \mathbf{x}$。

在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差。换言之，每个中间活性值$h$以**暂退概率**$p$由随机变量$h'$替换，如下所示：
$$
\begin{aligned}
h' =
\begin{cases}
  0 & \text{ ，概率为 } p \\\
  \frac{h}{1-p} & \text{ ，其他情况}
\end{cases}
\end{aligned}
$$
根据此模型的设计，其期望值保持不变，即$E[h'] = h$。

当我们将暂退法应用到多层感知机的隐藏层，以$p$的概率将隐藏单元置为零时。比如在下图中删除了$h_2$和$h_5$，那么输出的计算不再依赖于$h_2$或$h_5$，并且它们各自的梯度在执行反向传播时也会消失。这样，输出层的计算不能过度依赖于$h_1, \ldots, h_5$的任何一个元素。



<img src="http://d2l.ai/_images/dropout2.svg" alt="dropout前后的多层感知机" style="zoom:100%;" />
$$
\begin{aligned}
  \mathbf{h} & = \sigma(\mathbf{W_1} \mathbf{X} + \mathbf{b_1}) \\\
	\mathbf{h'} & = \mathbf{dropout}(\mathbf{h}) \\\
  \mathbf{o} & = \mathbf{W_2} \mathbf{h'} + \mathbf{b_2}\\\
  \mathbf{y} &= \mathbf{softmax}(\mathbf{o})
\end{aligned}
$$


通常，在测试时不用暂退法。正则项只在训练中使用，且常作用在多层感知机的隐藏层输出上，因为他们影响模型参数的更新，在预测中不需要修改模型参数，因此不需要正则项。

### 实现Dropout

下面代码实现 `dropout_layer` 函数，该函数以`dropout`的概率丢弃张量输入`X`中的元素，如上所述重新缩放剩余部分：将剩余部分除以`1.0-dropout`。

```python
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float() #rand生成0-1的均匀分布，若生成的数大于dropout则是1，否则是0
    return mask * X / (1.0 - dropout)
```

测试`dropout_layer`函数。将输入`X`通过暂退法操作，暂退概率分别为0、0.5和1。

```python
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
output: 
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15.]])
tensor([[ 0.,  2.,  4.,  6.,  0.,  0., 12.,  0.],
        [16.,  0., 20., 22., 24., 26., 28.,  0.]])
tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
```

定义模型，定义具有两个隐藏层的多层感知机，我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后），并且可以为每一层分别设置暂退概率：常见的技巧是在靠近输入层的地方设置较低的暂退概率。下面的模型将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5，并且暂退法只在训练期间有效。

```python
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

训练和测试

```python
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://zh.d2l.ai/_images/output_dropout_1110bf_66_0.svg" alt="dropout训练1" style="zoom:100%;" />

简洁实现Dropout

```python
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

训练和测试

```python
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://zh.d2l.ai/_images/output_dropout_1110bf_96_0.svg" style="zoom:100%;" />

## 前向传播、反向传播和计算图

### 前向传播

*前向传播*（forward propagation或forward pass） 指的是：按顺序（从输入层到输出层）计算和存储神经网络中每层的结果。

为了简单起见，假设输入样本是 $\mathbf{x}\in \mathbb{R}^d$，并且隐藏层不包括偏置项。这里的中间变量是：$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x}$，其中$\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$是隐藏层的权重参数。将中间变量$\mathbf{z}\in \mathbb{R}^h$通过激活函数$\phi$后，得到长度为$h$的隐藏激活向量：$\mathbf{h}= \phi (\mathbf{z})$，隐藏变量$\mathbf{h}$也是一个中间变量。假设输出层的参数只有权重$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$，我们可以得到输出层变量，它是一个长度为$q$的向量：$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}$。假设损失函数为$l$，样本标签为$y$，可以计算单个数据样本的损失项，$L = l(\mathbf{o}, y)$。根据$L_2$正则化的定义，给定超参数$\lambda$，正则化项为$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right)$，其中矩阵的Frobenius范数是将矩阵展平为向量后应用的$L_2$范数。最后，模型在给定数据样本上的正则化损失为：$J = L + s$，在下面的讨论中，会将$J$称为**目标函数**（objective function）。

下图是前向传播计算图，其中正方形表示变量，圆圈表示操作符。左下角表示输入，右上角表示输出。注意显示数据流的箭头方向主要是向右和向上的。

<img src="https://zh.d2l.ai/_images/forward.svg" alt="前向传播的计算图" style="zoom:100%;" />

### 反向传播

**反向传播**（backward propagation或backpropagation）指的是计算神经网络参数梯度的方法。简言之，该方法根据微积分中的**链式规则**，按相反的顺序从输出层到输入层遍历网络。该算法存储了计算某些参数梯度时所需的任何中间变量（偏导数）。假设有函数$\mathsf{Y}=f(\mathsf{X})$和$\mathsf{Z}=g(\mathsf{Y})$，其中输入和输出$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$是任意形状的张量。利用链式法则，可以计算$\mathsf{Z}$关于$\mathsf{X}$的导数
$$
\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right)
$$
在这里，$\text{prod}$运算符在执行必要的操作（如换位和交换输入位置）后将其参数相乘。对于向量，它只是矩阵-矩阵乘法。对于高维张量，我们使用适当的对应项。运算符$\text{prod}$指代了所有的这些符号。

在计算图7-1中的单隐藏层简单网络的参数是$\mathbf{W}^{(1)}$和$\mathbf{W}^{(2)}$，反向传播的目的是计算梯度$\partial J/\partial \mathbf{W}^{(1)}$和$\partial J/\partial \mathbf{W}^{(2)}$。为此，应用链式法则，依次计算每个中间变量和参数的梯度。计算的顺序与前向传播中执行的顺序相反，因为需要从计算图的**结果**开始，并朝着参数的方向努力。第一步是计算目标函数$J=L+s$相对于损失项$L$和正则项$s$的梯度。
$$
\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1
$$
接下来，根据链式法则计算目标函数关于输出层变量$\mathbf{o}$的梯度：
$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q
$$
接下来，计算正则化项相对于两个参数的梯度：
$$
\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}
$$
现在可以计算最接近输出层的模型参数的梯度$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$。使用链式法则得出：
$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}
$$
为了获得关于$\mathbf{W}^{(1)}$的梯度，我们需要继续沿着输出层到隐藏层反向传播。关于隐藏层输出的梯度$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$由下式给出：
$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}
$$
由于激活函数$\phi$是按元素计算的，计算中间变量$\mathbf{z}$的梯度$\partial J/\partial \mathbf{z} \in \mathbb{R}^h$需要使用按元素乘法运算符，我们用$\odot$表示：
$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right)
$$
最后，可以得到最接近输入层的模型参数的梯度$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$。根据链式法则，我们得到：
$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}
$$

### 训练神经网络

在训练神经网络时，前向传播和反向传播相互依赖。对于前向传播，我们沿着依赖的方向遍历计算图并计算其路径上的所有变量。然后将这些用于反向传播，其中计算顺序与计算图的相反。

以上述简单网络为例：一方面，前向传播期间计算正则项取决于模型参数$\mathbf{W}^{(1)}$和$\mathbf{W}^{(2)}$的当前值。它们是由优化算法根据最近迭代的反向传播给出的。另一方面，反向传播期间参数(公式94)的梯度计算，取决于由前向传播给出的隐藏变量$\mathbf{h}$的当前值。

因此，在训练神经网络时，在初始化模型参数后，需要交替使用前向传播和反向传播，利用反向传播给出的梯度来更新模型参数。注意，反向传播重复利用前向传播中存储的中间值，以避免重复计算。带来的影响之一是我们需要保留中间值，直到反向传播完成。这也是训练比单纯的预测需要更多的内存（显存）的原因之一。此外，这些中间值的大小与网络层的数量和批量的大小大致成正比。因此，使用更大的批量来训练更深层次的网络更容易导致**内存不足**（out of memory）错误。

## 数值稳定性和模型初始化

初始化方案的选择在神经网络学习中起着举足轻重的作用，它对保持数值稳定性至关重要。此外，这些初始化方案的选择可以与非线性激活函数的选择有趣的结合在一起。我们选择哪个函数以及如何初始化参数可以决定优化算法收敛的速度有多快，糟糕选择可能会导致在训练时遇到梯度爆炸或梯度消失。

### 梯度消失和梯度爆炸

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231021155848733.png" alt="image-20231021155848733" style="zoom:33%;" />

上图为神经网络的梯度计算，可以看到最后会有许多的矩阵乘法，这会带来一些问题。

#### 梯度消失

激活函数sigmoid函数$1/(1 + \exp(-x))$，它会导致梯度消失问题。

```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

<img src="https://zh.d2l.ai/_images/output_numerical-stability-and-init_e60514_6_0.svg" alt="sigmoid函数" style="zoom:100%;" />

正如上图，当sigmoid函数的输入很大或是很小时，它的梯度都会消失。 此外，当反向传播通过许多层时，除非我们在刚刚好的地方， 这些地方sigmoid函数的输入接近于零，否则整个乘积的梯度可能会消失。 当我们的网络有很多层时，除非我们很小心，否则在某一层可能会切断梯度。

梯度消失的问题：

- 梯度值变成0
  - 对16位浮点数尤为严重
- 训练没有进展
  - 不管如何选择学习率
- 对于底部层尤为严重
  - 仅仅顶部层训练的较好
  - 无法让神经网络更深

#### 梯度爆炸

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231021162756721.png" alt="image-20231021162756721" style="zoom: 33%;" />

相反，梯度爆炸可能同样令人烦恼。我们生成100个高斯随机矩阵，并将它们与某个初始矩阵相乘。对于我们选择的尺度（方差$\sigma^2=1$），矩阵乘积发生爆炸。当这种情况是由于深度网络的初始化所导致时，我们没有机会让梯度下降优化器收敛。

```python
M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)
output:
一个矩阵 
 tensor([[ 0.1969,  0.9725,  0.0580, -0.7313],
        [-0.1515,  0.3700,  2.4064, -0.8915],
        [ 0.3983,  0.0538,  0.9967, -1.4327],
        [-0.7009,  1.2494, -0.2294,  1.3601]])
乘以100个矩阵后
 tensor([[ 6.9463e+25, -6.1771e+25,  1.6948e+26, -1.7734e+26],
        [ 9.6462e+25, -8.5781e+25,  2.3535e+26, -2.4627e+26],
        [ 5.9542e+25, -5.2949e+25,  1.4527e+26, -1.5201e+26],
        [ 2.8045e+25, -2.4939e+25,  6.8424e+25, -7.1598e+25]])
```

梯度爆炸的问题：

- 值超出值域(infinity)
  - 对于16位浮点数尤为严重(数值区间6e-5 - 6e4)
- 对学习率敏感
  - 如果学习率太大 -> 大参数值 -> 更大梯度
  - 如果学习率太小 -> 训练无进展
  - 我么可能需要在训练过程不断调整学习率

#### 打破对称性

神经网络设计中的另一个问题是其参数化所固有的对称性。假设有一个简单的多层感知机，它有一个隐藏层和两个隐藏单元。在这种情况下，我们可以对第一层的权重$\mathbf{W}^{(1)}$进行重排列，并且同样对输出层的权重进行重排列，可以获得相同的函数。第一个隐藏单元与第二个隐藏单元没有什么特别的区别。换句话说，在每一层的隐藏单元之间具有排列对称性。

假设输出层将上述两个隐藏单元的多层感知机转换为仅一个输出单元。如果将隐藏层的所有参数初始化为$\mathbf{W}^{(1)} = c$，$c$为常量，在这种情况下，在前向传播期间，两个隐藏单元采用相同的输入和参数，产生相同的激活，该激活被送到输出单元。在反向传播期间，根据参数$\mathbf{W}^{(1)}$对输出单元进行微分，得到一个梯度，其元素都取相同的值。因此，在基于梯度的迭代（例如，小批量随机梯度下降）之后，$\mathbf{W}^{(1)}$的所有元素仍然采用相同的值。这样的迭代永远不会打破对称性，我们可能永远也无法实现网络的表达能力。隐藏层的行为就好像只有一个单元。请注意，虽然小批量随机梯度下降不会打破这种对称性，但暂退法正则化可以。

### 参数初始化

解决（或至少减轻）上述问题的一种方法是进行参数初始化，优化期间的注意和适当的正则化也可以进一步提高稳定性。

权重初始化需要注意的地方：

- 在合理值区间里随机初始化参数
- 训练开始的时候更容易有数值不稳定
  - 远离最优解的地方损失函数表面可能很复杂
  - 最优解附近表面会比较平
- 使用$\mathbf{N}(0,0.01)$来初始可能对小网络没问题，但不能保证深度神经网络



1. 如果我们不指定初始化方法， 框架将使用默认的随机初始化方法，对于中等难度的问题，这种方法通常很有效。
2. Xavier初始化：权重初始化时的方差是根据输入和输出维度来定的

希望让每一层输出的方差和梯度的方差是一个常数

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231021221016566.png" alt="image-20231021221016566" style="zoom:33%;" />

## 环境和分布偏移



## 实战Kaggle比赛：预测房价

[预测房价](https://zh.d2l.ai/chapter_multilayer-perceptrons/kaggle-house-price.html)


---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/4-%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E6%9C%BA/  

