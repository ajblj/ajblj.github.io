# 3 线性神经网络


## 线性回归

在机器学习领域中的大多数任务通常都与*预测*（prediction）有关。 当我们想预测一个数值时，就会涉及到回归问题。 

> 我们希望根据房屋的面积（平方英尺）和房龄（年）来估算房屋价格（美元）。 为了开发一个能预测房价的模型，我们需要收集一个真实的数据集。 这个数据集包括了房屋的销售价格、面积和房龄。 在机器学习的术语中，该数据集称为***训练数据集***或***训练集***。 每行数据（比如一次房屋交易相对应的数据）称为***样本***（sample）， 也可以称为*数据点*（data point）或*数据样本*（data instance）。 我们把试图预测的目标（比如预测房屋价格）称为***标签***（label）或***目标***（target）。 预测所依据的自变量（面积和房龄）称为***特征***（feature）或***协变量***（covariate）。

通常，使用$n$来表示数据集中的样本数。对索引为$i$的样本，其输入表示为$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，其对应的标签是$y^{(i)}$。

### 线性回归的基本元素

#### 线性模型

当输入包含$d$个特征时，将预测结果$\hat{y}$（通常使用“尖角”符号表示$y$的估计值）表示为：
$$
\hat{y} = w_1  x_1 + ... + w_d  x_d + b
$$
将所有特征放到向量$\mathbf{x} \in \mathbb{R}^d$中，并将所有权重放到向量$\mathbf{w} \in \mathbb{R}^d$中，可以用点积形式来简洁地表达模型，其中向量$\mathbf{x}$对应于单个数据样本的特征：
$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$
用符号表示的矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$，可以很方便地引用整个数据集的$n$个样本，其中，$\mathbf{X}$的每一行是一个样本，每一列是一种特征。

对于特征集合$\mathbf{X}$，预测值$\hat{\mathbf{y}} \in \mathbb{R}^n$可以通过矩阵-向量乘法表示为：
$$
{\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b
$$
无论使用什么手段来观察特征$\mathbf{X}$和标签$\mathbf{y}$，都可能会出现少量的观测误差，因此，即使确信特征与标签的潜在关系是线性的，也会加入一个噪声项来考虑观测误差带来的影响。

#### 损失函数

**损失函数**（loss function）能够量化目标的**实际**值与**预测**值之间的差距。通常我们会选择非负数作为损失，且数值越小表示损失越小，完美预测时的损失为0。回归问题中最常用的损失函数是平方误差函数。当样本$i$的预测值为$\hat{y}^{(i)}$，其相应的真实标签为$y^{(i)}$时，平方误差可以定义为以下公式：
$$
l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
$$
为了度量模型在整个数据集上的质量，我们需计算在训练集$n$个样本上的损失均值（也等价于求和）：
$$
L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2
$$
在训练模型时，我们希望寻找一组参数（$\mathbf{w}', b'$），这组参数能最小化在所有训练样本上的总损失。如下式：
$$
\mathbf{w}', b' = \operatorname{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b)
$$

#### 解析解

与其他大部分模型不同，线性回归的解可以用一个公式简单地表达出来， 这类解叫作解析解（analytical solution）

首先，将偏置$b$合并到参数$\mathbf{w}$中，合并方法是在包含所有参数的矩阵中附加一列，上述预测问题是最小化$\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$。这在损失平面上只有一个临界点，这个临界点对应于整个区域的损失极小点。将损失关于$\mathbf{w}$的导数设为0，得到解析解：

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231011162021797.png" alt="image-20231011162021797" style="zoom:33%;" />

#### 随机梯度下降

梯度下降（gradient descent）方法， 几乎可以优化所有深度学习模型。它通过不断地在损失函数递减的方向上更新参数来降低误差。

梯度下降最简单的用法是计算损失函数（数据集中所有样本的损失均值） 关于模型参数的导数（在这里也可以称为梯度）。 但实际中的执行可能会非常慢：因为在每一次更新参数之前，我们必须遍历整个数据集。 因此，我们通常会在每次需要计算更新的时候随机抽取一小批样本， 这种变体叫做**小批量随机梯度下降**（minibatch stochastic gradient descent）。

在每次迭代中，首先随机抽样一个小批量$\mathcal{B}$，它是由固定数量的训练样本组成的，然后计算小批量的平均损失关于模型参数的导数（也可以称为梯度）。最后，将梯度乘以一个预先确定的正数$\eta$，并从当前参数的值中减掉。如：
$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)
$$
总结来说，算法步骤如下：

- 初始化模型参数的值，如随机初始化
- 从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这一步骤

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231011164929895.png" alt="image-20231011164929895" style="zoom:33%;" />
$$
\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}
$$
上述公式中的$\mathbf{w}$和$\mathbf{x}$都是向量，$|\mathcal{B}|$表示每个小批量中的样本数，这也称为批量大小（batch size），$\eta$表示学习率（learning rate）。批量大小和学习率的值通常是手动预先指定，而不是通过模型训练得到的。

这些可以调整但不在训练过程中更新的参数称为**超参数**（hyperparameter）。**调参**（hyperparameter tuning）是选择超参数的过程，超参数通常是根据训练迭代结果来调整的，而训练迭代结果是在独立的**验证数据集**（validation dataset）上评估得到的。

### 正态分布

正态分布和线性回归之间的关系很密切。正态分布（normal distribution），也称为**高斯分布**（Gaussian distribution），简单的说，若随机变量$x$具有均值$\mu$和方差$\sigma^2$（标准差$\sigma$），其正态分布概率密度函数如下：
$$
p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right)
$$

```python
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

下面可视化正态分布：

```python
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

输出下面可视化：

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231011210425233.png" alt="image-20231011210425233" style="zoom: 50%;" />

改变均值会产生沿$x$轴的偏移，增加方差将会分散分布、降低其峰值。

### 线性回归到深度网络

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231011173736046.png" alt="image-20231011173736046" style="zoom:33%;" />

在上图所示的神经网络中，输入为$x_1, \ldots, x_d$，因此输入层中的**输入数**（或称为**特征维度**）为$d$。网络的输出为$o_1$，因此输出层中的**输出数**是1。需要注意的是，输入值都是已经给定的，并且只有一个**计算**神经元。由于模型重点在发生计算的地方，所以通常我们在计算层数时不考虑输入层。所以上述神经网络的层数为1，可以将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络。

对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连，这种变换称为**全连接层**（fully-connected layer）或称为**稠密层**（dense layer）。

## 实现线性回归

### 生成数据集

假设生成一个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征。所以数据集是一个矩阵$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$。使用线性模型参数$\mathbf{w} = [2, -3.4]^\top$、$b = 4.2$和噪声项$\epsilon$生成数据集及其标签：
$$
\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon
$$
$\epsilon$可以视为模型预测和标签时的潜在观测误差。在这里认为标准假设成立，即$\epsilon$服从均值为0的正态分布。为了简化问题，我们将标准差设为0.01。python代码如下，其中$\mathbf{X}$是随机生成的均值为0、标准差为1的`num_examples`行`len(w)`列的矩阵

```python
def synthetic_data(w, b, num_examples):
  """生成y=Xw+b+噪声"""
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

### 读取数据集

在下面的代码中，定义一个`data_iter`函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为`batch_size`的小批量。 每个小批量包含一组特征和标签。

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] # 返回迭代器
        

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
output: tensor([[ 0.3934,  2.5705],
                [ 0.5849, -0.7124],
                [ 0.1008,  0.6947],
                [-0.4493, -0.9037],
                [ 2.3104, -0.2798],
                [-0.0173, -0.2552],
                [ 0.1963, -0.5445],
                [-1.0580, -0.5180],
                [ 0.8417, -1.5547],
                [-0.6316,  0.9732]])
				tensor([[-3.7623],
                [ 7.7852],
                [ 2.0443],
                [ 6.3767],
                [ 9.7776],
                [ 5.0301],
                [ 6.4541],
                [ 3.8407],
                [11.1396],
                [-0.3836]])
```

当运行迭代时，会连续地获得不同的小批量，直至遍历完整个数据集。 上面实现的迭代对教学来说很好，但它的执行效率很低，可能会在实际问题上陷入麻烦。 例如，它要求我们将所有数据加载到内存中，并执行大量的随机内存访问。 在深度学习框架中实现的内置迭代器效率要高得多， 它可以处理存储在文件中的数据和数据流提供的数据。

### 初始化模型参数

在我们开始用小批量随机梯度下降优化模型参数之前， 需要先有一些参数。 在下面的代码中，通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。

```python
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

在初始化参数之后，需要更新这些参数，直到这些参数足够拟合数据。 每次更新都需要计算损失函数关于模型参数的梯度， 有了这个梯度，就可以向减小损失的方向更新每个参数。 这里使用 前面引入的自动微分来计算梯度。

### 定义模型

定义模型，将模型的输入和参数同模型的输出关联起来。只需计算输入特征$\mathbf{X}$和模型权重$\mathbf{w}$的矩阵-向量乘法后加上偏置$b$。

```python
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

> 补充：`torch.mul()` 、 `torch.mm()` 及`torch.matmul()`的区别
>
> - `torch.mul(a,b)` ：矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵。
> - `torch.mm(a,b)`：矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵。限定二维，且不支持广播。
> - `torch.matmul(a,b)`：该操作取决于张量的维度
>   - 如果a、b都是一维的，则返回点积（标量）；`torch.dot()`
>   - 如果a、b都是二维的，则返回矩阵乘积；`torch.mm()`
>   - 如果a是二维的，b是一维的，则返回矩阵向量积；`torch.mv()`
>   - 如果a是一维的，b是二维的，那么为了矩阵相乘，在其维数前面加上1。在矩阵相乘之后，前面的维度被移除；
>   - 如果a和b都至少是一维的，并且至少有一个参数是N维的（其中N>2），则返回一个成批矩阵乘法。

### 定义损失函数

这里使用平方损失函数。 在实现中，需要将真实值`y`的形状转换为和预测值`y_hat`的形状相同。

```python
def squared_loss(y_hat, y):
    """均方损失"""
    return 1/2 * (y_hat - y.reshape(y_hat.shape)) ** 2
```

### 定义优化函数

这里使用小批量随机梯度下降法，该函数接受模型参数集合、学习速率和批量大小作为输入。每一步更新的大小由学习速率`lr`决定。 因为计算的损失是一个批量样本的总和，所以用梯度除掉批量大小（`batch_size`） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。

```python
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad(): # 定义函数的时候不用参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size # 计算公式参照公式(8)
            param.grad.zero_()
```

### 训练

在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。 计算完损失后，我们开始反向传播，存储每个参数的梯度。 最后，我们调用优化算法`sgd`来更新模型参数。

概括步骤如下：

- 初始化参数
- 重复以下训练，直到完成
  - 计算梯度$\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
  - 更新参数$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

在每个*迭代周期*（epoch）中，使用`data_iter`函数遍历整个数据集， 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。这里的迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设为3和0.03。 设置超参数很棘手，需要通过反复试验进行调整。

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        
output: epoch 1, loss 0.039428
				epoch 2, loss 0.000156
				epoch 3, loss 0.000053
```

## 使用深度学习框架简洁实现线性回归

### 生成数据集

```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

### 读取数据集

调用框架中现有的API来读取数据，将`features`和`labels`作为API的参数传递，并通过数据迭代器指定`batch_size`。 此外，布尔值`is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。使用深度学习框架读取数据可以帮你选好batch

```python
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

使用`data_iter`的方式与我们在**2.2节**中使用`data_iter`函数的方式相同。为了验证是否正常工作，让我们读取并打印第一个小批量样本。 与2.2节不同，这里使用`iter`构造Python迭代器，并使用`next`从迭代器中获取第一项。

```python
next(iter(data_iter))
output: [tensor([[-0.3375,  0.3114],
                 [ 0.4994,  0.8915],
                 [ 0.7587, -0.9327],
                 [-1.0011,  1.5100],
                 [ 0.6620,  0.4560],
                 [-0.4271, -0.7234],
                 [ 1.2387, -0.4021],
                 [ 1.7149, -1.5818],
                 [ 0.5998, -1.4765],
                 [ 1.4491,  0.4250]]),
					tensor([[ 2.4706],
                  [ 2.1752],
                  [ 8.8973],
                  [-2.9463],
                  [ 3.9697],
                  [ 5.7897],
                  [ 8.0600],
                  [13.0047],
                  [10.4142],
                  [ 5.6422]])]
```

### 定义模型

对于标准深度学习模型，我们可以使用框架的预定义好的层。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。 我们首先定义一个模型变量`net`，它是一个`Sequential`类的实例。 `Sequential`类将多个层串联在一起。 当给定输入数据时，`Sequential`实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。 在下面的例子中，我们的模型只包含一个层，因此实际上不需要`Sequential`。 但是由于以后几乎所有的模型都是多层的，在这里使用`Sequential`可以熟悉“标准的流水线”。

回顾**1.3节Figure1-4**中的单层网络架构， 这一单层被称为*全连接层*（fully-connected layer）， 因为它的每一个输入都通过矩阵-向量乘法得到它的每个输出。

在PyTorch中，全连接层在`Linear`类中定义。 将两个参数传递到`nn.Linear`中。 第一个指定输入特征形状，即2，第二个指定输出特征形状，输出特征形状为单个标量，因此为1。

```python
# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
```

### 初始化模型参数

深度学习框架通常有预定义的方法来初始化参数，正如我们在构造`nn.Linear`时指定输入和输出尺寸一样， 现在我们能直接访问参数以设定它们的初始值。 我们通过`net[0]`选择网络中的第一层， 然后使用`weight.data`和`bias.data`方法访问参数。 我们还可以使用替换方法`normal_`和`fill_`来重写参数值。

```python
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
net[0].weight.data, net[0].bias.data
output: (tensor([[0.0153, 0.0152]]), tensor([0.]))
```

### 定义损失函数

计算均方误差使用的是`MSELoss`类，也称为平方$L_2$范数，默认情况下，它返回所有样本损失的平均值。

```python
loss = nn.MSELoss()
```

### 定义优化算法

小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在`optim`模块中实现了该算法的许多变种。当实例化一个`SGD`实例时，要指定优化的参数 （可通过`net.parameters()`从模型中获得）以及优化算法所需的超参数字典。 小批量随机梯度下降只需要设置`lr`值，这里设置为0.03。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

### 训练

在每个迭代周期里，将完整遍历一次数据集（`train_data`），不停地从中获取一个小批量的输入和相应的标签。对于每一个小批量，我们会进行以下步骤:

* 通过调用`net(X)`生成预测并计算损失`l`（前向传播）。
* 通过进行反向传播来计算梯度。

* 通过调用优化器来更新模型参数。

```python
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward() # 反向传播计算梯度，pytorch已经做了sum()的操作
        trainer.step() # 更新模型参数
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    
output: epoch 1, loss 0.000184
				epoch 2, loss 0.000099
				epoch 3, loss 0.000098
```

## softmax回归

### 分类问题

一种表示分类数据的简单方法：独热编码，它是一个向量，它的分量和类别一样多。类别对应的分量设置为1，其他所有分量设置为0。

### 网络架构

为了估计所有可能类别的条件概率，需要一个有多个输出的模型，每个类别对应一个输出。为了解决线性模型的分类问题，我们需要和输出一样多的**仿射函数**（affine function），每个输出对应于它自己的仿射函数。在例子中，由于有4个特征和3个可能的输出类别，需要12个标量来表示权重（带下标的$w$），3个标量来表示偏置（带下标的$b$）。下面为每个输入计算三个**未规范化的预测**（logit）：$o_1$、$o_2$和$o_3$。
$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$
我们可以用神经网络图来描述这个计算过程，与线性回归一样，softmax回归也是一个单层神经网络。由于计算每个输出$o_1$、$o_2$和$o_3$取决于所有输入$x_1$、$x_2$、$x_3$和$x_4$，所以softmax回归的输出层也是全连接层。

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/softmaxreg.svg" alt="softmaxreg" style="zoom:120%;" />

通过向量形式表示为$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$​，由此，已经将所有权重放到一个$3 \times 4$矩阵中。对于给定数据样本的特征$\mathbf{x}$，输出是由权重与输入特征进行矩阵-向量乘法再加上偏置$\mathbf{b}$得到的。

### 全连接层的参数开销

全连接层是“完全”连接的，可能有很多可学习的参数。具体来说，对于任何具有$d$个输入和$q$个输出的全连接层，参数开销为$\mathcal{O}(dq)$，这个数字在实践中可能高得令人望而却步。幸运的是，将$d$个输入转换为$q$个输出的成本可以减少到$\mathcal{O}(\frac{dq}{n})$，其中超参数$n$可以由我们灵活指定，以在实际应用中平衡参数节约和模型有效性。

### softmax运算

我们不能将未规范化的预测$o$直接视作感兴趣的输出，因为将线性层的输出直接视为概率时存在一些问题：一方面，没有限制这些输出数字的总和为1；另一方面，根据输入的不同，它们可以为负值。

softmax函数能够将未规范化的预测变换为非负数并且总和为1，同时让模型保持可导的性质。为了完成这一目标，首先对每个未规范化的预测求幂，这样可以确保输出非负。 为了确保最终输出的概率值总和为1，我们再让每个求幂后的结果除以它们的总和。如下式：
$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{其中}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}
$$
softmax运算不会改变未规范化的预测$\mathbf{o}$之间的大小次序，只会确定分配给每个类别的概率。

因此，在预测过程中仍然可以用下式来选择最有可能的类别。
$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j
$$
尽管softmax是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。因此，softmax回归是一个**线性模型**（linear model）。

### 小批量样本的矢量化

通常会对小批量样本的数据执行矢量计算。假设读取了一个批量的样本$\mathbf{X}$，其中特征维度（输入数量）为$d$，批量大小为$n$。此外，假设在输出中有$q$个类别。那么小批量样本的特征为$\mathbf{X} \in \mathbb{R}^{n \times d}$，权重为$\mathbf{W} \in \mathbb{R}^{d \times q}$，偏置为$\mathbf{b} \in \mathbb{R}^{1\times q}$。softmax回归的矢量计算表达式为：
$$
\begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned}
$$
相对于一次处理一个样本，小批量样本的矢量化加快了$\mathbf{X}和\mathbf{W}$的矩阵-向量乘法。由于$\mathbf{X}$中的每一行代表一个数据样本，那么softmax运算可以**按行**（rowwise）执行：对于$\mathbf{O}$的每一行，我们先对所有项进行幂运算，然后通过求和对它们进行标准化。其中，$\mathbf{X} \mathbf{W} + \mathbf{b}$的求和会使用广播机制，小批量的未规范化预测$\mathbf{O}$和输出概率$\hat{\mathbf{Y}}$都是形状为$n \times q$的矩阵。

### 损失函数

使用最大似然估计。

#### 对数似然

softmax函数给出了一个向量$\hat{\mathbf{y}}$，可以将其视为“对给定任意输入$\mathbf{x}$的每个类的条件概率”。例如，$\hat{y}_ 1=P(y=\text{猫} \mid \mathbf{x})$。假设整个数据集$\{\mathbf{X}, \mathbf{Y}\}$具有$n$个样本，其中索引$i$的样本由特征向量$\mathbf{x}^{(i)}$和独热标签向量$\mathbf{y}^{(i)}$组成。可以将估计值与实际值进行比较：
$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
$$
根据最大似然估计，我们最大化$P(\mathbf{Y} \mid \mathbf{X})$，相当于最小化负对数似然：
$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)})
$$
其中，对于任何标签$\mathbf{y}$和模型预测$\hat{\mathbf{y}}$，损失函数为：
$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j
$$
上述的损失函数通常被称为**交叉熵损失**（cross-entropy loss）。由于$\mathbf{y}$是一个长度为$q$的独热编码向量，所以除了一个项以外的所有项$j$都消失了。由于所有$\hat{y}_j$都是预测的概率，所以它们的对数永远不会大于$0$。

#### softmax及其导数

将**公式12**代入损失**公式17**中，利用softmax的定义，得到：
$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$
考虑相对于任何未规范化的预测$o_j$的导数，得到：
$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$
换句话说，导数是softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。从这个意义上讲，这与在回归中看到的非常相似，其中梯度是观测值$y$和估计值$\hat{y}$之间的差异。

### 交叉熵损失

信息论的核心思想是量化数据中的信息内容。在信息论中，该数值被称为分布$P$的**熵**（entropy）。可以通过以下方程得到：
$$
H[P] = \sum_j - P(j) \log P(j)
$$
信息论的基本定理之一指出，为了对从分布$p$中随机抽取的数据进行编码，至少需要$H[P]$“纳特（nat）”对其进行编码。“纳特”相当于**比特**（bit），但是对数底为$e$而不是2。因此，一个纳特是$\frac{1}{\log(2)} \approx 1.44$比特。

交叉熵方法，常用来衡量两个概率的区别：
$$
H(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)})=-\sum_{j=1}^q y_j^{(i)}\log \hat{y}_j^{(i)}
$$

### 图像分类数据集

[解析与实现](https://zh.d2l.ai/chapter_linear-networks/image-classification-dataset.html)

## 实现softmax回归

### 读取数据

参照**4.8**图像分类数据集

### 初始化模型参数

原始数据集中的每个样本都是$28 \times 28$的图像，但这里展平每个图像，把它们看作长度为784的向量，并且暂时只把每个像素位置看作一个特征。因为我们的数据集有10个类别，所以网络输出维度为10，因此，权重将构成一个$784 \times 10$的矩阵，偏置将构成一个$1 \times 10$的行向量。

与线性回归一样，这里将使用正态分布初始化权重`W`，偏置初始化为0。

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 定义softmax操作

实现softmax由三步骤组成：

- 对每个项求幂（使用`exp`）；
- 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
- 将每一行除以其规范化常数，确保结果的和为1。

$$
\mathrm{softmax}(\mathbf{X})_ {ij} = \frac{\exp(\mathbf{X}_ {ij})}{\sum_k \exp(\mathbf{X}_{ik})}
$$

分母或规范化常数，有时也称为**配分函数**（其对数称为对数-配分函数）

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制
```

使用上述softmax的例子如下：

```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
output: (tensor([[0.0370, 0.2185, 0.0159, 0.6100, 0.1186],
                 [0.1783, 0.2683, 0.0036, 0.2175, 0.3323]]),
         tensor([1., 1.]))
```

### 定义模型

定义softmax操作后，就可以实现softmax回归模型，下面的代码定义了输入如何通过网络映射到输出。(注意，将数据传递到模型之前，我们使用`reshape`函数将每张原始图像展平为向量。)

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

### 定义损失函数

这里实现交叉熵损失函数，这可能是深度学习中最常见的损失函数，因为目前分类问题的数量远远超过回归问题的数量。

这里不使用Python的for循环迭代预测（这往往是低效的），而是通过一个运算符选择所有元素。我们创建一个数据样本`y_hat`，其中包含2个样本在3个类别的预测概率，以及它们对应的标签`y`。有了`y`，就知道在第一个样本中，第一类是正确的预测；而在第二个样本中，第三类是正确的预测。

然后**使用`y`作为`y_hat`中概率的索引**，选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
output: tensor([0.1000, 0.5000])
```

下面实现交叉熵损失函数

```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
output: tensor([2.3026, 0.6931])
```

### 分类精度

分类精度即正确预测数量与总预测数量之比，精度通常是我们最关心的性能衡量标准，在训练分类器时几乎总会关注它。

首先，如果`y_hat`是矩阵，那么假定第二个维度存储每个类的预测分数。使用`argmax`获得每行中最大元素的索引来获得预测类别。 然后将预测类别与真实`y`元素进行比较。 由于等式运算符“`==`”对数据类型很敏感， 因此我们将`y_hat`的数据类型转换为与`y`的数据类型一致。 结果是一个包含0（错）和1（对）的张量。 最后，求和会得到正确预测的数量。

```python
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

同样，对于任意数据迭代器`data_iter`可访问的数据集， 可以评估在任意模型`net`的精度。

```python
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

这里定义一个实用程序类`Accumulator`，用于对多个变量进行累加。 在上面的`evaluate_accuracy`函数中， 我们在`Accumulator`实例中创建了2个变量， 分别用于存储正确预测的数量和预测的总数量。 当我们遍历数据集时，两者都将随着时间的推移而累加。

```python
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

### 训练

在这里，我们重构训练过程的实现以使其可重复使用。 首先，我们定义一个函数来训练一个迭代周期。 `updater`是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是`d2l.sgd`函数，也可以是框架的内置优化函数。

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

在展示训练函数的实现之前，我们定义一个在动画中绘制数据的实用程序类`Animator`， 它能够简化本书其余部分的代码。

```python
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

接下来实现一个训练函数， 它会在`train_iter`访问到的训练数据集上训练一个模型`net`。 该训练函数将会运行多个迭代周期（由`num_epochs`指定）。 在每个迭代周期结束时，利用`test_iter`访问到的测试数据集对模型进行评估。 我们将利用`Animator`类来可视化训练进度。

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

作为一个从零开始的实现，我们使用小批量随机梯度下降来优化模型的损失函数，设置学习率为0.1。

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

现在训练模型10个迭代周期，在这里迭代周期（`num_epochs`）和学习率（`lr`）都是可调节的超参数。 通过更改它们的值，我们可以提高模型的分类精度。

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231015171831676.png" alt="image-20231015171831676" style="zoom: 50%;" />

### 预测

现在训练已经完成，模型已经准备好对图像进行分类预测。 给定一系列图像，我们将比较它们的实际标签（文本输出的第一行）和模型预测（文本输出的第二行）。

```python
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## 使用深度学习框架简洁实现softmax回归

### 读取数据

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

### 初始化模型参数和定义模型

softmax回归的输出层是一个全连接层。 因此，为了实现我们的模型，只需在`Sequential`中添加一个带有10个输出的全连接层。注意，这里还是将$28 \times 28$的图像，展平成长度为784的向量。

```python
# PyTorch不会隐式地调整输入的形状。因此，我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

### softmax实现和定义损失函数

#### 重新审视softmax的实现

- 问题一：softmax函数$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，其中$\hat y_j$是预测的概率分布。$o_j$是未规范化的预测$\mathbf{o}$的第$j$个元素。如果$o_k$中的一些数值非常大，那么$\exp(o_k)$可能大于数据类型容许的最大数字，即**上溢**（overflow）。这将使分母或分子变为`inf`（无穷大），最后得到的是0、`inf`或`nan`（不是数字）的$\hat y_j$。在这些情况下，我们无法得到一个明确定义的交叉熵值。

解决这个问题的一个技巧是：在softmax计算之前，先从所有$o_k$中减去$\max(o_k)$。这里可以看到每个$o_k$按常数进行的移动不会改变softmax的返回值：
$$
\hat y_j =  \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}
$$

- 问题二：在减法和规范化步骤之后，可能有些$o_j - \max(o_k)$具有较大的负值。由于精度受限，$\exp(o_j - \max(o_k))$将有接近零的值，即**下溢**（underflow）。这些值可能会四舍五入为零，使$\hat y_j$为零，并且使得$\log(\hat y_j)$的值为`-inf`。反向传播几步后，我们可能会发现自己面对一屏幕可怕的`nan`结果。

尽管我们要计算指数函数，但我们最终在计算交叉熵损失时会取它们的对数。通过将softmax和交叉熵结合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题。如下面的等式所示，我们避免计算$\exp(o_j - \max(o_k))$，而可以直接使用$o_j - \max(o_k)$，因为$\log(\exp(\cdot))$被抵消了。
$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$
我们也希望保留传统的softmax函数，以备我们需要评估通过模型输出的概率。但是，我们没有将softmax概率传递到损失函数中，而是在交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其对数。

#### 定义损失函数

```python
loss = nn.CrossEntropyLoss(reduction='none')
```

### 定义优化算法

这里使用学习率为0.1的小批量随机梯度下降作为优化算法。

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

### 训练

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231015181351105.png" alt="image-20231015181351105" style="zoom:50%;" />


---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/3-%E7%BA%BF%E6%80%A7%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/  

