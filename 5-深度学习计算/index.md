# 5 深度学习计算


## 层和块

事实证明，研究讨论“比单个层大”但“比整个模型小”的组件更有价值，由此引入了神经网络**块**的概念。**块**（block）可以描述单个层、由多个层组成的组件或整个模型本身。使用块进行抽象的一个好处是可以将一些块组合成更大的组件，这一过程通常是递归的，如下图所示。

通过定义代码来按需生成任意复杂度的块，可以通过简洁的代码实现复杂的神经网络。

<img src="http://d2l.ai/_images/blocks.svg" alt="多个层被组合成块，形成更大的模型" style="zoom:100%;" />

从编程的角度来看，块由**类**（class）表示。它的任何子类都必须定义一个将其输入转换为输出的前向传播函数，并且必须存储任何必需的参数（但有些块不需要任何参数）。最后，为了计算梯度，块必须具有反向传播函数。在定义我们自己的块时，由于自动微分提供了一些后端实现，我们只需要考虑前向传播函数和必需的参数。

下面的代码生成一个网络，其中包含一个具有256个单元和ReLU激活函数的全连接隐藏层，然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
output: tensor([[ 0.2172,  0.2640,  0.1634,  0.1557,  0.0725,  0.0620,  0.0563, -0.1232,
         -0.0499,  0.0680],
        [ 0.3428,  0.1960,  0.1856,  0.0338, -0.0193,  0.0314, -0.0053, -0.0298,
          0.0883,  0.0390]], grad_fn=<AddmmBackward0>)
```

上述例子通过实例化`nn.Sequential`来构建模型，层的执行顺序是作为参数传递的。简而言之，(**`nn.Sequential`定义了一种特殊的`Module`**)，即在PyTorch中表示一个块的类，它维护了一个由`Module`组成的有序列表。注意，两个全连接层都是`Linear`类的实例，`Linear`类本身就是`Module`的子类。另外，到目前为止，我们一直在通过`net(X)`调用我们的模型来获得模型的输出。这实际上是`net.__call__(X)`的简写。这个前向传播函数非常简单：它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。

### 自定义块

在自定义一个块之前，需要总结每一个块必须提供的基本功能：

- 将输入数据作为其前向传播函数的参数
- 通过前向传播函数来生成输出
- 计算其输出关于输入的梯度，可通过其反向传播函数进行访问
- 存储和访问前向传播计算所需的参数
- 根据需要初始化模型参数

下面的`MLP`类继承了表示块的类。我们的实现只需要提供构造函数（Python中的`__init__`函数）和前向传播函数。

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

上述代码的前向传播函数，以`X`作为输入，计算带有激活函数的隐藏表示，并输出其未规范化的输出值。在这个`MLP`实现中，两个层都是实例变量。接着实例化多层感知机的层，然后在每次调用前向传播函数时调用这些层。首先，`__init__`函数通过`super().__init__()`调用父类的`__init__`函数，然后实例化两个全连接层，分别为`self.hidden`和`self.out`。注意，我们不必担心反向传播函数或参数初始化，系统将自动生成这些。

```python
net = MLP()
net(X)
output: tensor([[-0.1039, -0.1376,  0.0847,  0.1495,  0.0881, -0.1072, -0.5245,  0.1337,
          0.1617,  0.1733],
        [-0.0435, -0.0553, -0.0714,  0.2861,  0.2305, -0.2955, -0.4526,  0.1166,
          0.1557,  0.1899]], grad_fn=<AddmmBackward0>)
```

块的一个主要优点是它的多功能性。我们可以子类化块以创建层（如全连接层的类）、整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。比如在处理卷积神经网络时，充分利用了这种多功能性。

### 顺序块

`Sequential`的设计是为了把其他模块串起来。为了构建我们自己的简化的`MySequential`， 我们只需要定义两个关键函数：

- 一种将块逐个追加到列表中的函数；
- 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

下面的`MySequential`类提供了与默认`Sequential`类相同的功能。

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

`__init__`函数将每个模块逐个添加到有序字典`_modules`中。简而言之，`_modules`的主要优点是：在模块的参数初始化过程中，系统知道在`_modules`字典中查找需要初始化参数的子块。

当`MySequential`的前向传播函数被调用时， 每个添加的块都按照它们被添加的顺序执行。 现在可以使用`MySequential`类重新实现多层感知机。

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
output: tensor([[-0.2120,  0.1035,  0.1243, -0.1841,  0.1651,  0.0483, -0.0668, -0.0528,
         -0.1805, -0.0286],
        [-0.2596,  0.2426,  0.1686, -0.3241, -0.0032,  0.1208, -0.0288, -0.1468,
         -0.1011,  0.0968]], grad_fn=<AddmmBackward0>)
```

### 在前向传播函数中执行代码

`Sequential`类使模型构造变得简单，允许我们组合新的架构，而不必定义自己的类。然而，并不是所有的架构都是简单的顺序架构。当需要更强的灵活性时，我们需要定义自己的块。例如，我们可能希望在前向传播函数中执行Python的控制流。此外，我们可能希望执行任意的数学运算，而不是简单地依赖预定义的神经网络层。

到目前为止，我们网络中的所有操作都对网络的激活值及网络的参数起作用。然而，有时我们可能希望合并既不是上一层的结果也不是可更新参数的项，我们称之为**常数参数**（constant parameter）。例如，我们需要一个计算函数$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$的层，其中$\mathbf{x}$是输入，$\mathbf{w}$是参数，$c$是某个在优化过程中没有更新的指定常量。因此实现了一个`FixedHiddenMLP`类，如下所示：

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。这个权重不是一个模型参数，因此它永远不会被反向传播更新。然后，神经网络将这个固定层的输出通过一个全连接层。

注意，在返回输出之前，模型做了一些不寻常的事情：它运行了一个while循环，在$L_1$范数大于$1$的条件下，将输出向量除以$2$，直到它满足条件为止。最后，模型返回了`X`中所有项的和。此操作可能不会常用于在任何实际任务中，我们只展示如何将任意代码集成到神经网络计算的流程中。

```python
net = FixedHiddenMLP()
net(X)
output: tensor(-0.0914, grad_fn=<SumBackward0>)
```

我们可以混合搭配各种组合块的方法。在下面的例子中，我们以一些方法嵌套块：

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
output: tensor(0.0713, grad_fn=<SumBackward0>)
```

## 参数管理

在选择了架构并设置了超参数后，就进入了训练阶段。此时，我们的目标是找到使损失函数最小化的模型参数值。经过训练后，我们将需要使用这些参数来做出未来的预测。此外，有时我们希望提取参数，以便在其他环境中复用它们，将模型保存下来，以便它可以在其他软件中执行，或者为了获得科学的理解而进行检查。

首先定义一个具有单隐藏层的多层感知机：

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
output: tensor([[-0.0598],
        [ 0.1842]], grad_fn=<AddmmBackward0>)
```

### 参数访问

我们从已有模型中访问参数。当通过`Sequential`类定义模型时，我们可以通过索引来访问模型的任意层。这就像模型是一个列表一样，每层的参数都在其属性中。如下所示，检查第二个全连接层的参数：

```python
print(net[2].state_dict())
output: OrderedDict([('weight', tensor([[ 0.0051, -0.3365,  0.2276, -0.1116,  0.0760, -0.3333,  0.2590,  0.1495]])), ('bias', tensor([-0.0554]))])
```

从输出的结果可以知道：首先，这个全连接层包含两个参数，分别是该层的权重和偏置。两者都存储为单精度浮点数（float32）。注意，参数名称允许唯一标识每个参数，即使在包含数百个层的网络中也是如此。

每个参数都表示为参数类的一个**实例**。要对参数执行任何操作，需要先访问底层的数值。下面的代码从第二个全连接层（即第三个神经网络层）提取偏置，提取后返回的是一个参数类实例，并进一步访问该参数的值。

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

output:
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([-0.0554], requires_grad=True)
tensor([-0.0554])  
```

参数是复合的对象，包含值、梯度和额外信息。这就是我们需要显式参数值的原因。除了值之外，我们还可以访问每个参数的梯度。在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。

```python
net[2].weight.grad == None
output: True
```

当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂， 因为我们需要递归整个树来提取每个子块的参数。下面将通过演示来比较访问第一个全连接层的参数和访问所有层。

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
output: 
('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
```

如果我们将多个块相互嵌套，下面将研究参数命名约定是如何工作的。我们首先定义一个生成块的函数（可以说是“块工厂”），然后将这些块组合到更大的块中。

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
X = torch.rand(size=(2, 4))
rgnet(X)
output: tensor([[0.2746],
        [0.2747]], grad_fn=<AddmmBackward0>)
```

设计了网络后，可以看到它是如何工作的：

```python
print(rgnet)
output: 
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
```

因为层是分层嵌套的，所以可以像通过嵌套列表索引一样访问它们。下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。

```python
rgnet[0][1][0].bias.data
output: tensor([-0.0221, -0.2074, -0.1290,  0.3668, -0.1693, -0.3298, -0.2983,  0.0146])
```

### 参数初始化

默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵，这个范围是根据输入和输出维度计算出的。PyTorch的`nn.init`模块提供了多种预置初始化方法。

#### 内置初始化

首先调用内置的初始化器。下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量，且将偏置参数设置为0。

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
output: (tensor([ 0.0081, -0.0059, 0.0007, -0.0083]), tensor(0.))
```

我们还可以将所有参数初始化为给定的常数，比如初始化为1。

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
output: (tensor([1., 1., 1., 1.]), tensor(0.))
```

我们还可以对某些块应用不同的初始化方法。例如，下面我们使用Xavier初始化方法初始化第一个神经网络层，然后将第三个神经网络层初始化为常量值42。

```python
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
output:
tensor([ 0.0999, -0.5481,  0.3967,  0.0342])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```

#### 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：
$$
\begin{aligned}
  w \sim \begin{cases}
    U(5, 10) & \text{ 可能性 } \frac{1}{4} \\\
    0   & \text{ 可能性 } \frac{1}{2} \\\
    U(-10, -5) & \text{ 可能性 } \frac{1}{4}
  \end{cases}
\end{aligned}
$$
同样，下面实现了一个`my_init`函数来应用到`net`。

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]

output: 
Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[0.0000, 7.1143, -0.0000, -0.0000],
        [0.0000, 0.0000, 9.2134, 7.8409]], grad_fn=<SliceBackward0>)
```

我们始终可以直接设置参数：

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
output: tensor([42.0000,  8.1143,  1.0000,  1.0000])
```

### 参数绑定

有时我们希望在多个层间共享参数：我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

output:
tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])
```

这个例子表明第三个和第五个神经网络层的参数是绑定的。它们不仅值相等，而且由相同的张量表示。因此，如果我们改变其中一个参数，另一个参数也会改变。这里有一个问题：当参数绑定时，梯度会发生什么情况？答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层（即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。

## 自定义层

深度学习成功背后的一个因素是神经网络的灵活性：我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。在这些情况下，必须构建自定义层。

### 不带参数的层

首先，构造一个没有任何参数的自定义层。下面的`CenteredLayer`类要从其输入中减去均值。要构建它，我们只需继承基础层类并实现前向传播功能。

```python
import torch
import torch.nn.functional as F
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

向该层提供一些数据：

```python
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
output: tensor([-2., -1.,  0.,  1.,  2.])
```

现在，我们可以将层作为组件合并到更复杂的模型中。

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0。由于我们处理的是浮点数，因为存储精度的原因，仍然可能会看到一个非常小的非零数。

```python
Y = net(torch.rand(4, 8))
Y.mean()
output: tensor(-1.8626e-09, grad_fn=<MeanBackward0>)
```

### 带参数的层

以上我们知道了如何定义简单的层，下面我们继续定义具有参数的层，这些参数可以通过训练进行调整。我们可以使用内置函数来创建参数，这些函数提供一些基本的管理功能。比如管理访问、初始化、共享、保存和加载模型参数。这样做的好处之一是：我们不需要为每个自定义层编写自定义的序列化程序。

现在，让我们实现自定义版本的全连接层。该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。在此实现中，我们使用修正线性单元（relu）作为激活函数。该层需要输入参数：`in_units`和`units`，分别表示输入数和输出数。

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

接下来，我们实例化`MyLinear`类并访问其模型参数：

```python
linear = MyLinear(5, 3)
linear.weight
output:
Parameter containing:
tensor([[-1.4967,  0.4762,  0.1728],
        [ 3.3487,  1.5446, -1.4548],
        [ 0.4705,  0.0455,  0.5935],
        [ 0.5505, -1.8667,  0.3717],
        [-0.0396, -0.0339, -1.2275]], requires_grad=True)
```

使用自定义层直接执行前向传播计算：

```python
linear(torch.rand(2, 5))
output: tensor([[1.6577, 0.6769, 0.0000],
                [0.0000, 0.3016, 0.0000]])
```

我们还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
output: tensor([[2.0534],
                [0.0973]])
```

## 读写文件

到目前为止，我们讨论了如何处理数据，以及如何构建、训练和测试深度学习模型。然而，有时我们希望保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）。此外，当运行一个耗时较长的训练过程时，最佳的做法是定期保存中间结果，以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。

### 加载和保存张量

对于单个张量，可以直接调用`load`和`save`函数分别读写它们。这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

现在就可以将存储在文件中的数据读回内存：

```python
x2 = torch.load('x-file')
x2
output: tensor([0, 1, 2, 3])
```

我们也可以存储一个张量列表，然后把它们读回内存。

```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
output: (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
```

我们甚至可以写入或读取从字符串映射到张量的字典，当我们要读取或写入模型中的所有权重时，这很方便

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
output: {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
```

### 加载和保存模型参数

保存单个权重向量（或其他张量）确实有用，但是如果想保存整个模型，并在以后加载它们，单独保存每个向量则会变得很麻烦。毕竟可能有数百个参数散布在各处。因此，深度学习框架提供了内置函数来保存和加载整个网络。需要注意的一个重要细节是，这将**保存模型的参数**而不是保存整个模型。例如，如果我们有一个3层多层感知机，我们需要单独指定架构。因为模型本身可以包含任意代码，所以模型本身难以序列化。因此，为了恢复模型，我们需要用代码生成架构，然后从磁盘加载参数。先定义一个多层感知机：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

接下来，将模型的参数存储在一个叫做“`mlp.params`”的文件中：

```python
torch.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，可以直接读取文件中存储的参数：

```python
net_clone = MLP()
net_clone.load_state_dict(torch.load('mlp.params'))
net_clone.eval()
output:
MLP(
  (hidden): Linear(in_features=20, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```

由于两个实例具有相同的模型参数，在输入相同的`X`时，两个实例的计算结果应该相同。验证如下：

```python
Y_clone = net_clone(X)
Y_clone == Y
output: tensor([[True, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True, True, True, True]])
```

## GPU

查看显卡信息

```python
nvidia-smi
```

<img src="https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image-20231025212321759.png" alt="image-20231025212321759" style="zoom: 67%;" />

### 计算设备

我们可以指定用于存储和计算的设备，如CPU和GPU。默认情况下，张量是在内存中创建的，然后使用CPU计算它。

在PyTorch中，CPU和GPU可以用`torch.device('cpu')`和`torch.device('cuda')`表示。`cpu`设备意味着所有物理CPU和内存，这意味着PyTorch的计算将尝试使用所有CPU核心。然而，`gpu`设备只代表一个卡和相应的显存。如果有多个GPU，可以使用`torch.device(f'cuda:{i}')`来表示第$i$块GPU（$i$从0开始）。另外，`cuda:0`和`cuda`是等价的。

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
output: (device(type='cpu'), device(type='cuda'), device(type='cuda', index=1))
```

查询可用gpu的数量：

```python
torch.cuda.device_count()
output: 4
```

现在定义了两个方便的函数，检查是否存在GPU，若存在返回所有的GPU

```python
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
output:
(device(type='cuda', index=0),
 device(type='cpu'),
 [device(type='cuda', index=0)]
 [device(type='cuda', index=1)]
 [device(type='cuda', index=2)]
 [device(type='cuda', index=3)])
```

### 张量与GPU

我们可以查询张量所在的设备。 默认情况下，张量是在CPU上创建的。

```python
x = torch.tensor([1, 2, 3])
x.device
output: device(type='cpu')
```

需要注意的是，无论何时我们要对多个项进行操作，它们都必须在同一个设备上。例如，如果我们对两个张量求和，我们需要确保两个张量都位于同一个设备上，否则框架将不知道在哪里存储结果，甚至不知道在哪里执行计算。

#### 存储在GPU上

有几种方法可以在GPU上存储张量。例如，可以在创建张量时指定存储设备。接下来，在第一个`gpu`上创建张量变量`X`。在GPU上创建的张量只消耗这个GPU的显存。我们可以使用`nvidia-smi`命令查看显存使用情况。一般来说，我们需要确保不创建超过GPU显存限制的数据。

```python
X = torch.ones(2, 3, device=try_gpu())
X
output: tensor([[1., 1., 1.],
                [1., 1., 1.]], device='cuda:0')
```

假设我们至少有两个GPU，下面的代码将在第二个GPU上创建一个随机张量。

```python
Y = torch.rand(2, 3, device=try_gpu(1))
Y
output: tensor([[0.4860, 0.1285, 0.0440],
                [0.9743, 0.4159, 0.9979]], device='cuda:1')
```

#### 复制

如果我们要计算`X + Y`，我们需要决定在哪里执行这个操作。例如，如下图所示，我们可以将`X`传输到第二个GPU并在那里执行操作。不要简单地`X`加上`Y`，因为这会导致异常，运行时引擎不知道该怎么做：它在同一设备上找不到数据会导致失败。由于`Y`位于第二个GPU上，所以我们需要将`X`移到那里，然后才能执行相加运算。

<img src="https://zh-v2.d2l.ai/_images/copyto.svg" alt="复制数据以在同一设备上执行操作" style="zoom:100%;" />

```python
Z = X.cuda(1)
print(X)
print(Z)
output: 
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:0')
tensor([[1., 1., 1.],
        [1., 1., 1.]], device='cuda:1')
```

现在数据在同一个GPU上（`Z`和`Y`都在），可以将它们相加。

```python
Y + Z
output: tensor([[1.4860, 1.1285, 1.0440],
                [1.9743, 1.4159, 1.9979]], device='cuda:1')
```

假设变量`Z`已经存在于第二个GPU上。如果还是调用`Z.cuda(1)`， 它将返回`Z`而不会复制并分配新内存。

```python
Z.cuda(1) is Z
output: True
```

### 神经网络与GPU

类似地，神经网络模型可以指定设备，下面的代码将模型参数放在GPU上：

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

当输入为GPU上的张量时，模型将在同一GPU上计算结果。

```python
X = torch.ones(2, 3, device=try_gpu())
net(X)
output: tensor([[-0.4275],
                [-0.4275]], device='cuda:0', grad_fn=<AddmmBackward0>)
```

下面代码可以确认模型参数是否存储在同一个GPU上：

```python
net[0].weight.data.device
output: device(type='cuda', index=0)
```

## 补充

- torch.rand()：构造均匀分布张量的方法

  `torch.rand`是用于生成**均匀随机分布**张量的函数，从区间`[0,1)`的均匀分布中随机抽取一个随机数生成一个张量。

- torch.randn()：构造标准正态分布张量的方法

  `torch.randn()`是用于生成正态随机分布张量的函数，从**标准正态分布**中随机抽取一个随机数生成一个张量。

- torch.randint()：构造区间分布张量的方法

  `torch.randint()`是用于生成任意区间分布张量的函数，从该区间的均匀分布中随机抽取一个随机数生成一个张量。

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/5-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%AE%A1%E7%AE%97/  

