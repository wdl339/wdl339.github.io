---
author : "wdl"
title : "实现一个微型深度学习框架Needle"
date : "2024-09-11"
description : "《CMU 10-414 Deep Learning System》课程配套项目"
tags : [
    "AI",
    "system",
    "自学课程"
]
categories : [
    "SelfStudy",
    "Project"
]
math: true
slug: "needle"
---

(内容持续更新中)

## 前言

**needle**（**ne**cessary **e**lements of **d**eep **le**arning）深度学习框架可以视为两层：上层是计算图，用于前向推理、自动微分和反向传播；下层是张量线性代数库，其负责底层的张量计算。



## 自动微分

### 计算图

首先使用 `numpy` 的 CPU 后端实现自动微分的基础功能。

`python/needle/autograd.py `定义了计算图框架的基础，并构成了自动微分框架的核心。其中最重要的一些类：

- `Value`：计算图中计算的值（也可以视为是图节点），是一个泛型类，目前主要通过其子类 `Tensor` 与该类进行交互。
- `Op`：计算图中的算子。算子需要在其 `compute()` 方法中定义“前向”传播（即如何在 `Value` 对象的底层数据上计算该算子），并通过 `gradient()` 方法定义“反向”传播，即如何与传入的输出梯度相乘。`compute()` 的输入是 `NDArray` 对象，也就是说，`compute()` 是在**原始数据对象**上计算前向传播，而不是在 `Tensor` 对象上。 `gradient()` 的输入是 `Tensor` 对象，在该函数中进行的任何调用都应通过 `TensorOp` 操作完成（以便可以对梯度进行进一步求导）
- `Tensor`： `Value` 的一个子类，对应于计算图中的实际张量输出，即多维数组。提供了一些便捷函数（例如运算符重载）以便使用常规的 Python 语法操作张量。
- `TensorOp`：`Op` 的一个子类，用于返回张量的算子。

```python
class Value:
    """A value in the computational graph."""

    op: Optional[Op]
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool
    
    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data
    
    ...
```

```python
class Op:
    """Operator definition."""

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        ...

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        ...

```

例如，运行代码块：

```
x1 = ndl.Tensor([3], dtype="float32")
x2 = ndl.Tensor([4], dtype="float32")
x3 = x1 * x2
x3
```

在最后一行：

- `autograd.TensorOp.__call__` 调用
- `autograd.Tensor.make_from_op`，后者调用
- `autograd.Tensor.realize_cached_data`，最终调用
- `x3.op.compute`

其中：

- `make_from_op` 负责构建计算图的节点。
- 实际的计算操作直到调用 `realize_cached_data` 时才会发生。



### 算子实现

`python/needle/ops/ops_mathematic.py`与`python/needle/ops/ops_logarithmic.py` 文件包含各种算子的实现。一些重要的算子：

- `EWiseDiv`：对输入进行逐元素的真除法（2 个输入）。

- `DivScalar`：将输入逐元素除以一个标量（1 个输入，`scalar` - 数字）。

- （`Add` ,`Mul`, `Pow`与`Div`类似）

- `MatMul`：对输入进行矩阵乘法，并非逐元素（2 个输入）。

- `Summation`：沿给定轴对数组元素求和（1 个输入，`axes` - 元组）。

- `BroadcastTo`：将数组广播到新形状（1 个输入，`shape` - 元组）。

- `Reshape`：在不改变数据的情况下为数组赋予新形状（1 个输入，`shape` - 元组）。

- `Negate`：逐元素取数值负（1 个输入）。

- `Transpose`：反转两个轴的顺序（axis1, axis2），默认为最后两个轴（1 个输入，`axes` - 元组）。

- `relu`: $ReLU(x) = max(0, x)$

- `LogSumExp`: 
  $$
  \text{LogSumExp}(z) = \log (\sum_{i} \exp (z_i - \max{z})) + \max{z}
  $$



### 反向传播

反向传播，即将函数的相关导数与传入的反向梯度相乘。反向模式自动微分的总体目标是计算某个下游函数$\ell$ 关于 $f(x,y)$ 对x（或 y）的梯度。用公式表示，我们可以将其写成：
$$
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x}.
$$
传入的反向梯度正是$\frac{\partial \ell}{\partial f(x,y)}$这一项，因此我们希望 `gradient()` 函数最终计算的是这个反向梯度与函数自身的导数$\frac{\partial f(x,y)}{\partial x}$的乘积。

例如elementwise multiplication：$f(x,y) = x \circ   y$， 它的$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)}  y$，因此`EWiseMul`：

```
class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs
```

`out_grad` 的大小始终是操作的输出的大小， `gradient()` 返回的 `Tensor` 对象的大小必须始终与操作的原始输入大小相同。

反向传播时

- 计算图最末端的那个`Tensor`的`backward()`，调用
- `compute_gradient_of_variables`，按拓扑排序的逆序，对每个node，调用
- `node.op.gradient_as_tuple`（作用是永远返回`Tuple["Value"]`），调用
- `gradient`



## 神经网络

### 重要的类实现

`Parameter`类用于表示可学习的参数，其是`Tensor`的子类。相比`Tensor`类，这个类不必再引入新的行为或者接口。

`Module`类用于表示神经网络中一个个子模块。

```python
class Parameter(Tensor):
    def _unpack_params(value: object) -> List[Tensor]:
        if isinstance(value, Parameter):
            return [value]
        elif isinstance(value, Module):
            return value.parameters()
        elif isinstance(value, dict):
            params = []
            for k, v in value.items():
                params += _unpack_params(v)
            return params
        elif isinstance(value, (list, tuple)):
            params = []
            for v in value:
                params += _unpack_params(v)
            return params
        else:
            return []

class Module:
	# 获取模块中所有的可学习的参数
    def parameters(self) -> List[Tensor]:
        return _unpack_params(self.__dict__)

	# 进行前向传播
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

重要的`Module`类有

- `Linear`：$y = xA^T + b$

- `ReLU`：$ReLU(x) = max(0, x)$

- `Sequential`: 

  ```
  class Sequential(Module):
      def __init__(self, *modules):
          super().__init__()
          self.modules = modules
  
      def forward(self, x: Tensor) -> Tensor:
          y = x
          for module in self.modules:
              y = module(y)
          return y
  ```

- `SoftmaxLoss`: 
  $$
  \ell_\text{softmax}(z,y) = \log \sum_{i=1}^k \exp z_i - z_y
  $$

- `LayerNorm1d`·:
  $$
  y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b
  $$
  where $\textbf{E}[x]$ denotes the empirical mean of the inputs, $\textbf{Var}[x]$ denotes their empirical variance

- `Flatten`: input shape `(B,X_0,X_1,...)`, output shape `(B, X_0 * X_1 * ...)`

- `BatchNorm1d`: 公式同`LayerNorm1d`



`Optimizer`类用于优化模型中可学习参数。

```python
class Optimizer:
    def __init__(self, params):
        self.params = params

    def reset_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplemented()
```



### 其他技术细节

#### 修改tensor的data域

在实现SGD时，由于存在多个batch，可能会在一个循环里对待学习参数进行更新，即：

```python
for _ in range(iterations):
	w -= lr * grad
```

如果直接使用Tensor之间的算子进行参数更新，会导致每次更新都会在计算图上增加一个新的需要求梯度的节点w，这个节点具有Op和inputs，严重拖累反向传播速度。

为了避免这种情况，needle库提供了`Tensor.data()`方法，用于创建一个与`Tensor`共享同一个底层data的节点，但其不存在Op和inputs，也不用对其进行求导，能在不干扰计算图反向传播的前提下对参数进行正常的更新，即：

```python
w.data -= lr * grad.data
```



#### 数值稳定性

每个数值在内存中的存储空间有限，保存的数值的范围和精度都有限，计算过程中难免出现溢出或者精度丢失的情况。

例如在softmax公式中，由于指数运算的存在，数值很有可能上溢，一个修正方式是在进行softmax运算前，每个元素都减去输入的最大值，以防止上溢。即：
$$
z_i = \frac{exp(x_i)}{\sum_k exp(x_k)} = \frac{exp(x_i-c)}{\sum_k exp(x_k-c)}
$$
其中 $c = max(x)$ 






## 参考

1. [CMU 10-414 Assignments 实验笔记](https://www.zhouxin.space/notes/notes-on-cmu-10-414-assignments/)
2. [深度学习系统作业 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1582462878204063744)





