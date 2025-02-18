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

## Automatic Differentiation Implementation

现代机器学习框架可以视为两层：上层是计算图，用于前向推理、自动微分和反向传播；下层是张量线性代数库，其负责底层的张量计算。

`autograd.py`实现自动微分相关的代码。其中最重要的两个类：`Value`类代表计算图上的节点，`Op`类代表各种算子

```python
class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool
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
        raise NotImplementedError()

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
        raise NotImplementedError()

```

其他的内容参见代码即可。


## Neural Network Library Implementation

### 修改tensor的data域

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



### 数值稳定性

每个数值在内存中的存储空间有限，保存的数值的范围和精度都有限，计算过程中难免出现溢出或者精度丢失的情况。

例如在softmax公式中，由于指数运算的存在，数值很有可能上溢，一个修正方式是在进行softmax运算前，每个元素都减去输入的最大值，以防止上溢。即：
$$
z_i = \frac{exp(x_i)}{\sum_k exp(x_k)} = \frac{exp(x_i-c)}{\sum_k exp(x_k-c)}
$$
其中 $c = max(x)$ 



### 重要的类实现

`Parameter`类用于表示可学习的参数，其是`Tensor`的子类。相比`Tensor`类，这个类不必再引入新的行为或者接口。

`Module`类用于表示神经网络中一个个子模块。

```python
def _get_params(value):
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _get_params(v)
        return params
    if isinstance(value, Module):
        return value.parameters()
    return []

class Module:
	# 获取模块中所有的可学习的参数
    def parameters(self):
        return _get_params(self.__dict__)

	# 进行前向传播
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

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

此外还实现了`TensorTuple`类，能返回多个`Value`。


## 参考

1. [CMU 10-414 Assignments 实验笔记](https://www.zhouxin.space/notes/notes-on-cmu-10-414-assignments/)
2. [深度学习系统作业 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1582462878204063744)





