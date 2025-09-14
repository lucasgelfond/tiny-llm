import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # For linear, it takes a tensor of the shape N.. x I, a weight matrix of the shape O x I, and a bias vector of the shape O. The output is of the shape N.. x O. I is the input dimension and O is the output dimension.
    output = x @ w.swapaxes(0,1) + (bias if bias is not None else 0)
    return output


def silu(x: mx.array) -> mx.array:
    pass
