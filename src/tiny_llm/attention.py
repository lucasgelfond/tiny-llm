import mlx.core as mx
from mlx.core import sqrt, zeros
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    d_k = query.shape[-1] # last dimension of query tensor
    scale_factor = 1 / sqrt(d_k) if scale is None else scale

    attention_weight = query @ key.swapaxes(-2, -1) * scale_factor
    if mask is not None:
        attention_weight = attention_weight + mask

    attention_weight = softmax(attention_weight, axis=-1)
    # skip dropout
    return attention_weight @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # query 
        self.wq = wq
        # key
        self.wk = wk
        #value 
        self.wv = wv
        # output - this is used to concatenate the heads 
        self.wo = wo
        

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:

        # each of these starts as L x E --> H*D (i.e. N, L, H*D)
        # we need to reshape for them to be (N, L, H, D)
        query_proj = linear(query, self.wq)
        key_proj = linear(key, self.wk)
        value_proj = linear(value, self.wv)

        query_heads = query_proj.reshape(query.shape[0], query.shape[1], self.num_heads, -1)
        key_heads = key_proj.reshape(key.shape[0], key.shape[1], self.num_heads, -1)
        value_heads = value_proj.reshape(value.shape[0], value.shape[1], self.num_heads, -1)
        print(query.shape, key.shape, value.shape)

        # transposed for attention
        # go from N x L x H x D --> N x H x L x D
        query_heads = query_heads.swapaxes(1, 2)
        key_heads = key_heads.swapaxes(1, 2)
        value_heads = value_heads.swapaxes(1, 2)

        attention_output = scaled_dot_product_attention_simple(
            query_heads, key_heads, value_heads, mask=mask
        )

        # merge heads back togehter - do the opposite
        attention_output = attention_output.swapaxes(1, 2)
        # reshape as well, back from N, L, H, D --> N, L, H*D
        attention_output = attention_output.reshape(attention_output.shape[0], attention_output.shape[1], -1)

        # final linear layer - combine heads
        return linear(attention_output, self.wo)

     



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
