import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
import dataclasses
import flax.linen as nn


@dataclasses.dataclass
class AttentionConfig:
    # heads (grouped-query-attention, multi-head-attention or regular)
    n_q_heads: int
    n_kv_heads: int
    d_model: int
    d_k: int
    d_v: int
    scale: float | None = None


@dataclasses.dataclass
class AttentionParams:
    query: Float[Array, "d_model (n_q_heads d_k)"]
    key: Float[Array, "d_model (n_kv_heads d_k)"]
    value: Float[Array, "d_model (n_kv_heads d_v)"]
    wo: Float[Array, "(n_kv_heads d_v) d_model"]


def scaled_dot_product_attention(
    query: Float[Array, "bs seq_len_q d_model"],
    key: Float[Array, "bs seq_len_k d_model"],
    value: Float[Array, "bs seq_len_v d_model"],
    params: AttentionParams,
    config: AttentionConfig,
) -> Float[Array, "bs seq_len_q d_v"]:
    bs, seq_len = query.shape[0], query.shape[1]
    query = (
        (query @ params.query)
        .reshape(bs, seq_len, config.n_q_heads, -1)
        .transpose((0, 2, 1, 3))
    )  # (bs, n_q_heads, seq_len_q, d_k)
    key = (
        (key @ params.key)
        .reshape(bs, seq_len, config.n_kv_heads, -1)
        .transpose((0, 2, 1, 3))
    )  # (bs, n_kv_heads, seq_len_k, d_k)
    value = (
        (value @ params.value)
        .reshape(bs, seq_len, config.n_kv_heads, -1)
        .transpose((0, 2, 1, 3))
    )  # (bs, n_kv_heads, seq_len_v, d_v)
    dk = query.shape[-1]
    scale = config.scale or jnp.sqrt(dk)
    attention_weights = jax.nn.softmax(
        (query @ key.transpose((0, 1, 3, 2))) / scale, axis=-1
    )  # (bs, n_q_heads, seq_len_q, seq_len_k)
    output = attention_weights @ value  # (bs, n_q_heads, seq_len_q, d_v)
    output = output.transpose((0, 2, 1, 3)).reshape(
        bs, seq_len, -1
    )  # (bs, seq_len_q, d_v)
    return output @ params.wo  # (bs, seq_len_q, d_model)


class Attention(nn.Module):
    config: AttentionConfig

    @nn.compact
    def __call__(
        self,
        query: Float[Array, "bs seq_len_q d_model"],
        key: Float[Array, "bs seq_len_k d_model"],
        value: Float[Array, "bs seq_len_v d_model"],
    ) -> Float[Array, "bs seq_len_q d_v"]:
        wq = self.param("query", lambda rng, shape: jax.random.normal(rng, shape))
        wk = self.param("key", lambda rng, shape: jax.random.normal(rng, shape))
        wv = self.param("value", lambda rng, shape: jax.random.normal(rng, shape))
        wo = self.param("wo", lambda rng, shape: jax.random.normal(rng, shape))

        params = AttentionParams(query=wq, key=wk, value=wv, wo=wo)
        return scaled_dot_product_attention(query, key, value, params, self.config)
