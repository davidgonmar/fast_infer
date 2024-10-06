import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
import dataclasses
import flax.linen as nn
import fast_infer.utils as utils
from fast_infer.ops.position_embeddings import (
    pos_embedding_kind_to_fn,
    PositionEmbeddingKind,
    RoPEConfig,
)


@utils.auto_init_dataclass
class AttentionConfig:
    # heads (grouped-query-attention, multi-head-attention or regular)
    n_q_heads: int
    n_kv_heads: int
    d_model: int
    d_k: int
    d_v: int
    scale: float | None = None
    pos_embedding: PositionEmbeddingKind | None = PositionEmbeddingKind.RoPE


class AttentionCache:
    def __init__(self, config: AttentionConfig):
        self._cfg = config
        self._cache = {}

    def add_kv(
        self,
        layer_name: str,
        key: Float[Array, "bs head seq_len_k d_model"],
        value: Float[Array, "bs head seq_len_k d_model"],
    ):
        if layer_name not in self._cache:
            self._cache[layer_name] = {"key": key, "value": value}
        else:
            self._cache[layer_name]["key"] = jnp.concatenate(
                [self._cache[layer_name]["key"], key], axis=2
            )
            self._cache[layer_name]["value"] = jnp.concatenate(
                [self._cache[layer_name]["value"], value], axis=2
            )

    def get_kv(self, layer_name: str) -> tuple[
        Float[Array, "bs head seq_len_k d_model"],
        Float[Array, "bs head seq_len_k d_model"],
    ]:
        assert layer_name in self._cache, "Cache for this layer is empty"
        return self._cache[layer_name]["key"], self._cache[layer_name]["value"]

    def reset(self, layer_name: str):
        if layer_name in self._cache:
            del self._cache[layer_name]

    def reset_all(self):
        self._cache.clear()


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
    mask: Float[Array, "bs seq_len_q seq_len_k"],
    cache: AttentionCache | None,
    config: AttentionConfig,
    curr_seq_pos: int,
    layer_name: str,
) -> Float[Array, "bs seq_len_q d_v"]:
    bs = query.shape[0]
    n_groups = config.n_q_heads // config.n_kv_heads
    query = (
        (query @ params.query)
        .reshape(bs, query.shape[1], config.n_q_heads, -1)
        .transpose((0, 2, 1, 3))
    )  # (bs, n_q_heads, seq_len_q, d_k)
    key = (
        (key @ params.key)
        .reshape(bs, key.shape[1], config.n_kv_heads, 1, -1)
        .repeat(n_groups, axis=3)
        .reshape(bs, key.shape[1], config.n_q_heads, -1)
        .transpose((0, 2, 1, 3))
    )  # (bs, n_q_heads, seq_len_k, d_k)
    value = (
        (value @ params.value)
        .reshape(bs, value.shape[1], config.n_kv_heads, 1, -1)
        .repeat(n_groups, axis=3)
        .reshape(bs, value.shape[1], config.n_q_heads, -1)
        .transpose((0, 2, 1, 3))
    )  # (bs, n_q_heads, seq_len_v, d_v)

    dk = query.shape[-1]
    scale = config.scale or jnp.sqrt(dk)

    query, key = pos_embedding_kind_to_fn[config.pos_embedding](
        RoPEConfig(**{"max_seq_len": 1024, "head_dim": dk, "has_groups_dim": False})
    )([query, key], [curr_seq_pos, curr_seq_pos])

    if cache is not None:
        cache.add_kv(layer_name, key, value)
        key, value = cache.get_kv(layer_name)

    raw_scores = (
        query @ key.swapaxes(-1, -2) / scale
    )  # (bs, n_q_heads, seq_len_q, seq_len_k)
    masked = (
        raw_scores
        + mask[:bs, None, curr_seq_pos : curr_seq_pos + query.shape[2], : key.shape[2]]
    )

    attention_weights = jax.nn.softmax(
        masked, axis=-1
    )  # (bs, n_q_heads, seq_len_q, seq_len_k)
    output = attention_weights @ value  # (bs, n_q_heads, seq_len_q, d_v)
    output = output.transpose((0, 2, 1, 3)).reshape(
        bs, query.shape[2], config.n_q_heads * config.d_v
    )  # (bs, seq_len_q, n_q_heads * d_v)
    return output @ params.wo  # (bs, seq_len_q, d_model)


class Attention(nn.Module):
    config: AttentionConfig

    @nn.compact
    def __call__(
        self,
        query: Float[Array, "bs seq_len_q d_model"],
        key: Float[Array, "bs seq_len_k d_model"],
        value: Float[Array, "bs seq_len_v d_model"],
        mask: Float[Array, "bs seq_len_q seq_len_k"],
        cache: AttentionCache | None,
        curr_seq_pos: int,
    ) -> Float[Array, "bs seq_len_q d_v"]:

        wq = self.param(
            "wq",
            lambda rng: jax.random.normal(
                rng, (self.config.d_model, self.config.n_q_heads * self.config.d_k)
            ),
        )
        wk = self.param(
            "wk",
            lambda rng: jax.random.normal(
                rng, (self.config.d_model, self.config.n_kv_heads * self.config.d_k)
            ),
        )
        wv = self.param(
            "wv",
            lambda rng: jax.random.normal(
                rng, (self.config.d_model, self.config.n_kv_heads * self.config.d_v)
            ),
        )
        wo = self.param(
            "wo",
            lambda rng: jax.random.normal(
                rng, (self.config.n_q_heads * self.config.d_v, self.config.d_model)
            ),
        )
        params = AttentionParams(query=wq, key=wk, value=wv, wo=wo)
        return scaled_dot_product_attention(
            query,
            key,
            value,
            params,
            mask,
            cache,
            self.config,
            curr_seq_pos,
            "".join(self.path),
        )
