import jax.numpy as jnp
from jaxtyping import Array, Float
import dataclasses
import flax.linen as nn
from enum import Enum


class PositionEmbeddingKind(Enum):
    RoPE = "RoPE"


@dataclasses.dataclass
class RoPEParams:
    freqs: Float[Array, "seq_len d_model"]


@dataclasses.dataclass
class RoPEConfig:
    max_seq_len: int
    head_dim: int


def rotate_half(x: Float[Array, "b s d"]) -> Float[Array, "b s d"]:
    # This is not the original rope impl, but is what works for huggingface's llama (TODO -- investigate why they do this)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.stack([-x2, x1], axis=-1).transpose(0, 1, 2, 4, 3).reshape(x.shape)


def create_rope_freqs(
    max_seq_len: int, head_dim: int
) -> Float[Array, "seq_len d_model"]:
    d_model = head_dim
    dims = jnp.arange(d_model // 2)
    seq_lens = jnp.arange(max_seq_len)
    freqs = 1.0 / jnp.power(10000, 2 * dims / d_model)
    # repeat interleave so freqs is freq1, freq1, freq2, freq2, ...
    mat = jnp.outer(seq_lens, freqs)  # (seq_len, d_model // 2)
    return jnp.concatenate([mat, mat], axis=-1)  # (seq_len, d_model)


def rope(
    x: Float[Array, "bs heads seq_len d_model"], params: RoPEParams, config: RoPEConfig
) -> Float[Array, "bs heads seq_len_q d_v"]:
    seq_len = x.shape[2]
    freq_cos, freq_sin = jnp.cos(params.freqs), jnp.sin(params.freqs)
    freq_cos, freq_sin = (
        freq_cos[None, None, :seq_len, :],
        freq_sin[None, None, :seq_len, :],
    )
    return x * freq_cos + rotate_half(x) * freq_sin


class RoPE(nn.Module):
    config: RoPEConfig

    @nn.compact
    def __call__(
        self, xs: list[Float[Array, "bs seq_len d_model"]]
    ) -> list[Float[Array, "bs seq_len d_v"]]:
        freqs = self.param(
            "inv_freqs",
            lambda *args: create_rope_freqs(
                self.config.max_seq_len, self.config.head_dim
            ),
        )
        return [rope(x, RoPEParams(freqs=freqs), self.config) for x in xs]


pos_embedding_kind_to_fn = {
    PositionEmbeddingKind.RoPE: RoPE,
}
