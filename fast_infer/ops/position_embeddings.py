import jax.numpy as jnp
from jaxtyping import Array, Float
import dataclasses
import flax.linen as nn
from enum import Enum
import jax


class PositionEmbeddingKind(Enum):
    RoPE = "RoPE"


@dataclasses.dataclass
class RoPEParams:
    freqs: Float[Array, "seq_len d_model"]


@dataclasses.dataclass
class RoPEConfig:
    max_seq_len: int
    head_dim: int
    has_groups_dim: bool = False


def rotate_half(x: Float[Array, "b s d"]) -> Float[Array, "b s d"]:
    # This is not the original rope impl, but is what works for huggingface's llama (TODO -- investigate why they do this)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.swapaxes(jnp.stack([-x2, x1], axis=-1), -1, -2).reshape(x.shape)


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
    x: Float[Array, "bs heads seq_len d_model"],
    pos: int,
    params: RoPEParams,
    config: RoPEConfig,
) -> Float[Array, "bs heads seq_len_q d_v"]:
    seq_len = x.shape[2]
    freq_cos, freq_sin = jnp.cos(params.freqs), jnp.sin(params.freqs)
    # use dynamic slicing instead of the commented out code above
    freq_cos, freq_sin = (
        jax.lax.dynamic_slice(freq_cos, (pos, 0), (seq_len, x.shape[-1])),
        jax.lax.dynamic_slice(freq_sin, (pos, 0), (seq_len, x.shape[-1])),
    )
    freq_cos, freq_sin = (
        freq_cos[None, None, ...],
        freq_sin[None, None, ...],
    )  # broadcast to batch and heads
    return x * freq_cos + rotate_half(x) * freq_sin


class RoPE(nn.Module):
    config: RoPEConfig

    @nn.compact
    def __call__(
        self, xs: list[Float[Array, "bs seq_len d_model"]], positions: list[int]
    ) -> list[Float[Array, "bs seq_len d_v"]]:
        freqs = self.param(
            "inv_freqs",
            lambda *args: create_rope_freqs(
                self.config.max_seq_len, self.config.head_dim
            ),
        )
        return [
            rope(x, pos, RoPEParams(freqs=freqs), self.config)
            for x, pos in zip(xs, positions)
        ]


pos_embedding_kind_to_fn = {
    PositionEmbeddingKind.RoPE: RoPE,
}
