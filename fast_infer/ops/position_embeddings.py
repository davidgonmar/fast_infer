import jax.numpy as jnp
from jaxtyping import Array, Float
import dataclasses
import flax.linen as nn


@dataclasses.dataclass
class RoPEParams:
    freqs: Float[Array, "seq_len d_model"]


@dataclasses.dataclass
class RoPEConfig:
    max_seq_len: int
    d_model: int


def rotate_half(x: Float[Array, "b s d"]) -> Float[Array, "b s d"]:
    x1 = x[:, :, 1::2]
    x2 = x[:, :, ::2]
    return jnp.stack([-x2, x1], axis=-1).reshape(x.shape)


def create_rope_freqs(max_seq_len: int, d_model: int) -> RoPEParams:
    dims = jnp.arange(d_model // 2)
    seq_lens = jnp.arange(max_seq_len)
    freqs = 1 / jnp.power(10000, 2 * dims / d_model)
    # repeat interleave so freqs is freq1, freq1, freq2, freq2, ...
    freqs = jnp.repeat(freqs, 2)
    return jnp.outer(seq_lens, freqs)


def rope(
    x: Float[Array, "bs seq_len d_model"], params: RoPEParams, config: RoPEConfig
) -> Float[Array, "bs seq_len_q d_v"]:
    seq_len = x.shape[1]
    freq_cos, freq_sin = jnp.cos(params.freqs), jnp.sin(params.freqs)
    freq_cos, freq_sin = freq_cos[:, :seq_len], freq_sin[:, :seq_len]
    return x * freq_cos + rotate_half(x) * freq_sin


class RoPE(nn.Module):
    config: RoPEConfig

    @nn.compact
    def __call__(
        self, x: Float[Array, "bs seq_len d_model"]
    ) -> Float[Array, "bs seq_len d_model"]:
        freqs = self.param(
            "freqs",
            lambda *args: create_rope_freqs(
                self.config.max_seq_len, self.config.d_model
            ),
        )
        return rope(x, RoPEParams(freqs=freqs), self.config)
