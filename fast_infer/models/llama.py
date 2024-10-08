import jax.numpy as jnp
from jaxtyping import Array, Float
import flax.linen as nn
from fast_infer.ops.attention import Attention, AttentionConfig, AttentionCache
from fast_infer.ops.mlp import MLP, MLPConfig
from fast_infer.generic import Activation
import fast_infer.utils as utils


class RMSNorm(nn.Module):
    dtype: jnp.dtype = jnp.float32
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        weight = self.param(
            "scale", lambda rng, shape: jnp.ones(shape), hidden_states.shape[-1]
        )
        return weight * jnp.asarray(hidden_states, dtype=self.dtype)


nn.RMSNorm = RMSNorm


@utils.auto_init_dataclass
class LlamaConfig:
    n_q_heads: int
    n_kv_heads: int
    d_model: int
    d_k: int
    d_v: int
    activation: Activation
    hidden_dim: int
    vocab_size: int
    n_layers: int
    scale: float | None = None

    pre_attention_layernorm: bool = True
    post_attention_layernorm: bool = True


class LlamaDecoderLayer(nn.Module):
    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "bs seq_len d_model"],
        mask: Float[Array, "bs seq_len seq_len"],
        cache: AttentionCache | None,
        curr_seq_pos: int,
    ) -> Float[Array, "bs seq_len d_model"]:
        residual = x
        if self.config.pre_attention_layernorm:
            x = nn.RMSNorm()(x)
        x = Attention(config=AttentionConfig(**self.config.to_dict()))(
            x, x, x, mask, cache, curr_seq_pos
        )
        x = residual + x
        residual = x
        if self.config.post_attention_layernorm:
            x = nn.RMSNorm()(x)
        x = MLP(config=MLPConfig(**self.config.to_dict()))(x)
        x = residual + x
        return x


class LlamaModel(nn.Module):
    config: LlamaConfig
    lm_head: bool = True

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "bs seq_len d_model"],
        mask: Float[Array, "bs seq_len seq_len"],
        cache: AttentionCache | None,
        curr_seq_pos: int,
    ):
        # embedding
        x = nn.Embed(
            num_embeddings=self.config.vocab_size, features=self.config.d_model
        )(x)
        for _ in range(self.config.n_layers):
            x = LlamaDecoderLayer(config=self.config)(x, mask, cache, curr_seq_pos)
        x = nn.RMSNorm(epsilon=1e-6)(x)
        if not self.lm_head:
            return x
        # lm head
        whead = self.param(
            "lm_head",
            lambda rng, shape: jnp.zeros(shape),
            (self.config.d_model, self.config.vocab_size),
        )
        return jnp.dot(x, whead), cache
