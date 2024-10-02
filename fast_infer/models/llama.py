import jax.numpy as jnp
from jaxtyping import Array, Float
import flax.linen as nn
from fast_infer.ops.attention import Attention, AttentionConfig
from fast_infer.ops.mlp import MLP, MLPConfig
from fast_infer.generic import Activation
import fast_infer.utils as utils


@utils.auto_init_dataclass
class LlamaConfig:
    n_q_heads: int
    n_kv_heads: int
    d_model: int
    d_k: int
    d_v: int
    activation: Activation
    hidden_dim: int
    output_dim: int
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
    ) -> Float[Array, "bs seq_len d_model"]:
        if self.config.pre_attention_layernorm:
            x = nn.LayerNorm()(x)
        residual = x
        x = nn.LayerNorm()(x)
        x = Attention(config=AttentionConfig(**self.config.to_dict()))(x, x, x)
        x = residual + x
        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(config=MLPConfig(**self.config.to_dict()))(x)
        x = residual + x

        if self.config.post_attention_layernorm:
            x = nn.LayerNorm()(x)
        return x


class LlamaModel(nn.Module):
    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "bs seq_len d_model"],
        mask: Float[Array, "bs seq_len seq_len"],
    ) -> Float[Array, "bs seq_len d_model"]:
        # embedding
        x = nn.Embed(
            num_embeddings=self.config.vocab_size, features=self.config.d_model
        )(x)
        for _ in range(self.config.n_layers):
            x = LlamaDecoderLayer(config=self.config)(x, mask)
            print(x.shape)
        x = nn.RMSNorm()(x)
        # lm head
        whead = self.param(
            "lm_head",
            lambda rng, shape: jnp.zeros(shape),
            (self.config.d_model, self.config.vocab_size),
        )
        return jnp.dot(x, whead)
