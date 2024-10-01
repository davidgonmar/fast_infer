import jax.numpy as jnp
from jaxtyping import Array, Float
import flax.linen as nn
from fast_infer.ops.attention import Attention, AttentionConfig
from fast_infer.ops.mlp import MLP, MLPConfig
from fast_infer.generic import Activation
import fast_infer.utils as utils


@utils.auto_init_dataclass
class LlamaDecoderLayerConfig:
    n_q_heads: int
    n_kv_heads: int
    d_model: int
    d_k: int
    d_v: int
    activation: Activation
    hidden_dim: int
    output_dim: int
    scale: float | None = None


class LlamaDecoderLayer(nn.Module):
    config: LlamaDecoderLayerConfig

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "bs seq_len d_model"],
        mask: Float[Array, "bs seq_len seq_len"],
    ) -> Float[Array, "bs seq_len d_model"]:
        residual = x
        x = nn.LayerNorm()(x)
        x = Attention(config=AttentionConfig(**self.config.to_dict()))(x, x, x)
        x = residual + x
        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(config=MLPConfig(**self.config.to_dict()))(x)
        x = residual + x
        return x


class LlamaModel(nn.Module):
    config: LlamaDecoderLayerConfig
    num_layers: int

    @nn.compact
    def __call__(
        self,
        x: Float[Array, "bs seq_len d_model"],
        mask: Float[Array, "bs seq_len seq_len"],
    ) -> Float[Array, "bs seq_len d_model"]:
        for _ in range(self.num_layers):
            x = LlamaDecoderLayer(config=self.config)(x, mask)
        return x
