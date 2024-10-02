import jax.numpy as jnp
from jaxtyping import Array, Float
import jax
import flax.linen as nn
import dataclasses
from fast_infer.generic import Activation
import fast_infer.utils as utils


@utils.auto_init_dataclass
class MLPConfig:
    d_model: int
    hidden_dim: int
    activation: Activation


@dataclasses.dataclass
class MLPParams:
    gate_proj: Float[Array, "hidden_dim intermediate_size"]
    up_proj: Float[Array, "hidden_dim intermediate_size"]
    down_proj: Float[Array, "intermediate_size hidden_dim"]


def mlp_forward(
    x: Float[Array, "bs d_model"], params: MLPParams, config: MLPConfig
) -> Float[Array, "bs output_dim"]:
    h = jnp.dot(x, params.gate_proj)
    if config.activation == Activation.RELU:
        h = jax.nn.relu(h)
    elif config.activation == Activation.GELU:
        h = jax.nn.gelu(h)
    elif config.activation == Activation.SILU:
        h = jax.nn.silu(h)
    output = jnp.dot(x, params.up_proj) * h
    return jnp.dot(output, params.down_proj)


class MLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, x: Float[Array, "bs d_model"]) -> Float[Array, "bs output_dim"]:
        gate_proj = self.param(
            "gate_proj",
            lambda rng, shape: nn.initializers.xavier_uniform()(rng, shape),
            (self.config.d_model, self.config.hidden_dim),
        )
        up_proj = self.param(
            "up_proj",
            lambda rng, shape: nn.initializers.xavier_uniform()(rng, shape),
            (self.config.d_model, self.config.hidden_dim),
        )
        down_proj = self.param(
            "down_proj",
            lambda rng, shape: nn.initializers.xavier_uniform()(rng, shape),
            (self.config.hidden_dim, self.config.d_model),
        )
        return mlp_forward(
            x,
            MLPParams(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj),
            self.config,
        )
