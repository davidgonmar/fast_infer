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
    output_dim: int
    activation: Activation


@dataclasses.dataclass
class MLPParams:
    w1: Float[Array, "d_model hidden_dim"]
    w2: Float[Array, "hidden_dim output_dim"]
    b1: Float[Array, "hidden_dim"]
    b2: Float[Array, "output_dim"]


def mlp_forward(
    x: Float[Array, "bs d_model"], params: MLPParams, config: MLPConfig
) -> Float[Array, "bs output_dim"]:
    h = jnp.dot(x, params.w1) + params.b1
    if config.activation == Activation.RELU:
        h = jax.nn.relu(h)
    elif config.activation == Activation.GELU:
        h = jax.nn.gelu(h)
    output = jnp.dot(h, params.w2) + params.b2
    return output


class MLP(nn.Module):
    config: MLPConfig

    @nn.compact
    def __call__(self, x: Float[Array, "bs d_model"]) -> Float[Array, "bs output_dim"]:
        w1 = self.param(
            "w1",
            lambda rng, shape: jax.random.normal(rng, shape),
            (self.config.d_model, self.config.hidden_dim),
        )
        b1 = self.param(
            "b1",
            lambda rng, shape: jax.random.normal(rng, shape),
            (self.config.hidden_dim,),
        )
        w2 = self.param(
            "w2",
            lambda rng, shape: jax.random.normal(rng, shape),
            (self.config.hidden_dim, self.config.output_dim),
        )
        b2 = self.param(
            "b2",
            lambda rng, shape: jax.random.normal(rng, shape),
            (self.config.output_dim,),
        )

        params = MLPParams(w1=w1, w2=w2, b1=b1, b2=b2)
        return mlp_forward(x, params, self.config)
