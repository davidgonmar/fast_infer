import pytest
import jax
import jax.numpy as jnp
from fast_infer.models.llama import LlamaDecoderLayerConfig, LlamaModel
from fast_infer.generic import Activation


@pytest.mark.parametrize(
    "n_q_heads, n_kv_heads, d_model, hidden_dim, output_dim, num_layers",
    [
        (8, 8, 512, 2048, 512, 4),
        (16, 16, 1024, 4096, 1024, 6),
        (4, 4, 256, 1024, 256, 2),
    ],
)
def test_llama_model_different_params(
    n_q_heads, n_kv_heads, d_model, hidden_dim, output_dim, num_layers
):
    config = LlamaDecoderLayerConfig(
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_model=d_model,
        d_k=64,
        d_v=64,
        activation=Activation.GELU,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    llama_model = LlamaModel(config=config, num_layers=num_layers)

    batch_size = 2
    seq_len = 10
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    mask = jnp.ones((batch_size, seq_len, seq_len))

    params = llama_model.init(jax.random.PRNGKey(0), x, mask)
    output = llama_model.apply(params, x, mask)

    assert output.shape == (batch_size, seq_len, d_model)
