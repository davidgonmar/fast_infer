import pytest
import jax
from position_embeddings import RoPE, RoPEConfig


@pytest.fixture
def sample_rope_config():
    return RoPEConfig(max_seq_len=128, d_model=64)


@pytest.fixture
def sample_rope_input():
    bs = 4  # batch size
    seq_len = 128  # sequence length
    d_model = 64  # embedding dimension
    return jax.random.normal(jax.random.PRNGKey(0), (bs, seq_len, d_model))


def test_rope_forward(sample_rope_config, sample_rope_input):
    model = RoPE(config=sample_rope_config)

    rng = jax.random.PRNGKey(1)
    variables = model.init(rng, sample_rope_input)
    output = model.apply(variables, sample_rope_input)

    expected_output_shape = sample_rope_input.shape
    assert (
        output.shape == expected_output_shape
    ), f"Expected {expected_output_shape}, but got {output.shape}"
