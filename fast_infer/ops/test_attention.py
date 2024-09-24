import pytest
import jax
from attention import scaled_dot_product_attention, AttentionConfig, AttentionParams


@pytest.fixture
def sample_attention_params():
    d_model = 64
    n_q_heads = 8
    n_kv_heads = 8
    d_k = d_model // n_q_heads
    d_v = d_model // n_kv_heads
    params = AttentionParams(
        query=jax.random.normal(jax.random.PRNGKey(0), (d_model, n_q_heads * d_k)),
        key=jax.random.normal(jax.random.PRNGKey(1), (d_model, n_kv_heads * d_k)),
        value=jax.random.normal(jax.random.PRNGKey(2), (d_model, n_kv_heads * d_v)),
        wo=jax.random.normal(jax.random.PRNGKey(3), (n_kv_heads * d_v, d_model)),
    )
    return params


@pytest.fixture
def sample_attention_config():
    return AttentionConfig(n_q_heads=8, n_kv_heads=8, d_model=64, d_k=8, d_v=8)


@pytest.fixture
def sample_inputs():
    bs = 4  # batch size
    seq_len_q = 10  # sequence length for queries
    seq_len_k = 10  # sequence length for keys
    seq_len_v = 10  # sequence length for values
    d_model = 64  # embedding dimension

    query = jax.random.normal(jax.random.PRNGKey(3), (bs, seq_len_q, d_model))
    key = jax.random.normal(jax.random.PRNGKey(4), (bs, seq_len_k, d_model))
    value = jax.random.normal(jax.random.PRNGKey(5), (bs, seq_len_v, d_model))

    return query, key, value


def test_scaled_dot_product_attention(
    sample_attention_params, sample_attention_config, sample_inputs
):
    query, key, value = sample_inputs

    expected_output_shape = (
        query.shape[0],
        query.shape[1],
        sample_attention_config.d_model,
    )

    output = scaled_dot_product_attention(
        query, key, value, sample_attention_params, sample_attention_config
    )
    assert (
        output.shape == expected_output_shape
    ), f"Expected {expected_output_shape}, but got {output.shape}"
