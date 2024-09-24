import pytest
import jax
from mlp import MLP, MLPConfig, Activation


@pytest.fixture
def sample_mlp_config():
    return MLPConfig(
        d_model=64, hidden_dim=128, output_dim=64, activation=Activation.RELU
    )


@pytest.fixture
def sample_input():
    bs = 4  # batch size
    d_model = 64  # input dimension
    return jax.random.normal(jax.random.PRNGKey(0), (bs, d_model))


def test_mlp_forward(sample_mlp_config, sample_input):
    model = MLP(config=sample_mlp_config)

    rng = jax.random.PRNGKey(1)
    variables = model.init(rng, sample_input)
    output = model.apply(variables, sample_input)

    expected_output_shape = (sample_input.shape[0], sample_mlp_config.output_dim)
    assert (
        output.shape == expected_output_shape
    ), f"Expected {expected_output_shape}, but got {output.shape}"
