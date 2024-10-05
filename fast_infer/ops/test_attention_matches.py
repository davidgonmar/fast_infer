import pytest
import jax
import jax.numpy as jnp
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig
import numpy as np
from fast_infer.ops.attention import Attention, AttentionConfig


def test_attention_forward():
    config = AttentionConfig(n_q_heads=8, n_kv_heads=8, d_model=512, d_k=64, d_v=64)
    custom_attention = Attention(config)

    cfg = LlamaConfig(num_attention_heads=8, num_key_value_heads=8, hidden_size=512)
    hf_attention_layer = LlamaAttention(cfg)

    input = np.random.rand(4, 128, 512).astype(
        np.float32
    )  # shape (bs, seq_len, d_model)
    mask = np.ones((4, 128, 128)).astype(np.float32)  # shape (bs, seq_len, seq_len)
    # Hugging Face LLaMA forward pass

    hf_output = (
        hf_attention_layer(torch.tensor(input), torch.tensor(mask))[0].detach().numpy()
    )

    # Custom attention forward pass
    rng = jax.random.PRNGKey(0)
    params = custom_attention.init(rng, input, input, input, mask)
    custom_output = custom_attention.apply(params, input, input, input, mask, config)

    # Assert that the outputs are close
    assert np.allclose(
        hf_output, np.array(custom_output), atol=1e-5
    ), "Outputs do not match!"
