import jax
import torch
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half as rotate_half_hf,
)
import numpy as np
from fast_infer.ops.position_embeddings import (
    RoPEConfig,
    RoPE,
    rotate_half as rotate_half_fi,
)


def check_all_close_or_print(out1, out2, atol, name):
    out1 = out1.numpy() if hasattr(out1, "numpy") else out1
    out2 = out2.numpy() if hasattr(out2, "numpy") else out2

    if not np.allclose(out1, out2, atol=atol):
        mismatch_idxs = np.argwhere(np.abs(out1 - out2) > atol)

        mismatch_idxs = mismatch_idxs[:10]

        print(f"First {len(mismatch_idxs)} mismatches (index, out1 value, out2 value):")

        for idx in mismatch_idxs:
            idx_tuple = tuple(idx)
            print(
                f"Index: {idx_tuple}, out1: {out1[idx_tuple]}, out2: {out2[idx_tuple]}"
            )

        raise AssertionError("Outputs do not match for " + name)


def test_forward():
    config = RoPEConfig(max_seq_len=128, head_dim=64)
    fi_fn = RoPE(config)
    hf_fn = LlamaRotaryEmbedding(64)

    input = np.random.rand(4, 8, 128, 64).astype(
        np.float32
    )  # shape (bs, n_heads, seq_len, head_dim)
    input = (
        np.arange(64)
        .reshape(1, 1, 1, 64)
        .astype(np.float32)
        .repeat(4, axis=0)
        .repeat(8, axis=1)
        .repeat(128, axis=2)
    )
    position_ids = torch.arange(128).unsqueeze(0).expand(4, 128).long()

    rng = jax.random.PRNGKey(0)

    params = fi_fn.init(rng, [input])

    # first test rotate_half
    hf_output = rotate_half_hf(torch.tensor(input)).detach().numpy()
    custom_output = rotate_half_fi(input)

    assert (
        hf_output.shape == custom_output.shape
    ), f"Expected {hf_output.shape}, but got {custom_output.shape}"

    check_all_close_or_print(hf_output, custom_output, atol=1e-5, name="rotate_half")

    # second see frequency

    inv_freqs_hf = torch.outer(torch.arange(128), hf_fn.inv_freq)
    inv_freqs_hf = torch.cat([inv_freqs_hf, inv_freqs_hf], dim=-1)
    inv_freqs_fi = params["params"]["inv_freqs"]
    check_all_close_or_print(inv_freqs_hf, inv_freqs_fi, atol=1e-5, name="inv_freqs")
    cos, sin = hf_fn(torch.tensor(input), position_ids)
    hf_output = (
        apply_rotary_pos_emb(torch.tensor(input), torch.tensor(input), cos, sin)[0]
        .detach()
        .numpy()
    )

    custom_output = fi_fn.apply(params, [input])[0]
    assert (
        hf_output.shape == custom_output.shape
    ), f"Expected {hf_output.shape}, but got {custom_output.shape}"
    check_all_close_or_print(hf_output, custom_output, atol=1e-3, name="RoPE forward")
