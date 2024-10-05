import jax
import jax.numpy as jnp
import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig
import numpy as np
from fast_infer.ops.attention import Attention, AttentionConfig


def check_all_close_or_print(out1, out2, atol, name):
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().numpy()
        elif hasattr(x, "numpy"):
            return x.numpy()
        return x

    out1 = _to_numpy(out1)
    out2 = _to_numpy(out2)

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
    D_MODEL = 8 * 64
    N_Q_HEADS = 8
    N_KV_HEADS = 8
    D_K = 64
    D_V = 64
    SEQ_LEN = 44
    BSIZE = 4
    config = AttentionConfig(
        n_q_heads=N_Q_HEADS, n_kv_heads=N_KV_HEADS, d_model=D_MODEL, d_k=D_K, d_v=D_V
    )
    fi_fn = Attention(config)
    hf_fn = LlamaAttention(
        LlamaConfig(
            num_attention_heads=N_Q_HEADS,
            num_key_value_heads=N_KV_HEADS,
            hidden_size=D_MODEL,
        )
    )

    input = np.random.rand(BSIZE, SEQ_LEN, D_MODEL).astype(
        np.float32
    )  # shape (bs, seq_len, head_dim)

    rng = jax.random.PRNGKey(0)

    mask = np.ones((BSIZE, SEQ_LEN, SEQ_LEN)).astype(np.float32)
    # tril
    mask = np.tril(mask)
    params = fi_fn.init(rng, input, input, input, mask)

    # tie the weights
    params["params"]["wq"] = jnp.array(hf_fn.q_proj.weight.T.detach().numpy())
    params["params"]["wk"] = jnp.array(hf_fn.k_proj.weight.T.detach().numpy())
    params["params"]["wv"] = jnp.array(hf_fn.v_proj.weight.T.detach().numpy())
    params["params"]["wo"] = jnp.array(hf_fn.o_proj.weight.T.detach().numpy())

    position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BSIZE, SEQ_LEN).long()
    hf_outs = (
        hf_fn(
            torch.tensor(input),
            torch.tensor(mask[:, None, ...]),
            position_ids=position_ids,
        )[0]
        .detach()
        .numpy()
    )

    custom_outs = fi_fn.apply(params, input, input, input, mask).astype(np.float32)

    check_all_close_or_print(hf_outs, custom_outs, atol=2e-1, name="Attention forward")
