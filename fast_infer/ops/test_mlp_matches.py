import jax
import jax.numpy as jnp
import torch
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaConfig
import numpy as np
from fast_infer.ops.mlp import MLP, MLPConfig, Activation


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
    D_MODEL = 64
    FFN_INTERMED_DIM = 128
    config = MLPConfig(
        d_model=D_MODEL, hidden_dim=FFN_INTERMED_DIM, activation=Activation.SILU
    )
    fi_fn = MLP(config)
    hf_fn = LlamaMLP(
        LlamaConfig(hidden_size=D_MODEL, intermediate_size=FFN_INTERMED_DIM)
    )

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

    rng = jax.random.PRNGKey(0)

    params = fi_fn.init(rng, input)

    # match torch params
    params["params"]["up_proj"] = jnp.array(hf_fn.up_proj.weight.T.detach().numpy())
    params["params"]["gate_proj"] = jnp.array(hf_fn.gate_proj.weight.T.detach().numpy())
    params["params"]["down_proj"] = jnp.array(hf_fn.down_proj.weight.T.detach().numpy())

    hf_outs = hf_fn(torch.tensor(input))

    custom_outs = fi_fn.apply(params, input)

    check_all_close_or_print(hf_outs, custom_outs, atol=2e-1, name="MLP forward")
