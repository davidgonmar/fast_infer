import jax
import jax.numpy as jnp
import torch
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as HfLlamaDecoderLayer,
    LlamaConfig as HfLlamaConfig,
    LlamaForCausalLM as HfLlamaModel,
)
import numpy as np
from fast_infer.models.llama import (
    LlamaDecoderLayer as FastInferLlamaDecoderLayer,
    LlamaConfig as FastInferLlamaConfig,
    LlamaModel as FastInferLlamaModel,
)
from fast_infer.generic import Activation


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


def test_decoder_forward():
    D_MODEL = 8 * 64
    N_Q_HEADS = 32
    N_KV_HEADS = 8
    D_K = D_MODEL // N_Q_HEADS
    D_V = D_MODEL // N_Q_HEADS
    SEQ_LEN = 44
    BSIZE = 4
    config = FastInferLlamaConfig(
        n_q_heads=N_Q_HEADS,
        n_kv_heads=N_KV_HEADS,
        d_model=D_MODEL,
        d_k=D_K,
        d_v=D_V,
        activation=Activation.SILU,
        hidden_dim=128,
        vocab_size=128,
        n_layers=4,
    )
    fi_fn = FastInferLlamaDecoderLayer(config)
    hf_fn = HfLlamaDecoderLayer(
        HfLlamaConfig(
            num_attention_heads=N_Q_HEADS,
            num_key_value_heads=N_KV_HEADS,
            hidden_size=D_MODEL,
            intermediate_size=128,
            num_hidden_layers=4,
            vocab_size=128,
            activation="silu",
        ),
        layer_idx=0,
    )

    input = np.random.rand(BSIZE, SEQ_LEN, D_MODEL).astype(
        np.float32
    )  # shape (bs, seq_len, head_dim)

    rng = jax.random.PRNGKey(0)

    mask = np.ones((BSIZE, SEQ_LEN, SEQ_LEN)).astype(np.float32)
    # tril
    mask = np.triu(mask) * 4 + np.eye(SEQ_LEN) * 10
    params = fi_fn.init(rng, input, mask, None, 0)

    # tie the weights
    params["params"]["Attention_0"]["wq"] = jnp.array(
        hf_fn.self_attn.q_proj.weight.T.detach().numpy()
    )
    params["params"]["Attention_0"]["wk"] = jnp.array(
        hf_fn.self_attn.k_proj.weight.T.detach().numpy()
    )
    params["params"]["Attention_0"]["wv"] = jnp.array(
        hf_fn.self_attn.v_proj.weight.T.detach().numpy()
    )
    params["params"]["Attention_0"]["wo"] = jnp.array(
        hf_fn.self_attn.o_proj.weight.T.detach().numpy()
    )

    params["params"]["MLP_0"]["up_proj"] = jnp.array(
        hf_fn.mlp.up_proj.weight.T.detach().numpy()
    )
    params["params"]["MLP_0"]["gate_proj"] = jnp.array(
        hf_fn.mlp.gate_proj.weight.T.detach().numpy()
    )
    params["params"]["MLP_0"]["down_proj"] = jnp.array(
        hf_fn.mlp.down_proj.weight.T.detach().numpy()
    )

    params["params"]["RMSNorm_0"]["scale"] = jnp.array(
        hf_fn.input_layernorm.weight.detach().numpy()
    )
    params["params"]["RMSNorm_1"]["scale"] = jnp.array(
        hf_fn.post_attention_layernorm.weight.detach().numpy()
    )

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

    custom_outs = fi_fn.apply(params, input, mask, None, 0)

    check_all_close_or_print(hf_outs, custom_outs, atol=1e-3, name="Decoder forward")


def test_model_forward():
    D_MODEL = 8 * 64
    N_Q_HEADS = 32
    N_KV_HEADS = 8
    D_K = D_MODEL // N_Q_HEADS
    D_V = D_MODEL // N_Q_HEADS
    SEQ_LEN = 44
    BSIZE = 1
    config = FastInferLlamaConfig(
        n_q_heads=N_Q_HEADS,
        n_kv_heads=N_KV_HEADS,
        d_model=D_MODEL,
        d_k=D_K,
        d_v=D_V,
        activation=Activation.SILU,
        hidden_dim=128,
        vocab_size=128,
        n_layers=4,
    )

    fi_fn = FastInferLlamaModel(config)
    hf_fn = HfLlamaModel(
        HfLlamaConfig(
            num_attention_heads=N_Q_HEADS,
            num_key_value_heads=N_KV_HEADS,
            hidden_size=D_MODEL,
            intermediate_size=128,
            num_hidden_layers=4,
            vocab_size=128,
            activation="silu",
        )
    )

    # shape of x is (batch_size, seq_len)
    x = jax.random.randint(jax.random.PRNGKey(0), (BSIZE, SEQ_LEN), 0, 127).astype(
        jnp.int32
    )

    attn_mask = jnp.ones((BSIZE, SEQ_LEN, SEQ_LEN))
    attn_mask = -jnp.triu(attn_mask, k=1) * 0.44 + -jnp.eye(SEQ_LEN) * 0.3

    params = fi_fn.init(jax.random.PRNGKey(0), x, attn_mask, None, 0)

    torch_state_dict = hf_fn.state_dict()

    params["params"]["Embed_0"]["embedding"] = jnp.array(
        torch_state_dict["model.embed_tokens.weight"].detach().numpy()
    )
    params["params"]["RMSNorm_0"]["scale"] = jnp.array(
        torch_state_dict["model.norm.weight"].detach().numpy()
    )

    params["params"]["lm_head"] = jnp.array(
        torch_state_dict["lm_head.weight"].T.detach().numpy()
    )

    for i in range(4):
        params["params"][f"LlamaDecoderLayer_{i}"]["Attention_0"]["wq"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
            .T.detach()
            .numpy()
        )
        params["params"][f"LlamaDecoderLayer_{i}"]["Attention_0"]["wk"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
            .T.detach()
            .numpy()
        )
        params["params"][f"LlamaDecoderLayer_{i}"]["Attention_0"]["wv"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
            .T.detach()
            .numpy()
        )
        params["params"][f"LlamaDecoderLayer_{i}"]["Attention_0"]["wo"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
            .T.detach()
            .numpy()
        )

        params["params"][f"LlamaDecoderLayer_{i}"]["MLP_0"]["up_proj"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.mlp.up_proj.weight"].T.detach().numpy()
        )

        params["params"][f"LlamaDecoderLayer_{i}"]["MLP_0"]["gate_proj"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
            .T.detach()
            .numpy()
        )
        params["params"][f"LlamaDecoderLayer_{i}"]["MLP_0"]["down_proj"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.mlp.down_proj.weight"]
            .T.detach()
            .numpy()
        )
        params["params"][f"LlamaDecoderLayer_{i}"]["RMSNorm_0"]["scale"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.input_layernorm.weight"]
            .detach()
            .numpy()
        )
        params["params"][f"LlamaDecoderLayer_{i}"]["RMSNorm_1"]["scale"] = jnp.array(
            torch_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"]
            .detach()
            .numpy()
        )

    hf_outs = (
        hf_fn(
            torch.tensor(np.array(x)),
            attention_mask=torch.tensor(np.array(attn_mask)[:, None, ...]),
            position_ids=torch.arange(SEQ_LEN)
            .unsqueeze(0)
            .expand(BSIZE, SEQ_LEN)
            .long(),
        )[0]
        .detach()
        .numpy()
    )

    custom_outs, _ = fi_fn.apply(params, x, attn_mask, None, 0)

    check_all_close_or_print(hf_outs, custom_outs, atol=1e-2, name="Model forward")
