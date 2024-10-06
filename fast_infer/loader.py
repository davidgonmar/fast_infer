from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, AutoConfig
from fast_infer.models.llama import LlamaConfig, LlamaModel
from fast_infer.ops.attention import AttentionCache, AttentionConfig
from fast_infer.generic import Activation
import jax.numpy as jnp
import jax
import torch


def dot_dict_to_nested_keys(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        parts = key.split(".")
        current_dict = new_dict
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[parts[-1]] = value
    return new_dict


def nested_dict_to_dot_keys(nested_dict, parent_key="", sep="."):
    items = []
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(nested_dict_to_dot_keys(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def llama_key_converter(hf_dot_string: str) -> str:
    # Replace model.layers.<index> with LlamaDecoderLayer_<index>
    res = hf_dot_string.replace("model.layers.", "LlamaDecoderLayer_")

    # Replace attention and MLP keys
    res = res.replace("self_attn.q_proj.weight", "Attention_0.wq")
    res = res.replace("self_attn.k_proj.weight", "Attention_0.wk")
    res = res.replace("self_attn.v_proj.weight", "Attention_0.wv")
    res = res.replace("self_attn.o_proj.weight", "Attention_0.wo")
    res = res.replace("mlp.gate_proj.weight", "MLP_0.gate_proj")
    res = res.replace("mlp.up_proj.weight", "MLP_0.up_proj")
    res = res.replace(
        "mlp.down_proj.weight", "MLP_0.down_proj"
    )  # Assuming this was meant to be MLP_0.w2 instead of w1
    res = res.replace("input_layernorm.weight", "RMSNorm_0.scale")
    res = res.replace("input_layernorm.bias", "RMSNorm_0.bias")
    res = res.replace("post_attention_layernorm.weight", "RMSNorm_1.scale")
    res = res.replace("post_attention_layernorm.bias", "RMSNorm_1.bias")

    # embeds
    res = res.replace("model.embed_tokens.weight", "Embed_0.embedding")

    # norm and lm head
    res = res.replace("model.norm.weight", "RMSNorm_0.scale")
    res = res.replace("lm_head.weight", "lm_head")
    return res


def _torch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.detach().cpu().numpy())


def _torch_to_jax_dtype(dtype: torch.dtype) -> jnp.dtype:
    if dtype == torch.float32:
        return jnp.float32
    elif dtype == torch.float64:
        return jnp.float64
    elif dtype == torch.int32:
        return jnp.int32
    elif dtype == torch.int64:
        return jnp.int64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _substr(key, li):
    return any(map(lambda x: x in key, li))


def assign_dict(src, dst, strict=True):
    # make sure everything is assigned
    assigned_keys = set()
    for k, v in src.items():
        if k in assigned_keys:
            continue
        if isinstance(v, dict):
            assign_dict(v, dst[k])
        else:
            # dst should be jax array
            assert isinstance(dst[k], jnp.ndarray)
            # srcshould be torch tensor
            assert isinstance(v, torch.Tensor)
            # shapes, dtypes, must match
            if k == "lm_head" or _substr(
                k, ["gate_proj", "up_proj", "down_proj", "wq", "wk", "wv", "wo"]
            ):
                v = v.T  # Transpose the tensor
            assert (
                dst[k].shape == v.shape
            ), f"Shape mismatch: {dst[k].shape} != {v.shape} for key {k}"
            assert dst[k].dtype == _torch_to_jax_dtype(
                v.dtype
            ), f"Dtype mismatch: {dst[k].dtype} != {v.dtype} for key {k}"
            dst[k] = _torch_to_jax(v)
            assigned_keys.add(k)
    if strict:
        assert assigned_keys == set(src.keys()), "Keys mismatch!"
    return dst


def llama(model_name):
    # returns state dict and tokenizer
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    torch_state_dict = model.state_dict()
    hf_cfg = AutoConfig.from_pretrained(model_name)
    print(hf_cfg)
    cfg = LlamaConfig(
        n_q_heads=hf_cfg.num_attention_heads,
        n_kv_heads=hf_cfg.num_key_value_heads,
        d_model=hf_cfg.hidden_size,
        d_k=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        d_v=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
        activation=Activation.from_str(hf_cfg.hidden_act),
        hidden_dim=hf_cfg.intermediate_size,
        output_dim=hf_cfg.hidden_size,
        n_layers=hf_cfg.num_hidden_layers,
        vocab_size=hf_cfg.vocab_size,
        pre_attention_layernorm=True,
        post_attention_layernorm=True,
    )

    imodel = LlamaModel(config=cfg)

    # shape of x is (batch_size, seq_len)
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 128)).astype(jnp.int32)
    attn_mask = jnp.ones((8, 128, 128))
    params = imodel.init(jax.random.PRNGKey(0), x, attn_mask, None, 0)

    params_dict = params["params"]
    params_dict = nested_dict_to_dot_keys(params_dict)

    torch_dict_updated = {
        llama_key_converter(k): v for k, v in torch_state_dict.items()
    }
    assigned_dict = assign_dict(torch_dict_updated, params_dict)

    state_dict = dot_dict_to_nested_keys(assigned_dict)

    return imodel, tokenizer, {"params": state_dict}, cfg


class Sampler:
    def __init__(self, model, params, tok, cfg, use_cache=True):
        self.model = model
        self.params = params
        self.tok = tok
        self.cache = (
            AttentionCache(AttentionConfig(**cfg.to_dict())) if use_cache else None
        )

    def sample(self, prompt, max_len=100):
        rand = jax.random.PRNGKey(0)
        inptoks = self.tok(prompt)
        inp = jnp.array(inptoks["input_ids"]).reshape(1, -1)
        acc = inp
        end_token = self.tok.eos_token_id
        causal_mask = jnp.ones((1, max_len, max_len))
        causal_mask = jnp.triu(causal_mask) * -1e9
        curr_seq_pos = 0
        for i in range(max_len):
            print(curr_seq_pos)
            res = self.model.apply(
                self.params,
                acc if self.cache is None else inp,
                causal_mask,
                self.cache,
                curr_seq_pos if self.cache is not None else 0,
            )
            # sample
            rand, use = jax.random.split(rand)
            next_token = jax.random.categorical(use, res[:, -1, :].squeeze())
            acc = jnp.concatenate([acc, next_token.reshape(1, 1)], axis=1)
            inp = next_token.reshape(1, 1)
            print(self.tok.decode(acc[0]))
            if next_token == end_token:
                print("[INFO] Found end token. Stopping.")
                break
            curr_seq_pos = acc.shape[1] - 1
        return self.tok.decode(acc[0])


if __name__ == "__main__":
    model, tok, params, cfg = llama("TinyLlama/TinyLlama_v1.1")
    sampler = Sampler(model, params, tok, cfg, use_cache=True)
    print(sampler.sample("The quick brown fox jumps over the lazy dog."))
