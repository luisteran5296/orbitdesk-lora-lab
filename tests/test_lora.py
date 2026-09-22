"""A tiny random model tests adapter behavior without downloading weights."""

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM


def test_only_adapters_update_and_disabling_recovers_the_base():
    torch.manual_seed(42)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    ids = torch.tensor([[1, 4, 6, 8, 2]])
    base.eval()
    with torch.no_grad():
        expected = base(ids).logits.clone()
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    assert all(
        "lora_" in name for name, p in model.named_parameters() if p.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    model.train()
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        actual = model(ids).logits
    torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        adapted = model(ids).logits
    assert not torch.allclose(adapted, expected)
