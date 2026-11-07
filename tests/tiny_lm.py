"""Build a tiny random Qwen2-style chat model + tokenizer on disk (no downloads)."""

from __future__ import annotations

import os

CHAT_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + "
    "'<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tiny_lm(path: str, corpus: list[str]) -> str:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    os.makedirs(path, exist_ok=True)
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus * 3, trainers.BpeTrainer(vocab_size=400, special_tokens=specials,
                                                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                             show_progress=False))
    # Like Qwen's tokenizer: no token_type_ids (decoder-only models reject them in generate()).
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"],
                                  model_input_names=["input_ids", "attention_mask"])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      bos_token_id=None)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    return path
