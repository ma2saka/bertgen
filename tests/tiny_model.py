"""Builds a tiny random ModernBERT and an offline character-level tokenizer for tests."""

import string
from pathlib import Path

from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(SPECIALS + list(string.printable.strip("\x0b\x0c")))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def save_tiny_model(directory: Path) -> str:
    """Write a randomly initialised model and tokenizer to `directory`; returns its path."""
    tokenizer = build_tokenizer()
    pad, cls, sep = (tokenizer.convert_tokens_to_ids(t) for t in ("[PAD]", "[CLS]", "[SEP]"))
    assert isinstance(pad, int)
    assert isinstance(cls, int)
    assert isinstance(sep, int)
    config = ModernBertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        local_attention=8,
        pad_token_id=pad,
        cls_token_id=cls,
        sep_token_id=sep,
        bos_token_id=cls,
        eos_token_id=sep,
    )
    ModernBertModel(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)
