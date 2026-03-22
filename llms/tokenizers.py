from typing import Any

import tiktoken
from transformers import LlamaTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # GPT-5 family (gpt-5, gpt-5-mini, gpt-5.1, gpt-5.2, etc.) use o200k_base
                if model_name.startswith("gpt-5"):
                    try:
                        self.tokenizer = tiktoken.get_encoding("o200k_base")
                    except Exception:
                        self.tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    # New/unmapped model names (e.g. gpt-4o, gpt-4o-mini) use cl100k_base
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
