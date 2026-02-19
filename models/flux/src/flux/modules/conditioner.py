from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from typing import Union, Dict, Any

class HFEmbedder(nn.Module):
    def __init__(self, version: Union[str, Dict[str, Any]], max_length: int, **hf_kwargs):
        super().__init__()

        # ---- NEW: allow dict {"tokenizer": ..., "model": ...} ----
        if isinstance(version, dict):
            tok_src = version.get("tokenizer", None)
            model_src = version.get("model", None)
            if tok_src is None or model_src is None:
                raise ValueError(f"HFEmbedder: dict version must contain 'tokenizer' and 'model', got keys={list(version.keys())}")
            vtag = str(model_src)
        else:
            tok_src = version
            model_src = version
            vtag = str(version)
        self.is_clip = version["type"] == "clip"

        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tok_src, max_length=max_length,local_files_only=True)
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(model_src, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tok_src, max_length=max_length,local_files_only=True)
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(model_src, **hf_kwargs)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
