from typing import List, Union

import clip
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    T5Config,
    T5EncoderModel,
    T5Tokenizer,
)

DEFAULT_T5_NAME = "google/t5-v1_1-base"
# "google/t5-v1_1-base"
# "google/flan-t5-xl"

from dataclasses import dataclass


@dataclass
class TextEncoderParams:
    padding: str = "longest"
    target: str = "google/t5-v1_1-base"
    max_length: int = 128


class T5(nn.Module):
    def __init__(self, params: TextEncoderParams = TextEncoderParams) -> None:
        super().__init__()
        self.max_length = params.max_length
        self.padding = params.padding
        self.config = T5Config.from_pretrained(params.target)
        self.dim = self.config.d_model
        self.tokenizer = T5Tokenizer.from_pretrained(params.target)
        self.encoder = T5EncoderModel.from_pretrained(params.target)

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def load(self, path):
        pkg = torch.load(str(path), map_location="cuda")
        self.encoder.load_state_dict(pkg["model"])

    def tokenize(self, texts: Union[str, List[str]]):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        return input_ids, attn_mask

    def get_text_embedding(
        self,
        texts: Union[str, List[str]],
        mask_id: float = 0.0,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids, attn_mask = self.tokenize(texts)

        with torch.no_grad():
            encoded_text = self.encoder(
                input_ids=input_ids, attention_mask=attn_mask
            ).last_hidden_state

        attn_mask = attn_mask.bool()
        encoded_text = encoded_text.masked_fill(~attn_mask[..., None], mask_id)

        return encoded_text[:, 0, :]


class Clip(nn.Module):
    def __init__(self, params: TextEncoderParams = None) -> None:
        super().__init__()
        self.encoder, self.preprocess = clip.load("ViT-B/32")
        clip.model.convert_weights(self.encoder)

        self.encoder = self.encoder.eval()
        self.config = {"hidden_size": 512}
        self.freeze()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def get_text_embedding(self, texts, mask_id=None):
        if isinstance(texts, str):
            texts = [texts]

        return self.encoder.encode_text(
            clip.tokenize(texts, truncate=True).to(self.device)
        ).float()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


class BERTBASE(nn.Module):
    def __init__(self, params: TextEncoderParams = TextEncoderParams) -> None:
        super().__init__()
        self.max_length = params.max_length
        self.padding = params.padding
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder = self.encoder.eval()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def load(self, path):
        pkg = torch.load(str(path), map_location="cpu")
        self.encoder.load_state_dict(pkg["model"])

    def tokenize(self, texts: Union[str, List[str]]):
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        return input_ids, attn_mask

    def get_text_embedding(
        self,
        texts: Union[str, List[str]],
        mask_id: float = 0.0,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids, attn_mask = self.tokenize(texts)

        with torch.no_grad():
            encoded_text = self.encoder(
                input_ids=input_ids, attention_mask=attn_mask, return_dict=True
            ).last_hidden_state

        attn_mask = attn_mask.bool()
        encoded_text = encoded_text.masked_fill(~attn_mask[..., None], mask_id)

        return encoded_text, attn_mask
