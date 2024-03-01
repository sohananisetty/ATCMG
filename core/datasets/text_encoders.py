import logging
import typing as tp
import warnings
from typing import List, Union

import clip
import torch
import torch.nn as nn
from core.models.utils import TorchAutocast
from transformers import (AutoTokenizer, BertConfig, BertForMaskedLM,
                          BertModel, T5Config, T5EncoderModel, T5Tokenizer)

ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class T5Conditioner(nn.Module):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """

    MODELS = [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
        "google/flan-t5-small",
        "google/flan-t5-base",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(
        self,
        name: str,
        device: str,
        autocast_dtype: tp.Optional[str] = "float32",
        word_dropout: float = 0.0,
    ):
        assert (
            name in self.MODELS
        ), f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__()
        self.dim = self.MODELS_DIMS[name]
        self.device = device
        self.name = name
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == "cpu":
            self.autocast = TorchAutocast(enabled=False)
            # if self.device != "cpu":
            #     logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            # logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device, dtype=dtype
            )
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name, cache_dir="./")
                self.t5 = (
                    T5EncoderModel.from_pretrained(name, cache_dir="./")
                    .eval()
                    .to(device)
                )
            finally:
                logging.disable(previous_level)

        self.t5 = self.t5.eval()
        self.freeze()

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string'
        if x is not None and isinstance(x, str):
            x = [x]
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(
            self.device
        )
        inputs["attention_mask"][
            empty_idx, :
        ] = 0  # zero-out index where the input is non-existant
        return inputs

    def freeze(self):
        for p in self.t5.parameters():
            p.requires_grad = False

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs["attention_mask"]
        with torch.no_grad(), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = embeds * mask.unsqueeze(-1)
        return embeds, mask

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]

        with torch.no_grad(), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state

        embeds = embeds * mask.unsqueeze(-1)

        encoding = torch.mean(embeds, -2)
        mask = mask[:, 0:1]

        return encoding, mask


class ClipConditioner(nn.Module):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "ViT-B/32",
    ):
        super().__init__()
        self.device = device
        self.encoder, self.preprocess = clip.load(name)
        clip.model.convert_weights(self.encoder)

        self.encoder = self.encoder.eval()
        self.config = {"hidden_size": 512}
        self.freeze()

    # @property
    # def device(self):
    #     return next(self.encoder.parameters()).device

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        if x is not None and isinstance(x, str):
            x = [x]
        entries = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = {}

        inputs["input_ids"] = clip.tokenize(entries, truncate=True).to(self.device)
        inputs["attention_mask"] = torch.ones(len(inputs)).to(self.device)  ## B

        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        return self.get_text_embedding(inputs)

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]

        with torch.no_grad():
            embeds = self.encoder.encode_text(inputs["input_ids"]).float()

        embeds = embeds * mask.unsqueeze(-1)

        return embeds, mask

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


class BERTConditioner(nn.Module):
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name: str = "bert-base-uncased",
    ):
        super().__init__()
        self.device = device

        self.config = BertConfig.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.encoder = BertModel.from_pretrained(name)

        self.freeze()

        self.encoder = self.encoder.eval()

    # @property
    # def device(self):
    #     return next(self.encoder.parameters()).device

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def tokenize(self, x: Union[str, List[str]]):

        if x is not None and isinstance(x, str):
            x = [x]
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.tokenizer(
            entries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        inputs["attention_mask"][empty_idx, :] = 0

        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        return self.get_text_embedding(inputs)

    def get_text_embedding(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:

        mask = inputs["attention_mask"]
        mask = mask[:, 0:1]

        with torch.no_grad(), self.autocast:
            embeds = self.encoding(**inputs).last_hidden_state

        encoding = embeds[:, 0] * mask

        return encoding, mask
