from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.conformer import ConformerBlock
from core.models.dataclasses import ConformerParams, TransformerParams
from core.models.transformer import TransformerBlock
from core.models.utils import exists, pad_to_multiple
from einops import rearrange


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, "b (n s) d -> b n (s d)", s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b n (s d) -> b (n s) d", s=self.shorten_factor)


class ConvDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.dim = dim
        if shorten_factor % 2 == 0:
            filter_t, pad_t = shorten_factor * 2, shorten_factor // 2
        else:
            filter_t, pad_t = 3, 1
        self.conv = nn.Conv1d(dim, dim, filter_t, shorten_factor, pad_t)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        need_transpose = x.shape[-1] == self.dim
        if need_transpose:
            x = x.rearrange(0, 2, 1).contiguous()
        x = self.conv(x)
        if need_transpose:
            x = x.rearrange(0, 2, 1).contiguous()
        return x


class ConvUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.dim = dim

        self.ups = nn.Upsample(scale_factor=shorten_factor, mode="nearest")
        self.conv = nn.Conv1d(dim, dim, 3, 1, 1)

    def forward(self, x):
        need_transpose = x.shape[-1] == self.dim
        if need_transpose:
            x = x.rearrange(0, 2, 1).contiguous()

        x = self.ups(x)
        x = self.conv(x)
        if need_transpose:
            x = x.rearrange(0, 2, 1).contiguous()
        return x


class DecoderTransfomer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        up_sampling_ratio=4,
        transfomer_params: TransformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.dim = transfomer_params.attention_params.dim

        for _ in range(int(np.log2(up_sampling_ratio))):
            block = nn.Sequential(
                TransformerBlock(
                    transfomer_params.attention_params, transfomer_params.depth
                ),
                ConvUpsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.blocks.append(nn.Linear(self.dim, input_dim))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, TransformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

        else:
            out = self.model(x)

        return out


class EncoderTransfomer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        down_sampling_ratio=4,
        transfomer_params: TransformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.updown = down_sampling_ratio
        self.dim = transfomer_params.attention_params.dim

        self.blocks.append(nn.Linear(input_dim, self.dim))

        for _ in range(int(np.log2(down_sampling_ratio))):
            block = nn.Sequential(
                TransformerBlock(
                    transfomer_params.attention_params, transfomer_params.depth
                ),
                ConvDownsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d
        x = pad_to_multiple(x, self.updown, dim=-2)

        if mask is not None:
            mask = pad_to_multiple(mask, self.updown, dim=-1, value=False)
            for l in self.blocks:
                if isinstance(l, TransformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x
            return out, mask

        else:
            out = self.model(x)

        return out


class DecoderConformer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        up_sampling_ratio=4,
        conformer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.dim = conformer_params.attention_params.dim

        for _ in range(int(np.log2(up_sampling_ratio))):
            block = nn.Sequential(
                ConformerBlock(conformer_params, conformer_params.depth),
                LinearUpsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.blocks.append(nn.Linear(self.dim, input_dim))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

        else:
            out = self.model(x)

        return out


class EncoderConformer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        down_sampling_ratio=4,
        conformer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.updown = down_sampling_ratio
        self.dim = conformer_params.attention_params.dim

        self.blocks.append(nn.Linear(input_dim, self.dim))

        for _ in range(int(np.log2(down_sampling_ratio))):
            block = nn.Sequential(
                ConformerBlock(conformer_params, conformer_params.depth),
                LinearDownsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d
        x = pad_to_multiple(x, self.updown, dim=-2)

        if mask is not None:
            mask = pad_to_multiple(mask, self.updown, dim=-1, value=False)
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x
            return out, mask

        else:
            out = self.model(x)

        return out
