import torch
import torch.nn.functional as F
from core import ConformerParams
from core.models.attend import Attention
from core.models.positional_embeddings import (PositionalEmbeddingParams,
                                               RelativePositionBias)
from core.models.utils import calc_same_padding, default, exists
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding, stride=1):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, stride=stride, groups=chan_in
        )

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        bias = -torch.abs(
            rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1")
        )
        return bias


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlockBlock(nn.Module):
    def __init__(
        self,
        params: ConformerParams,
    ):
        super().__init__()
        self.dim = params.attention_params.dim
        self.ff1 = FeedForward(
            dim=self.dim,
            mult=params.ff_mult,
            dropout=params.ff_dropout,
        )

        self.attn = Attention(params.attention_params)
        if params.positional_embedding.name not in ["ABS", "SINE"]:

            self.pos_emb = RelativePositionBias(
                PositionalEmbeddingParams(
                    dim=self.dim, heads=params.attention_params.heads
                )
            )
        else:
            self.pos_emb = None

        self.conv = ConformerConvModule(
            dim=self.dim,
            causal=params.conv_causal,
            expansion_factor=params.conv_expansion_factor,
            kernel_size=params.conv_kernel_size,
            dropout=params.conv_dropout,
        )
        self.ff2 = FeedForward(
            dim=self.dim,
            mult=params.ff_mult,
            dropout=params.ff_dropout,
        )

        self.attn = PreNorm(self.dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(self.dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(self.dim, self.ff2))

        self.post_norm = nn.LayerNorm(self.dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask, rel_pos=self.pos_emb) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


# Conformer


class ConformerBlock(nn.Module):
    def __init__(self, params: ConformerParams, depth: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([])
        depth = default(params.depth, depth)

        for _ in range(depth):
            self.layers.append(ConformerBlockBlock(params))

    def forward(self, x, mask=None):
        for block in self.layers:
            x = block(x, mask)

        return x
