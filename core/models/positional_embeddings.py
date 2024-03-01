import math
from dataclasses import dataclass
from enum import Enum
from functools import partial

import torch
import torch.nn.functional as F
from core import PositionalEmbeddingParams
from core.models.utils import exists, pad_at_dim
from einops import rearrange, repeat
from torch import einsum, nn


class ShawRelativePositionalEmbedding(nn.Module):
    def __init__(self, params: PositionalEmbeddingParams):
        super().__init__()
        self.scale = params.dim**-0.5
        self.max_seq_len = params.max_seq_len
        self.rel_pos_emb = nn.Embedding(2 * params.max_seq_len + 1, params.dim)

    def forward(self, q, k):
        b, h, n, d = q.shape
        seq = torch.arange(n, device=q.device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-self.max_seq_len, self.max_seq_len) + self.max_seq_len
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        return pos_attn


class RelativePositionBias(nn.Module):
    def __init__(self, params: PositionalEmbeddingParams):
        super().__init__()
        self.scale = params.scale
        self.causal = params.causal
        self.num_buckets = params.num_buckets
        self.max_distance = params.max_distance
        self.relative_attention_bias = nn.Embedding(params.num_buckets, params.heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> h i j")
        return bias * self.scale


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, params: PositionalEmbeddingParams):
        super().__init__()
        self.scale = params.dim**-0.5
        self.max_seq_len = params.max_seq_len
        self.emb = nn.Embedding(params.max_seq_len, params.dim)

    def forward(self, x, pos=None):
        seq_len, device = x.shape[1], x.device
        assert (
            seq_len <= self.max_seq_len
        ), f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, params: PositionalEmbeddingParams):
        super().__init__()
        assert (params.dim % 2) == 0
        self.scale = nn.Parameter(torch.ones(1) * params.dim**-0.5)

        half_dim = params.dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = params.theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device=device)

        emb = einsum("i, j -> i j", pos, self.inv_freq.to(device))
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale.to(device)


class AlibiPositionalBias(nn.Module):
    def __init__(self, params: PositionalEmbeddingParams):
        super().__init__()
        self.heads = params.heads
        self.total_heads = params.total_heads

        slopes = torch.Tensor(self._get_slopes(params.heads))
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

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., :i, :j]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim=0)
        self.register_buffer("bias", bias, persistent=False)

        return self.bias


# class RotaryEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim,
#         use_xpos = False,
#         scale_base = 512,
#         interpolation_factor = 1.,
#         base = 10000,
#         base_rescale_factor = 1.
#     ):
#         super().__init__()
#         # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
#         # has some connection to NTK literature
#         # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
#         base *= base_rescale_factor ** (dim / (dim - 2))

#         inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
#         self.register_buffer('inv_freq', inv_freq)

#         assert interpolation_factor >= 1.
#         self.interpolation_factor = interpolation_factor

#         if not use_xpos:
#             self.register_buffer('scale', None)
#             return

#         scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

#         self.scale_base = scale_base
#         self.register_buffer('scale', scale)

#     def forward_from_seq_len(self, seq_len):
#         device = self.inv_freq.device

#         t = torch.arange(seq_len, device = device)
#         return self.forward(t)

#     @autocast(enabled = False)
#     def forward(self, t):
#         max_pos = t.max()+1

#         freqs = torch.einsum('i , j -> i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
#         freqs = torch.cat((freqs, freqs), dim = -1)

#         if not exists(self.scale):
#             return freqs, 1.

#         power = (t - (max_pos // 2)) / self.scale_base
#         scale = self.scale ** rearrange(power, 'n -> n 1')
#         scale = torch.cat((scale, scale), dim = -1)

#         return freqs, scale


# def rotate_half(x):
#     x = rearrange(x, '... (j d) -> ... j d', j = 2)
#     x1, x2 = x.unbind(dim = -2)
#     return torch.cat((-x2, x1), dim = -1)

# @autocast(enabled = False)
# def apply_rotary_pos_emb(t, freqs, scale = 1):
#     rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
#     freqs = freqs[-seq_len:, :]
#     scale = scale[-seq_len:, :] if isinstance(scale, torch.Tensor) else scale

#     if t.ndim == 4 and freqs.ndim == 3:
#         freqs = rearrange(freqs, 'b n d -> b 1 n d')

#     # partial rotary embeddings, Wang et al. GPT-J
#     t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
#     t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
#     return torch.cat((t, t_unrotated), dim = -1)


# class PositionalEmbeddingType(Enum):
#     REL = partial(RelativePositionBias)
#     SINE = partial(ScaledSinusoidalEmbedding)
#     ALIBI = partial(AlibiPositionalBias)
#     ABS = partial(AbsolutePositionalEmbedding)
#     SHAW = partial(ShawRelativePositionalEmbedding)
