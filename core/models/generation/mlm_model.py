import math
import typing as tp
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from core import MotionTokenizerParams
from core.datasets.conditioner import ClassifierFreeGuidanceDropout, ConditionFuser
from core.models.attend2 import CustomMHA
from core.models.positional_embeddings import (
    ScaledSinusoidalEmbedding,
    ContinuousPositionBias,
    AlibiPositionalBias,
)
from core.models.utils import FeedForward, LayerNorm, default, exists, get_obj_from_str
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from einops.layers.torch import Rearrange


dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = (
    flash_attn_func
) = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    pass
ConditionType = Tuple[torch.Tensor, torch.Tensor]
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


@dataclass
class MuseOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    embed: torch.Tensor = None
    ce_per_codebook: List[torch.Tensor] = None
    align_loss: torch.Tensor = None


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


# classes
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim**-0.5))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x / norm.clamp(min=self.eps) * self.g


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g


def FeedForward(
    dim,
    mult=4,
    dropout=0.1,
):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        fused_if_available=True,
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(
                self.fused_mlp_func(
                    x=x,
                    weight1=self.fc1.weight,
                    weight2=self.fc2.weight,
                    bias1=self.fc1.bias,
                    bias2=self.fc2.bias,
                    activation="gelu_approx",
                    save_pre_act=self.training,
                    return_residual=False,
                    checkpoint_lvl=0,
                    heuristic=0,
                    process_group=None,
                )
            )
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f"fused_mlp_func={self.fused_mlp_func is not None}"


class FiLM(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2, bias=False)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim=-1)
        return x * gamma + beta


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`)."""

    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


class TransformerBlockMuseCustom(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
        film_skip: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.causal = causal

        for d in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=causal,
                            flash=flash,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=True,
                            flash=flash,
                            causal=False,
                        ),
                        (
                            FiLM(dim, dim)
                            if (
                                film_skip != 0 and d % film_skip == 0 and d != depth - 1
                            )
                            else nn.Identity()
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = LayerNorm(dim)
        self.norm_cross = LayerNorm(dim)
        self.norm_film = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)
        self.norm_film_context = LayerNorm(dim)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        context_mask=None,
        film_context=None,
        rel_pos=None,
    ):
        for attn, cross_attn, film, ff1 in self.layers:

            x = self.norm_sa(x)
            x = (
                attn(
                    q=x,
                    k=x,
                    v=x,
                    key_padding_mask=mask,
                    rel_pos=rel_pos if self.causal else None,
                )
                + x
            )

            x = (
                cross_attn(
                    q=self.norm_cross(x),
                    k=context,
                    v=context,
                    key_padding_mask=context_mask,
                )
                + x
            )

            if not isinstance(film, nn.Identity):

                x = (
                    film(
                        self.norm_film(x),
                        cond=self.norm_film_context(film_context),
                    )
                    + x
                )

            x = self.norm_out(ff1(x) + x)

        return x


class TransformerBlockMuseSpatialCustom(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
        causal: bool = False,
        film_skip: int = 0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.causal = causal

        for d in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=causal,
                            flash=flash,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=causal,
                            flash=flash,
                        ),
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=True,
                            flash=flash,
                            causal=False,
                        ),
                        (
                            FiLM(dim, dim)
                            if (
                                film_skip != 0 and d % film_skip == 0 and d != depth - 1
                            )
                            else nn.Identity()
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_spatial_attn = LayerNorm(dim)
        self.norm_temporal_attn = LayerNorm(dim)
        self.norm_cross = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)
        self.norm_film = LayerNorm(dim)
        self.norm_film_context = LayerNorm(dim)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        context_mask=None,
        film_context=None,
        rel_pos=None,
    ):

        b, n, c, d = x.shape
        x = rearrange(x, "b n c d -> (b c) n d", b=b, n=n, c=c, d=d)

        mask = None if mask is None else repeat(mask, "b n -> (b c) n", c=c)
        film_context = (
            None
            if film_context is None
            else repeat(film_context, "b n d -> (b c) n d", c=c)
        )
        context = (
            None if context is None else repeat(context, "b n d -> (b c) n d", c=c)
        )
        context_mask = (
            None
            if context_mask is None
            else repeat(context_mask, "b n -> (b c) n", c=c)
        )

        for sp_attn, tp_attn, cross_attn, film, ff1 in self.layers:

            if not isinstance(sp_attn, nn.Identity):
                x = rearrange(x, "(b c) n d -> (b n) c d", b=b, n=n, c=c, d=d)

                x = self.norm_spatial_attn(x)
                x = (
                    sp_attn(
                        q=x,
                        k=x,
                        v=x,
                        key_padding_mask=mask,
                        rel_pos=rel_pos if self.causal else None,
                    )
                    + x
                )
                x = rearrange(x, "(b n) c d -> (b c) n d", b=b, n=n, c=c, d=d)

            x = self.norm_temporal_attn(x)
            x = (
                tp_attn(
                    q=x,
                    k=x,
                    v=x,
                    key_padding_mask=mask,
                    rel_pos=rel_pos if self.causal else None,
                )
                + x
            )

            x = (
                cross_attn(
                    q=self.norm_cross(x),
                    k=(context),
                    v=(context),
                    key_padding_mask=context_mask,
                )
                + x
            )

            if not isinstance(film, nn.Identity) and film_context is not None:

                x = (
                    film(
                        self.norm_film(x),
                        cond=self.norm_film_context(film_context),
                    )
                    + x
                )

            x = self.norm_out(ff1(x) + x)

        x = rearrange(x, "(b c) n d -> b c n d", b=b, n=n, c=c, d=d)

        return x


class TransformerBlockMuse(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            bias=False,
                            batch_first=True,
                        ),
                        nn.MultiheadAttention(
                            embed_dim=dim,
                            num_heads=heads,
                            dropout=attn_dropout,
                            bias=False,
                            batch_first=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.norm_cross = nn.LayerNorm(dim, eps=1e-5, bias=False)
        self.norm_out = nn.LayerNorm(dim, eps=1e-5, bias=False)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:

            x = self.norm_sa(x)

            x = (
                attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=mask,
                    need_weights=False,
                )[0]
                + x
            )

            x = (
                cross_attn(
                    query=self.norm_cross(x),
                    key=context,
                    value=context,
                    key_padding_mask=context_mask,
                    need_weights=False,
                )[0]
                + x
            )

            x = self.norm_out(ff1(x) + x)

        return x


class MLMModel(nn.Module):
    def __init__(
        self,
        fuser: ConditionFuser,
        num_tokens,
        dim,
        heads,
        n_q=3,
        rel_pos=False,
        var_len=True,
        custom=True,
        spatial=False,
        film=False,
        film_skip=1,
        flatten=False,
        quality_emb=False,
        emb_lr: tp.Optional[float] = None,
        bias_proj=False,
        emb_dropout=0.0,
        cond_dropout=0.0,
        post_emb_norm: bool = False,
        audio_input_dim: int = 128,
        text_input_dim: int = 768,
        proj_input=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.cond_dropout = cond_dropout
        self.var_len = var_len
        self.quality_emb = quality_emb
        self.spatial = spatial
        self.flatten = flatten
        self.film_skip = film_skip
        self.film = film_skip > 0

        ## if flatten num_tokens = sum of all num_tokens

        self.set_token_ids(num_tokens, False)

        self.token_emb = nn.ModuleList(
            [ScaledEmbedding(self.vocab_size, self.dim, lr=emb_lr) for _ in range(n_q)]
        )

        if quality_emb:

            self.qual_emb = nn.Embedding(2, self.dim)

        self.pos_emb = ScaledSinusoidalEmbedding(dim=self.dim)
        self.rel_pos_bias = None
        if rel_pos:

            self.rel_pos_bias = AlibiPositionalBias(heads=heads // 2, total_heads=heads)

        if custom:
            if self.spatial:
                self.transformer_blocks = TransformerBlockMuseSpatialCustom(
                    dim=dim, heads=heads, film_skip=film_skip, **kwargs
                )
            else:

                self.transformer_blocks = TransformerBlockMuseCustom(
                    dim=dim, heads=heads, film_skip=film_skip, **kwargs
                )
        else:
            self.transformer_blocks = TransformerBlockMuse(
                dim=dim, heads=heads, **kwargs
            )
        self.norm = LayerNorm(dim)
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()

        self.condition_fuser = fuser
        self.translation_present = (
            self.condition_fuser.cond2fuse.get("translation", None) is not None
        )

        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=self.cond_dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_input = (
            nn.Linear(n_q * self.dim, self.dim, bias=False)
            if proj_input
            else nn.Identity()
        )

        self.project_audio = (
            nn.Linear(self.audio_input_dim, self.dim, bias=False)
            if self.audio_input_dim != self.dim
            else nn.Identity()
        )
        self.downsample_audio = nn.Sequential(
            Rearrange("b n d -> b d n"),
            nn.Conv1d(
                self.audio_input_dim,
                self.audio_input_dim,
                3,
                2,
                1,
                dilation=1,
                bias=False,
            ),
            nn.Conv1d(self.audio_input_dim, self.audio_input_dim, 3, 2, 1, bias=False),
            Rearrange("b d n -> b n d"),
        )  ## downsample factor 4 hard coded. assert audio sampling rate is equal to fps

        self.project_text = (
            nn.Linear(self.text_input_dim, self.dim, bias=False)
            if self.text_input_dim != self.dim
            else nn.Identity()
        )

        self.linears = (
            nn.Linear(dim, self.num_tokens, bias=bias_proj)
            if self.spatial
            else nn.ModuleList(
                [nn.Linear(dim, self.num_tokens, bias=bias_proj) for _ in range(n_q)]
            )
        )
        self.out_norm = LayerNorm(dim)

    def set_token_ids(self, num_tokens, add_special_token=False):
        motion_tokenizer_params = MotionTokenizerParams(num_tokens, add_special_token)
        self.num_tokens = motion_tokenizer_params.num_tokens
        self.mask_token_id = motion_tokenizer_params.mask_token_id
        self.pad_token_id = motion_tokenizer_params.pad_token_id
        self.special_token_id = motion_tokenizer_params.special_token_id
        self.vocab_size = motion_tokenizer_params.vocab_size

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def _compute_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
        ignore_index=-100,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """Compute cross entropy between multi-codebook targets and model's logits.
        The cross entropy is computed per codebook to provide codebook-level cross entropy.
        Valid timesteps for each of the codebook are pulled from the mask, where invalid
        timesteps are set to 0.

        Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
        Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
        """
        B, K, T = targets.shape
        assert logits.shape[:-1] == targets.shape
        # assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = (
                logits[:, k, ...].contiguous().view(-1, logits.size(-1))
            )  # [B x T, card]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            if mask is not None:
                mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
                targets_k = targets_k[mask_k]
                logits_k = logits_k[mask_k]
            q_ce = F.cross_entropy(logits_k, targets_k, ignore_index=ignore_index)
            ce += q_ce
            ce_per_codebook.append(q_ce)
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def _prepare_inputs(self, input, conditions):
        new_conditions = {}

        if conditions.get("audio", None) is not None:
            audio_embed = self.downsample_audio(conditions["audio"][0])
            audio_embed = self.project_audio(audio_embed)
            new_conditions["audio"] = (audio_embed, conditions["audio"][1])

        if conditions.get("text", None) is not None:
            text_embed = self.project_text(conditions["text"][0])
            new_conditions["text"] = (text_embed, conditions["text"][1])

        rem_keys = [x for x in list(conditions.keys()) if x not in ["audio", "text"]]

        for k in rem_keys:
            new_conditions[k] = conditions[k]

        inputs_, cross_inputs = self.condition_fuser(input, new_conditions)

        if self.film:
            film_cond_mask = (
                F.interpolate(
                    conditions["audio"][1].unsqueeze(1).to(torch.float),
                    size=input[0].shape[1],
                )
                .squeeze(1)
                .to(torch.bool)
            )

            film_cond = audio_embed * film_cond_mask[..., None]

            return inputs_, cross_inputs, film_cond

        return inputs_, cross_inputs

    def forward_with_cond_scale(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        if cond_scale == 1:
            out = self.forward(
                inputs,
                conditions,
                cond_drop_prob=0.0,
                **kwargs,
            )
            if return_embed:
                return out.logits, pos_out.embed

            return out.logits

        pos_out = self.forward(inputs, conditions, cond_drop_prob=0.0, **kwargs)

        null_out = self.forward(inputs, conditions, cond_drop_prob=1.0, **kwargs)

        scaled_logits = (
            null_out.logits + (pos_out.logits - null_out.logits) * cond_scale
        )

        if return_embed:
            return scaled_logits, pos_out.embed

        return scaled_logits

    def forward_with_neg_prompt(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        neg_conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        return_embed=False,
        **kwargs,
    ):
        neg_out = self.forward(
            inputs,
            neg_conditions,
            cond_drop_prob=0.0,
            **kwargs,
        )
        pos_out = self.forward(
            inputs,
            conditions,
            cond_drop_prob=0.0,
            **kwargs,
        )

        scaled_logits = neg_out.logits + (pos_out.logits - neg_out.logits) * cond_scale

        if return_embed:
            return scaled_logits, pos_out.embed

        return scaled_logits

    def forward(
        self,
        inputs: ConditionTensors,
        conditions: ConditionTensors,
        labels: Optional[torch.IntTensor] = None,
        cond_drop_prob: float = None,
        ignore_index: int = -100,
        quality_list=None,
        return_embed=False,
    ):

        sequence = inputs[0]
        sequence_mask = inputs[1]
        B, K, S = sequence.shape

        if self.quality_emb is True and quality_list is None:
            quality_list = torch.ones(B, dtype=torch.long, device=sequence.device)

        assert (
            K == self.num_codebooks
        ), "Sequence shape must match the specified number of codebooks"

        if self.spatial:
            input_ = torch.stack(
                [self.token_emb[k](sequence[:, k]) for k in range(K)], -1
            )

            input_ = rearrange(input_, "b n d c -> b n c d")

        else:

            if isinstance(self.project_input, nn.Identity):

                input_ = sum([self.token_emb[k](sequence[:, k]) for k in range(K)])
            else:

                input_ = torch.stack(
                    [self.token_emb[k](sequence[:, k]) for k in range(K)], -1
                )
                input_ = input_.reshape(B, S, -1)

                input_ = self.project_input(input_)

        # classifier free guidance
        cond_drop_prob = default(cond_drop_prob, self.cond_dropout)
        if cond_drop_prob > 0.0:
            conditions = self.cfg_dropout(conditions, cond_drop_prob)
        # fuse conditions

        processed_inputs = self._prepare_inputs((input_, sequence_mask), conditions)

        if self.film:
            input_, cross_attention_input, film_input_ = processed_inputs
        else:
            input_, cross_attention_input = processed_inputs

        x_ = input_[0]
        x_padding_mask = input_[1]
        context = cross_attention_input[0]
        context_padding_mask = cross_attention_input[1]

        if self.spatial:
            # x_ = rearrange(x_, "b n (c d) -> b n c d", c=K)
            pos_emb_spatial = self.pos_emb(pos=torch.arange(K, device=x_.device))
            pos_emb_tmp = self.pos_emb(x_)
            x_ = x_ + pos_emb_spatial[None, None, :, :]
            x_ = x_ + pos_emb_tmp[None, :, None, :]
        else:
            x_ = x_ + self.pos_emb(x_)

        x_ = self.post_emb_norm(x_)
        x_ = self.emb_dropout(x_)

        if quality_list is not None and self.quality_emb == True:
            x_ = torch.cat([self.qual_emb(quality_list).unsqueeze(1), x_], dim=-2)
            x_padding_mask = F.pad(x_padding_mask, (1, 0), value=True)

        if self.film:
            embed = self.transformer_blocks(
                x=x_,
                mask=x_padding_mask if self.var_len else None,
                context=context,
                context_mask=context_padding_mask,
                film_context=film_input_,
                rel_pos=self.rel_pos_bias,
            )

        else:

            embed = self.transformer_blocks(
                x=x_,
                mask=x_padding_mask if self.var_len else None,
                context=context,
                context_mask=context_padding_mask,
                rel_pos=self.rel_pos_bias,
            )
        if self.out_norm:
            embed = self.out_norm(embed)

        if return_embed:
            return MuseOutput(embed=embed)

        if self.spatial:
            logits = self.linears(embed)

        else:

            logits = torch.stack([self.linears[k](embed) for k in range(K)], dim=1)

        if (
            len(self.condition_fuser.fuse2cond.get("prepend", [])) > 0
            or self.quality_emb
        ):
            logits = logits[:, :, -S:]

        if not exists(labels):
            return MuseOutput(logits=logits, embed=embed)  ## B K N num_tokens

        if self.num_codebooks == 1:

            loss = F.cross_entropy(
                rearrange(logits.squeeze(1), "b n c -> b c n"),
                labels.squeeze(1),
                ignore_index=ignore_index,
            )
            return MuseOutput(loss=loss, logits=logits, embed=embed)

        else:
            loss, ce_per_codebook = self._compute_cross_entropy(
                logits,
                labels,
                ignore_index=ignore_index,
            )
            return MuseOutput(
                loss=loss, logits=logits, embed=embed, ce_per_codebook=ce_per_codebook
            )
