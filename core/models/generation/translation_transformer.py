import math
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from core.datasets.conditioner import ConditionFuser
from core.models.attend import Attend, Attention
from core.models.dataclasses import (
    AttentionParams,
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
    TranslationTransformerParams,
)
from core.models.utils import FeedForward, LayerNorm, default, exists, get_obj_from_str
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN
from core.models.dataclasses import TranslationTransformerOutput

ConditionType = Tuple[torch.Tensor, torch.Tensor]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        self_attention_params: AttentionParams,
        cross_attention_params: AttentionParams = None,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.do_cross = cross_attention_params is not None

        assert cross_attention_params.dim == self_attention_params.dim, "need same dim"

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(self_attention_params),
                        (
                            Attention(cross_attention_params)
                            if self.do_cross
                            else nn.Identity()
                        ),
                        FeedForward(dim=self_attention_params.dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(self_attention_params.dim)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:
            x = attn(x, mask=mask, rel_pos=rel_pos) + x

            if self.do_cross:

                x = (
                    cross_attn(x, mask=mask, context=context, context_mask=context_mask)
                    + x
                )

            x = ff1(x) + x

        return self.norm(x)


class SeperateCrossTransformerBlock(nn.Module):
    def __init__(
        self,
        self_attention_params: AttentionParams,
        cross_attention_params_audio: AttentionParams,
        cross_attention_params_text: AttentionParams,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        assert (
            cross_attention_params_audio.dim == self_attention_params.dim
        ), "need same dim"

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(self_attention_params),
                        Attention(cross_attention_params_audio),  ## audio
                        Attention(cross_attention_params_text),  ## text
                        FeedForward(dim=self_attention_params.dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(self_attention_params.dim)

    def forward(
        self,
        x,
        mask=None,
        context: List = None,
        context_mask: List = None,
        rel_pos=None,
    ):
        for attn, cross_attn_audio, cross_attn_text, ff1 in self.layers:
            x = attn(x, mask=mask, rel_pos=rel_pos) + x

            x = (
                cross_attn_audio(
                    x,
                    mask=mask,
                    context=context["audio"],
                    context_mask=context_mask["audio"],
                )
                + x
            )

            x = (
                cross_attn_text(
                    x,
                    mask=mask,
                    context=context["text"],
                    context_mask=context_mask["text"],
                )
                + x
            )

            x = ff1(x) + x

        return self.norm(x)


class ClassifierFreeGuidanceDropout(nn.Module):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """

    def __init__(self, p: float, seed: int = 42):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.p = p

    def prob_mask_like(self, shape, prob, device=None):
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

    def forward(
        self, conditions: Dict[str, ConditionType], drop_prob: float = None
    ) -> Dict[str, ConditionType]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """

        drop_prob = drop_prob if drop_prob is not None else self.p

        conditions_ = deepcopy(conditions)

        for condition_modality, (embedding, mask) in conditions.items():
            b, n = mask.shape

            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)

            if drop_prob > 0.0 and (condition_modality != "audio" or not self.training):
                drop_mask = self.prob_mask_like((b, 1), 1.0 - drop_prob, mask.device)
                new_mask = mask & drop_mask
                new_embedding = embedding * new_mask.unsqueeze(-1)
                conditions_[condition_modality] = (new_embedding, new_mask)

        return conditions_

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class TranslationTransformer(nn.Module):
    def __init__(self, config: CN):
        super().__init__()

        self_attention_params = AttentionParams(
            dim=config.dim, causal=config.is_self_causal
        )
        cross_attention_params = AttentionParams(
            dim=config.dim,
            causal=config.is_cross_causal,
            add_null_kv=True,
        )
        # transformer_params = TranslationTransformerParams(
        #     self_attention_params=sap,
        #     cross_attention_params=cap,
        #     depth=config.depth,
        #     positional_embedding_params=PositionalEmbeddingParams(dim=config.dim),
        #     positional_embedding=PositionalEmbeddingType.SINE,
        #     fuse_method=config.fuse_method[0],
        #     cond_dropout=config.cond_dropout,
        #     audio_input_dim=config.audio_input_dim,
        #     text_input_dim=config.text_input_dim,
        #     dim_out=config.dim_out,
        #     loss_fnc=config.loss_fnc,
        # )

        self.dim = config.dim
        self.dim_out = config.dim_out
        self.audio_input_dim = config.audio_input_dim
        self.text_input_dim = config.text_input_dim

        fuse_method = config.fuse_method[0]

        self.cfg_dropout = ClassifierFreeGuidanceDropout(config.cond_dropout)

        if config.loss_fnc == "l1":
            self.loss_fnc = torch.nn.L1Loss()
        elif config.loss_fnc == "l2":
            self.loss_fnc = torch.nn.MSELoss()
        elif config.loss_fnc == "l1_smooth":
            self.loss_fnc = torch.nn.SmoothL1Loss()

        if self.dim_out == 8:
            self.contact_loss_func = torch.nn.BCEWithLogitsLoss()

        self.project_audio = (
            nn.Linear(self.audio_input_dim, self.dim)
            if self.audio_input_dim != self.dim
            else nn.Identity()
        )
        self.project_text = (
            nn.Linear(self.text_input_dim, self.dim)
            if self.text_input_dim != self.dim
            else nn.Identity()
        )

        self.condition_fuser = ConditionFuser(fuse_method)

        positional_embedding_params = PositionalEmbeddingParams(dim=self.dim)

        self.pos_emb = get_obj_from_str(
            PositionalEmbeddingType[config.positional_embedding_type].value
        )(positional_embedding_params)

        self.emb_dropout = nn.Dropout(config.emb_dropout)

        if "cross_seperate" in fuse_method:
            cross_attention_params2 = AttentionParams(
                dim=config.dim,
                causal=False,
                add_null_kv=True,
            )
            self.transformer_blocks = SeperateCrossTransformerBlock(
                self_attention_params=self_attention_params,
                cross_attention_params_audio=cross_attention_params,  ## audio
                cross_attention_params_text=cross_attention_params2,  ## txt
                depth=config.depth,
                ff_mult=config.ff_mult,
            )
        else:

            self.transformer_blocks = TransformerBlock(
                self_attention_params=self_attention_params,
                cross_attention_params=cross_attention_params,
                depth=config.depth,
                ff_mult=config.ff_mult,
            )
        self.out_norm = LayerNorm(self.dim)

        self.to_out = nn.Linear(self.dim, config.dim_out, bias=False)

        self.post_emb_norm = (
            nn.LayerNorm(self.dim) if config.post_emb_norm else nn.Identity()
        )

    def _prepare_inputs(self, input, conditions):
        audio_embed = self.project_audio(conditions["audio"][0])
        text_embed = self.project_text(conditions["text"][0])
        conditions["audio"] = (audio_embed, conditions["audio"][1])
        conditions["text"] = (text_embed, conditions["text"][1])

        inputs_, cross_inputs = self.condition_fuser(input, conditions)

        return inputs_, cross_inputs

    # def forward_with_cond_scale(
    #     self, *args, cond_scale=3.0, return_embed=False, **kwargs
    # ):
    #     if cond_scale == 1:
    #         return self.forward(
    #             *args, return_embed=return_embed, cond_drop_prob=0.0, **kwargs
    #         )

    #     logits, embed = self.forward(
    #         *args, return_embed=True, cond_drop_prob=0.0, **kwargs
    #     )

    #     null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)

    #     scaled_logits = null_logits + (logits - null_logits) * cond_scale

    #     if return_embed:
    #         return scaled_logits, embed

    #     return scaled_logits

    # def forward_with_neg_prompt(
    #     self,
    #     *args,
    #     text_embed: torch.Tensor,
    #     neg_text_embed: torch.Tensor,
    #     cond_scale=3.0,
    #     return_embed=False,
    #     **kwargs,
    # ):
    #     neg_logits = self.forward(
    #         *args, neg_text_embed=neg_text_embed, cond_drop_prob=0.0, **kwargs
    #     )
    #     pos_logits, embed = self.forward(
    #         *args,
    #         return_embed=True,
    #         text_embed=text_embed,
    #         cond_drop_prob=0.0,
    #         **kwargs,
    #     )

    #     logits = neg_logits + (pos_logits - neg_logits) * cond_scale

    #     if return_embed:
    #         return logits, embed

    #     return logits

    def forward(
        self,
        inputs,
        conditions,
        return_embed: bool = False,
        cond_drop_prob: float = None,
    ) -> TranslationTransformerOutput:

        motion = inputs["motion"][0].clone()
        motion_padding_mask = inputs["motion"][1].clone()

        device, b, n, d = motion.device, *motion.shape

        x = self.pos_emb(motion).repeat(b, 1, 1) * motion_padding_mask.unsqueeze(-1)

        x = self.post_emb_norm(x)
        x = self.emb_dropout(x)

        x_tple = (x, motion_padding_mask)

        conditions = self.cfg_dropout(conditions, cond_drop_prob)
        inputs_, cross_inputs_ = self._prepare_inputs(x_tple, conditions)

        x_ = inputs_[0]
        x_padding_mask = inputs_[1]
        context = cross_inputs_[0]
        context_padding_mask = cross_inputs_[1]

        embed = self.transformer_blocks(
            x=x_,
            mask=x_padding_mask,
            context=context,
            context_mask=context_padding_mask,
        )

        embed = self.out_norm(embed)

        if len(self.condition_fuser.fuse2cond.get("prepend", [])) > 0:
            embed = embed[:, -n:, :]

        embed = self.to_out(embed)
        embed = embed * motion_padding_mask.unsqueeze(-1)

        if return_embed:
            return TranslationTransformerOutput(pred_motion=embed)

        # translation_gt = motion[..., :4]

        recon_loss = self.loss_fnc(motion[..., :4], embed[..., :4])
        contact_loss = None

        if self.dim_out == 8:
            contact_loss = self.contact_loss_func(
                input=embed[..., -4:], target=motion[..., -4:]
            )

        return TranslationTransformerOutput(
            pred_motion=embed, recon_loss=recon_loss, contact_loss=contact_loss
        )
