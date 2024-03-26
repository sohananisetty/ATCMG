import math
import pathlib
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import typing as tp
import torch
import torch.nn.functional as F
from core import (
    AttentionParams,
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
    TranslationTransformerOutput,
    TranslationTransformerParams,
)
from core.datasets.conditioner import ConditionFuser
from core.models.attend import Attend, Attention
from core.models.utils import default, exists, get_obj_from_str
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN
from core.datasets.conditioner import ClassifierFreeGuidanceDropout, ConditionFuser
from core.models.attend2 import CustomMHA
from core.models.positional_embeddings import ScaledSinusoidalEmbedding
from core.models.resnetVQ.encdec import Encoder, Decoder
from utils.motion_processing.quaternion import qinv, qrot, quaternion_to_cont6d
import utils.rotation_conversions as geometry

ConditionType = Tuple[torch.Tensor, torch.Tensor]
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


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


class TransformerBlockCustom(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        CustomMHA(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            add_null_kv=False,
                            causal=False,
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
                        FeedForward(dim=dim, mult=ff_mult, dropout=attn_dropout),
                    ]
                )
            )

        self.norm_sa = LayerNorm(dim)
        self.norm_cross = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(self, x, mask=None, context=None, context_mask=None, rel_pos=None):
        for attn, cross_attn, ff1 in self.layers:

            x = self.norm_sa(x)
            x = (
                attn(
                    q=x,
                    k=x,
                    v=x,
                    key_padding_mask=mask,
                    rel_pos=rel_pos,
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

            x = self.norm_out(ff1(x) + x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        flash: bool = False,
        depth: int = 1,
        ff_mult: int = 4,
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


class EncTransDec(nn.Module):
    def __init__(
        self,
        fuser: ConditionFuser,
        input_dim,
        dim,
        dim_out,
        conv_depth=1,
        down_sampling_ratio=4,
        loss_fnc="l1_smooth",
        var_len=True,
        custom=True,
        quality_emb=False,
        emb_dropout=0.0,
        cond_dropout=0.0,
        post_emb_norm: bool = False,
        audio_input_dim: int = 128,
        text_input_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.cond_dropout = cond_dropout
        self.var_len = var_len
        # self.quality_emb = quality_emb
        self.down_sampling_ratio = down_sampling_ratio

        # if quality_emb:

        #     self.qual_emb = nn.Embedding(2, self.dim)

        self.pos_emb = ScaledSinusoidalEmbedding(
            PositionalEmbeddingParams(dim=self.dim)
        )
        self.conv_depth = conv_depth

        if conv_depth != 0:

            self.encoder = Encoder(
                input_dim,
                dim,
                down_sampling_ratio // 2,
                depth=conv_depth,
            )
            self.decoder = Decoder(
                dim_out,
                dim,
                down_sampling_ratio // 2,
                depth=conv_depth,
            )

        else:
            self.project_input = (
                nn.Linear(input_dim, self.dim, bias=False)
                if input_dim != self.dim
                else nn.Identity()
            )

        if custom:
            self.transformer_blocks = TransformerBlockCustom(dim=dim, **kwargs)
        else:
            self.transformer_blocks = TransformerBlock(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()

        self.condition_fuser = fuser

        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=self.cond_dropout)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_audio = (
            nn.Linear(self.audio_input_dim, self.dim, bias=False)
            if self.audio_input_dim != self.dim
            else nn.Identity()
        )
        self.project_text = (
            nn.Linear(self.text_input_dim, self.dim, bias=False)
            if self.text_input_dim != self.dim
            else nn.Identity()
        )

        self.out_norm = LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim_out)

        if loss_fnc == "l1":
            self.loss_fnc = torch.nn.L1Loss()
        elif loss_fnc == "l2":
            self.loss_fnc = torch.nn.MSELoss()
        elif loss_fnc == "l1_smooth":
            self.loss_fnc = torch.nn.SmoothL1Loss()

    def _prepare_inputs(self, input, conditions):
        audio_embed = self.project_audio(conditions["audio"][0])
        text_embed = self.project_text(conditions["text"][0])

        new_conditions = {
            "audio": (audio_embed, conditions["audio"][1]),
            "text": (text_embed, conditions["text"][1]),
        }

        inputs_, cross_inputs = self.condition_fuser(input, new_conditions)

        return inputs_, cross_inputs

    def forward(
        self,
        inputs: ConditionTensors,
        conditions: ConditionTensors,
        labels: Optional[torch.Tensor] = None,
        cond_drop_prob: float = None,
        # quality_list=None,
    ):

        sequence = inputs[0]
        sequence_mask = inputs[1]
        B, S, D = sequence.shape

        # if self.quality_emb is True and quality_list is None:
        #     quality_list = torch.ones(B, dtype=torch.long, device=sequence.device)

        # classifier free guidance
        cond_drop_prob = default(cond_drop_prob, self.cond_dropout)
        if cond_drop_prob > 0.0:
            conditions = self.cfg_dropout(conditions, cond_drop_prob)

        if self.conv_depth != 0:

            sequence = sequence.contiguous().permute(0, 2, 1)  ## b n d -> b d n
            sequence = self.encoder(sequence)
            sequence = sequence.permute(0, 2, 1)
            sequence_mask = torch.nn.functional.interpolate(
                sequence_mask[None].float(), scale_factor=(1 / self.down_sampling_ratio)
            )[0].to(torch.bool)

        else:
            sequence = self.project_input(sequence)

        input_, cross_attention_input = self._prepare_inputs(
            (sequence, sequence_mask), conditions
        )

        x_trans = input_[0]
        x_padding_mask = input_[1]
        context = cross_attention_input[0]
        context_padding_mask = cross_attention_input[1]

        x_trans = x_trans + self.pos_emb(x_trans)
        x_trans = self.post_emb_norm(x_trans)
        x_trans = self.emb_dropout(x_trans)

        # if quality_list is not None and self.quality_emb == True:
        #     x_trans = torch.cat(
        #         [self.qual_emb(quality_list).unsqueeze(1), x_trans], dim=-2
        #     )
        #     x_padding_mask = F.pad(x_padding_mask, (1, 0), value=True)

        embed = self.transformer_blocks(
            x=x_trans,
            mask=x_padding_mask if self.var_len else None,
            context=context,
            context_mask=context_padding_mask,
        )

        if self.conv_depth != 0:

            embed = embed.permute(0, 2, 1)
            embed = self.decoder(embed)
            embed = embed.permute(0, 2, 1)

        else:
            embed = self.out_proj(embed)

        if (
            len(self.condition_fuser.fuse2cond.get("prepend", []))
            > 0
            # or self.quality_emb
        ):
            logits = logits[:, :, -S:]

        if exists(labels):
            loss = 10 * self.loss_fnc(embed, labels)

            return loss, embed

        return embed


class TranslationTransformer(nn.Module):
    def __init__(
        self,
        tranformer_config,
        fuse_config,
    ):
        super().__init__()

        fuse_method = fuse_config.pop("fuse_method")
        if isinstance(fuse_method, list):
            fuse_method = fuse_method[0]
        condition_fuser = ConditionFuser(fuse_method, **fuse_config)

        if condition_fuser.cond2fuse["audio"] == "cross":
            assert (
                tranformer_config.custom == True
            ), "when audio is cross attention, you need custom attention"

        self.model = EncTransDec(
            fuser=condition_fuser,
            **tranformer_config,
        )

        self.dim_out = tranformer_config.dim_out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict["model"])

    def recover_root_rot_pos(self, data):
        rot_vel = data[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        """Get Y-axis rotation from rotation velocity"""
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
        data = data.to(torch.float)
        r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
        """Add Y-axis rotation to root position"""
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data[..., 3]
        return r_rot_quat, r_pos

    def predict(self, trajectory, conditions=None, cond_drop_prob=0.0):

        assert trajectory.shape[-1] == 2, "trajectory is xz"
        pred_r_rot = self.model(
            inputs=(trajectory, torch.ones_like(trajectory)[..., 0]),
            conditions=conditions,
            cond_drop_prob=cond_drop_prob,
            # quality_list=quality_list,
        )

        if self.dim_out == 1:
            pred_r_rot = torch.nn.functional.pad(pred_r_rot, (1, 1), value=0)
        elif self.dim_out == 2:
            pred_r_rot = torch.zeros(
                (pred_r_rot.shape[:-1] + (4,)),
                device=pred_r_rot.device,
                dtype=pred_r_rot.dtype,
            )
            pred_r_rot[..., [0, 2]] = pred_r_rot[..., [0, 2]]
        elif self.dim_out == 6:
            pred_r_rot = geometry.matrix_to_quaternion(
                geometry.rotation_6d_to_matrix(pred_r_rot)
            )

        return pred_r_rot

    def forward(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        cond_drop_prob=None,
        return_embed=True,
        # quality_list=None,
    ):
        # tokenize if needed

        motion_trajectory = inputs[0].to(torch.long)
        input_mask = inputs[1].to(torch.bool)
        r_rot, r_pos = self.recover_root_rot_pos(motion_trajectory)
        r_pos = r_pos[..., [0, 2]]

        if self.dim_out == 1:
            r_rot = geometry.quaternion_to_axis_angle(r_rot)[..., 1:2]
        elif self.dim_out == 2:
            r_rot = r_rot[..., [0, 2]]
        elif self.dim_out == 6:
            r_rot = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(r_rot))

        # get loss

        loss, embed = self.model(
            inputs=(r_pos, input_mask),
            conditions=conditions,
            labels=r_rot,
            cond_drop_prob=cond_drop_prob,
            # quality_list=quality_list,
        )

        if self.dim_out == 1:
            pred_r_rot_aa = torch.nn.functional.pad(embed, (1, 1), value=0)
            pred_r_rot = geometry.axis_angle_to_quaternion(pred_r_rot_aa)
        elif self.dim_out == 2:
            pred_r_rot = torch.zeros(
                (embed.shape[:-1] + (4,)), device=embed.device, dtype=embed.dtype
            )
            pred_r_rot[..., [0, 2]] = embed[..., [0, 2]]
        elif self.dim_out == 6:
            pred_r_rot = geometry.matrix_to_quaternion(
                geometry.rotation_6d_to_matrix(embed)
            )

        if not return_embed:
            return loss

        return loss, pred_r_rot


# class TranslationTransformer(nn.Module):
#     def __init__(self, config: CN):
#         super().__init__()

#         self_attention_params = AttentionParams(
#             dim=config.dim, causal=config.is_self_causal
#         )
#         cross_attention_params = AttentionParams(
#             dim=config.dim,
#             causal=config.is_cross_causal,
#             add_null_kv=True,
#         )
#         # transformer_params = TranslationTransformerParams(
#         #     self_attention_params=sap,
#         #     cross_attention_params=cap,
#         #     depth=config.depth,
#         #     positional_embedding_params=PositionalEmbeddingParams(dim=config.dim),
#         #     positional_embedding=PositionalEmbeddingType.SINE,
#         #     fuse_method=config.fuse_method[0],
#         #     cond_dropout=config.cond_dropout,
#         #     audio_input_dim=config.audio_input_dim,
#         #     text_input_dim=config.text_input_dim,
#         #     dim_out=config.dim_out,
#         #     loss_fnc=config.loss_fnc,
#         # )

#         self.dim = config.dim
#         self.dim_out = config.dim_out
#         self.audio_input_dim = config.audio_input_dim
#         self.text_input_dim = config.text_input_dim

#         fuse_method = config.fuse_method[0]

#         self.cfg_dropout = ClassifierFreeGuidanceDropout(config.cond_dropout)

#         if config.loss_fnc == "l1":
#             self.loss_fnc = torch.nn.L1Loss()
#         elif config.loss_fnc == "l2":
#             self.loss_fnc = torch.nn.MSELoss()
#         elif config.loss_fnc == "l1_smooth":
#             self.loss_fnc = torch.nn.SmoothL1Loss()

#         if self.dim_out == 8:
#             self.contact_loss_func = torch.nn.BCEWithLogitsLoss()

#         self.project_audio = (
#             nn.Linear(self.audio_input_dim, self.dim)
#             if self.audio_input_dim != self.dim
#             else nn.Identity()
#         )
#         self.project_text = (
#             nn.Linear(self.text_input_dim, self.dim)
#             if self.text_input_dim != self.dim
#             else nn.Identity()
#         )

#         self.condition_fuser = ConditionFuser(fuse_method)

#         positional_embedding_params = PositionalEmbeddingParams(dim=self.dim)

#         self.pos_emb = get_obj_from_str(
#             PositionalEmbeddingType[config.positional_embedding_type].value
#         )(positional_embedding_params)

#         self.emb_dropout = nn.Dropout(config.emb_dropout)

#         if "cross_seperate" in fuse_method:
#             cross_attention_params2 = AttentionParams(
#                 dim=config.dim,
#                 causal=False,
#                 add_null_kv=True,
#             )
#             self.transformer_blocks = SeperateCrossTransformerBlock(
#                 self_attention_params=self_attention_params,
#                 cross_attention_params_audio=cross_attention_params,  ## audio
#                 cross_attention_params_text=cross_attention_params2,  ## txt
#                 depth=config.depth,
#                 ff_mult=config.ff_mult,
#             )
#         else:

#             self.transformer_blocks = TransformerBlock(
#                 self_attention_params=self_attention_params,
#                 cross_attention_params=cross_attention_params,
#                 depth=config.depth,
#                 ff_mult=config.ff_mult,
#             )
#         self.out_norm = LayerNorm(self.dim)

#         self.to_out = nn.Linear(self.dim, config.dim_out, bias=False)

#         self.post_emb_norm = (
#             nn.LayerNorm(self.dim) if config.post_emb_norm else nn.Identity()
#         )

#     def _prepare_inputs(self, input, conditions):
#         audio_embed = self.project_audio(conditions["audio"][0])
#         text_embed = self.project_text(conditions["text"][0])
#         conditions["audio"] = (audio_embed, conditions["audio"][1])
#         conditions["text"] = (text_embed, conditions["text"][1])

#         inputs_, cross_inputs = self.condition_fuser(input, conditions)

#         return inputs_, cross_inputs

#     def forward(
#         self,
#         inputs,
#         conditions,
#         return_embed: bool = False,
#         cond_drop_prob: float = None,
#     ) -> TranslationTransformerOutput:

#         motion = inputs["motion"][0].clone()
#         motion_padding_mask = inputs["motion"][1].clone()

#         device, b, n, d = motion.device, *motion.shape

#         x = self.pos_emb(motion).repeat(b, 1, 1) * motion_padding_mask.unsqueeze(-1)

#         x = self.post_emb_norm(x)
#         x = self.emb_dropout(x)

#         x_tple = (x, motion_padding_mask)

#         conditions = self.cfg_dropout(conditions, cond_drop_prob)
#         inputs_, cross_inputs_ = self._prepare_inputs(x_tple, conditions)

#         x_ = inputs_[0]
#         x_padding_mask = inputs_[1]
#         context = cross_inputs_[0]
#         context_padding_mask = cross_inputs_[1]

#         embed = self.transformer_blocks(
#             x=x_,
#             mask=x_padding_mask,
#             context=context,
#             context_mask=context_padding_mask,
#         )

#         embed = self.out_norm(embed)

#         if len(self.condition_fuser.fuse2cond.get("prepend", [])) > 0:
#             embed = embed[:, -n:, :]

#         embed = self.to_out(embed)
#         embed = embed * motion_padding_mask.unsqueeze(-1)

#         if return_embed:
#             return TranslationTransformerOutput(pred_motion=embed)

#         # translation_gt = motion[..., :4]

#         recon_loss = self.loss_fnc(motion[..., :4], embed[..., :4])
#         contact_loss = None

#         if self.dim_out == 8:
#             contact_loss = self.contact_loss_func(
#                 input=embed[..., -4:], target=motion[..., -4:]
#             )

#         return TranslationTransformerOutput(
#             pred_motion=embed, recon_loss=recon_loss, contact_loss=contact_loss
#         )
