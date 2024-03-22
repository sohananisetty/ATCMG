import math
import pathlib
import typing as tp
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from core import (AttentionParams, MotionTokenizerParams,
                  PositionalEmbeddingParams, PositionalEmbeddingType,
                  TranslationTransformerParams)
from core.datasets.conditioner import (ClassifierFreeGuidanceDropout,
                                       ConditionFuser)
from core.models.attend2 import Attend, Attention, CustomMHA
from core.models.positional_embeddings import ScaledSinusoidalEmbedding
from core.models.resnetVQ.vqvae import HumanVQVAE
from core.models.utils import (FeedForward, LayerNorm, default, exists,
                               get_obj_from_str)
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN

from .streaming_transformer.codebooks_patterns import CodebooksPatternProvider

ConditionType = Tuple[torch.Tensor, torch.Tensor]
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]

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


# tensor helpers


# def get_mask_subset_prob(mask, prob, min_mask=0):
#     batch, seq, device = *mask.shape, mask.device
#     num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
#     logits = torch.rand((batch, seq), device=device)
#     logits = logits.masked_fill(~mask, -1)

#     randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

#     num_padding = (~mask).sum(dim=-1, keepdim=True)
#     randperm -= num_padding

#     subset_mask = randperm < num_to_mask
#     subset_mask.masked_fill_(~mask, False)
#     return subset_mask


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, K, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, K, seq), device=device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim=-1).argsort(dim=-1).float()

    num_padding = (~mask).sum(dim=-1, keepdim=True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask


# classes


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


def FeedForward2(
    dim,
    mult=4,
    dropout=0.1,
):

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


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
                            add_null_kv=False,
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


class TransformerBlockMuse(nn.Module):
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


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class MLMModel(nn.Module):
    def __init__(
        self,
        pattern_provider: CodebooksPatternProvider,
        fuser: ConditionFuser,
        num_tokens,
        dim,
        n_q=3,
        positional_embedding_type="SINE",
        var_len=True,
        custom=True,
        quality_emb=False,
        emb_lr: tp.Optional[float] = None,
        bias_proj=False,
        emb_dropout=0.0,
        cond_dropout=0.0,
        post_emb_norm: bool = False,
        audio_input_dim: int = 128,
        text_input_dim: int = 768,
        # fuse_method: Dict[str, List[str]] = {"cross": ["audio"], "prepend": ["text"]},
        proj_input=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.n_q = n_q
        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.cond_dropout = cond_dropout
        self.pattern_provider = pattern_provider
        self.var_len = var_len
        self.quality_emb = quality_emb

        self.set_token_ids(num_tokens, False)

        self.token_emb = nn.ModuleList(
            [ScaledEmbedding(self.vocab_size, self.dim, lr=emb_lr) for _ in range(n_q)]
        )

        if quality_emb:

            self.qual_emb = nn.Embedding(2, self.dim)

        self.pos_emb = ScaledSinusoidalEmbedding(
            PositionalEmbeddingParams(dim=self.dim)
        )

        if custom:
            self.transformer_blocks = TransformerBlockMuseCustom(dim=dim, **kwargs)
        else:
            self.transformer_blocks = TransformerBlockMuse(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()

        self.condition_fuser = fuser

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
        self.project_text = (
            nn.Linear(self.text_input_dim, self.dim, bias=False)
            if self.text_input_dim != self.dim
            else nn.Identity()
        )

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.num_tokens, bias=bias_proj) for _ in range(n_q)]
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

    def _prepare_inputs(self, input, conditions):
        audio_embed = self.project_audio(conditions["audio"][0])
        text_embed = self.project_text(conditions["text"][0])

        inputs_, cross_inputs = self.condition_fuser(
            input,
            {
                "text": (text_embed, conditions["text"][1]),
                "audio": (audio_embed, conditions["audio"][1]),
            },
        )

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
            return self.forward(
                inputs,
                conditions,
                return_embed=return_embed,
                cond_drop_prob=0.0,
                **kwargs,
            )

        # print("*")
        # print(inputs[0].shape)
        # print(conditions["audio"][0].shape, conditions["text"][0].shape)

        logits, embed = self.forward(
            inputs, conditions, return_embed=True, cond_drop_prob=0.0, **kwargs
        )
        # print("*")
        # print(inputs[0].shape)
        # print(conditions["audio"][0].shape, conditions["text"][0].shape)

        null_logits = self.forward(inputs, conditions, cond_drop_prob=1.0, **kwargs)

        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

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
        neg_logits = self.forward(
            inputs,
            neg_conditions,
            cond_drop_prob=0.0,
            **kwargs,
        )
        pos_logits, embed = self.forward(
            inputs,
            conditions,
            return_embed=True,
            cond_drop_prob=0.0,
            **kwargs,
        )

        scaled_logits = neg_logits + (pos_logits - neg_logits) * cond_scale

        if return_embed:
            return scaled_logits, embed

        return scaled_logits

    def compute_predictions(
        self,
        inputs: ConditionTensors,
        condition_tensors: Optional[ConditionTensors] = None,
        labels: Optional[torch.IntTensor] = None,
        keep_only_valid_steps: bool = True,
        cond_drop_prob: float = None,
    ) -> LMOutput:
        """Given an input tensor of codes [B, T, K] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, T, K] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list of ConditioningAttributes): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token in that case.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """

        codes = inputs[0]
        codes_mask = inputs[1]
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = (
            pattern.build_pattern_sequence(
                codes,
                self.special_token_id,
                keep_only_valid_steps=keep_only_valid_steps,
            )
        )
        new_codes_mask = torch.ones_like(sequence_mask).repeat(B, 1, 1)
        for i in range(K):
            new_codes_mask[:, i, i + 1 :] = codes_mask[:, 0 : T - i]

        new_new_mask = new_codes_mask.sum(1) == K

        # apply model on pattern sequence
        logits = self(
            inputs=(sequence_codes, new_new_mask.to(torch.bool)),
            conditions=condition_tensors,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
        )  # [B, K, S, card]
        # print(logits)
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float("nan"), keep_only_valid_steps=keep_only_valid_steps
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        # mask = new_codes_mask & logits_mask
        return LMOutput(logits, logits_mask)

    def forward(
        self,
        inputs: ConditionTensors,
        conditions: ConditionTensors,
        return_embed: bool = False,
        return_logits: bool = False,
        labels: Optional[torch.IntTensor] = None,
        cond_drop_prob: float = None,
        ignore_index: int = -100,
        quality_list=None,
    ):

        sequence = inputs[0]
        sequence_mask = inputs[1]
        B, K, S = sequence.shape

        if self.quality_emb is True and quality_list is None:
            quality_list = torch.ones(B, dtype=torch.long, device=sequence.device)

        assert (
            K == self.num_codebooks
        ), "Sequence shape must match the specified number of codebooks"

        if isinstance(self.project_input, nn.Identity):
            input_ = sum([self.token_emb[k](sequence[:, k]) for k in range(K)])
        else:

            input_ = torch.cat(
                [self.token_emb[k](sequence[:, k]) for k in range(K)], -1
            )

            input_ = self.project_input(input_)

        # classifier free guidance
        cond_drop_prob = default(cond_drop_prob, self.cond_dropout)
        if cond_drop_prob > 0.0:
            conditions = self.cfg_dropout(conditions, cond_drop_prob)
        # fuse conditions

        # print(input_.shape)
        # print(sequence_mask.shape)
        # print(conditions["audio"][0].shape, conditions["text"][0].shape)
        # print(conditions["audio"][1].shape, conditions["text"][1].shape)

        input_, cross_attention_input = self._prepare_inputs(
            (input_, sequence_mask), conditions
        )

        x_ = input_[0]
        x_padding_mask = input_[1]
        context = cross_attention_input[0]
        context_padding_mask = cross_attention_input[1]

        # print(x_.shape, x_padding_mask.shape)
        # print(context.shape, context_padding_mask.shape)

        x_ = x_ + self.pos_emb(x_)
        x_ = self.post_emb_norm(x_)
        x_ = self.emb_dropout(x_)

        if quality_list is not None and self.quality_emb == True:
            x_ = torch.cat([self.qual_emb(quality_list).unsqueeze(1), x_], dim=-2)
            x_padding_mask = F.pad(x_padding_mask, (1, 0), value=True)

        embed = self.transformer_blocks(
            x=x_,
            mask=x_padding_mask if self.var_len else None,
            context=context,
            context_mask=None,
            # =context_padding_mask if self.var_len else None,
        )
        if self.out_norm:
            embed = self.out_norm(embed)
        logits = torch.stack([self.linears[k](embed) for k in range(K)], dim=1)

        if (
            len(self.condition_fuser.fuse2cond.get("prepend", [])) > 0
            or self.quality_emb
        ):
            logits = logits[:, :, -S:]

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits  ## B K N num_tokens

        if self.num_codebooks == 1:

            loss = F.cross_entropy(
                rearrange(logits.squeeze(1), "b n c -> b c n"),
                labels.squeeze(1),
                ignore_index=ignore_index,
            )
        else:
            loss = None

        if not return_logits:
            return loss

        return loss, logits


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes
from core import MotionTokenizerParams, pattern_providers


class MotionMuse(nn.Module):
    def __init__(
        self,
        tranformer_config,
        fuse_config,
        pattern_config,
        noise_schedule: Callable = cosine_schedule,
        vqvae: Optional[HumanVQVAE] = None,
    ):
        super().__init__()
        self.vqvae = vqvae.eval().freeze() if exists(vqvae) else None
        self.no_mask_token_prob = tranformer_config.pop("no_mask_token_prob")

        fuse_method = fuse_config.pop("fuse_method")
        if isinstance(fuse_method, list):
            fuse_method = fuse_method[0]
        condition_fuser = ConditionFuser(fuse_method, **fuse_config)
        modeling = pattern_config.pop("modeling")

        pattern_provider = pattern_providers[modeling](
            n_q=tranformer_config.n_q,
            # delays=pattern_config.delays,
            # flatten_first=pattern_config.flatten_first,
            # empty_initial=pattern_config.empty_initial,
        )

        self.model = MLMModel(
            pattern_provider=pattern_provider,
            fuser=condition_fuser,
            **tranformer_config,
        )

        self.mask_token_id = self.model.mask_token_id
        self.noise_schedule = noise_schedule
        self.num_codeboks = self.model.num_codebooks

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    def muse_mask(self, motion_ids: torch.Tensor, ignore_index: int = -100):
        batch, K, seq_len, device = (
            *motion_ids.shape,
            motion_ids.device,
        )

        code_ids = motion_ids.clone()

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * K * rand_mask_probs).round().clamp(min=1)

        # mask_id = self.mask_token_id
        batch_randperm = torch.rand((batch, K, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1 1")
        mask[code_ids == self.model.pad_token_id] = False

        # mask_id = self.transformer.mask_token_id
        labels = torch.where(mask, code_ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, self.mask_token_id, code_ids)

        return x, labels

    def bert_muse_mask(self, motion_ids: torch.Tensor, ignore_index: int = -100):
        batch, K, seq_len, device = (
            *motion_ids.shape,
            motion_ids.device,
        )

        code_ids = motion_ids.clone()

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)

        # probability_matrix = torch.full(code_ids.shape, self.mlm_probability)

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        # mask_id = self.mask_token_id
        batch_randperm = torch.rand((batch, K, seq_len), device=device)
        batch_randperm[code_ids == self.model.pad_token_id] = 10.0
        batch_randperm = batch_randperm.argsort(dim=-1)

        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1 1")
        mask[code_ids == self.model.pad_token_id] = False

        # mask_id = self.transformer.mask_token_id
        labels = torch.where(mask, code_ids, ignore_index)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(code_ids.shape, 0.8)).bool().to(device) & mask
        )
        code_ids[indices_replaced] = self.model.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(code_ids.shape, 0.5)).bool().to(device)
            & mask
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.model.num_tokens, code_ids.shape, dtype=torch.long
        ).to(device)
        code_ids[indices_random] = random_words[indices_random]

        return code_ids, labels

    def get_null_context(self, B, device, dtype=torch.float):
        cond_list = list(self.model.condition_fuser.cond2fuse.keys())

        conditions = {}
        for cond_type in cond_list:
            cond = torch.zeros(
                (B, 1, self.model.audio_input_dim),
                device=device,
                dtype=dtype,
            )
            cond_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
            conditions[cond_type] = (cond, cond_mask)

        return conditions

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conditions: Dict[str, ConditionType],
        neg_conditions: Optional[Dict[str, ConditionType]] = None,
        duration_s: int = 8,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
    ):

        # begin with all image token ids masked

        assert self.num_codeboks == 1, "only 1 codebook supported  for now"

        device = next(self.parameters()).device
        duration = int(duration_s * (30 / 4))

        seq_len = duration
        try:
            if conditions.get("text", None) is not None:

                batch_size = conditions["text"][0].shape[0]
                if (
                    neg_conditions is not None
                    and neg_conditions.get("text", None) is not None
                ):
                    assert (
                        neg_conditions["text"][0].shape[0] == batch_size
                    ), "negetive text conditions should have same number as positive "

            else:
                batch_size = len(conditions["audio"][0].shape[0])
                if (
                    neg_conditions is not None
                    and neg_conditions.get("audio", None) is not None
                ):
                    assert (
                        neg_conditions["audio"][0].shape[0] == batch_size
                    ), "negetive audio conditions should have same number as positive "

        except:
            print("using null condition")
            batch_size = 1
            conditions = self.get_null_context(batch_size, device)
            cond_scale = 1

        shape = (batch_size, self.num_codeboks, seq_len)

        ids = torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
        mask = torch.ones_like(ids).to(torch.bool)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        demask_fn = self.model.forward_with_cond_scale

        # negative prompting, as in paper

        if exists(neg_conditions):

            demask_fn = partial(
                self.model.forward_with_neg_prompt,
                neg_conditions=neg_conditions,
            )

        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(
                int((rand_mask_prob * seq_len * self.num_codeboks).item()), 1
            )

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            # print(ids.shape, masked_indices.shape)

            ids = ids.scatter(-1, masked_indices, self.mask_token_id)

            # print(ids[0, 0])

            # print(ids.shape, mask.shape)

            logits, embed = demask_fn(
                inputs=(ids, mask.squeeze(1)),
                conditions=conditions,
                cond_scale=cond_scale,
                return_embed=True,
            )

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (
                steps_until_x0 / timesteps
            )  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = ids == self.mask_token_id

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)

            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, "... 1 -> ...")

            if not can_remask_prev_masked:
                scores = scores.masked_fill(~is_mask, -1e5)
            else:
                assert (
                    self.no_mask_token_prob > 0.0
                ), "without training with some of the non-masked tokens forced to predict, not sure if the logits will be meaningful for these token"

        # get ids

        if not exists(self.vqvae):
            return ids

        motion_generated = self.vqvae.decode(ids)
        return motion_generated

    def forward(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        ignore_index: int = -100,
        cond_drop_prob=None,
        return_logits=False,
        quality_list=None,
    ):
        # tokenize if needed

        motions_ids = inputs[0].to(torch.long)
        input_mask = inputs[1].to(torch.bool)
        B, K, T = motions_ids.shape

        x, labels = self.muse_mask(motions_ids, ignore_index)

        # get loss

        ce_loss, logits = self.model(
            inputs=(x, input_mask),
            conditions=conditions,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
            quality_list=quality_list,
        )

        if not return_logits:
            return ce_loss

        return ce_loss, logits
