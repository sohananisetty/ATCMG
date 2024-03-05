import math
import pathlib
from functools import partial
from pathlib import Path
from random import random
from typing import Callable, List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from core import (
    AttentionParams,
    PositionalEmbeddingParams,
    PositionalEmbeddingType,
    TranslationTransformerParams,
)
from core.datasets.conditioner import ConditionFuser
from core.models.attend2 import Attend, Attention
from core.models.utils import FeedForward, LayerNorm, default, exists, get_obj_from_str
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from yacs.config import CfgNode as CN
from copy import deepcopy
from core import MotionTokenizerParams
from core.models.resnetVQ.vqvae import HumanVQVAE

ConditionType = Tuple[torch.Tensor, torch.Tensor]
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


def get_mask_subset_prob(mask, prob, min_mask=0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim=-1, keepdim=True) * prob).clamp(min=min_mask)
    logits = torch.rand((batch, seq), device=device)
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


class GEGLU(nn.Module):
    """https://arxiv.org/abs/2002.05202"""

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return gate * F.gelu(x)


def FeedForward(dim, mult=4):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias=False),
    )


class TransformerBlockMuse(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        attn_dropout: float = 0.0,
        cross_attn_tokens_dropout: float = 0.0,
        flash: bool = False,
        causal: bool = True,
        depth: int = 1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.do_cross = causal

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dropout=attn_dropout,
                            cross_attn_tokens_dropout=cross_attn_tokens_dropout,
                            add_null_kv=True,
                            flash=flash,
                        ),
                        (
                            Attention(
                                dim=dim,
                                heads=heads,
                                dropout=attn_dropout,
                                cross_attn_tokens_dropout=cross_attn_tokens_dropout,
                                add_null_kv=True,
                                flash=flash,
                                causal=True,
                            )
                            if self.do_cross
                            else nn.Identity()
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

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


# transformer - it's all we need


class ClassifierFreeGuidanceDropout(nn.Module):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """

    def __init__(self, p: float = 0.0, seed: int = 42):
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

            if drop_prob == 1.0:

                drop_mask = self.prob_mask_like((b, 1), 1.0 - drop_prob, mask.device)
                new_mask = mask & drop_mask
                new_embedding = embedding * new_mask.unsqueeze(-1)
                # if condition_modality == "audio":
                new_embedding = new_embedding[:, :1, :]
                new_mask = new_mask[:, :1]
                conditions_[condition_modality] = (new_embedding, new_mask)

            elif drop_prob > 0.0 and (
                condition_modality != "audio" or not self.training
            ):
                drop_mask = self.prob_mask_like((b, 1), 1.0 - drop_prob, mask.device)
                new_mask = mask & drop_mask

                new_embedding = embedding * new_mask.unsqueeze(-1)

                conditions_[condition_modality] = (new_embedding, new_mask)

        return conditions_

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class Transformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        positional_embedding_type="SINE",
        emb_dropout=0.0,
        cond_dropout=0.0,
        post_emb_norm: bool = False,
        audio_input_dim: int = 128,
        text_input_dim: int = 768,
        fuse_method: Dict[str, List[str]] = {"cross": ["audio"], "prepend": ["text"]},
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.cond_dropout = cond_dropout

        motion_tokenizer_params = MotionTokenizerParams(num_tokens)

        self.num_tokens = motion_tokenizer_params.num_tokens
        self.mask_token_id = motion_tokenizer_params.mask_token_id
        self.pad_token_id = motion_tokenizer_params.pad_token_id
        self.token_emb = nn.Embedding(motion_tokenizer_params.vocab_size, dim)
        self.dim_out = num_tokens
        # default(dim_out, num_tokens)

        positional_embedding_params = PositionalEmbeddingParams(dim=self.dim)

        self.pos_emb = get_obj_from_str(
            PositionalEmbeddingType[positional_embedding_type].value
        )(positional_embedding_params)

        self.transformer_blocks = TransformerBlockMuse(dim=dim, **kwargs)
        self.norm = LayerNorm(dim)
        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()

        if isinstance(fuse_method, list):
            fuse_method = fuse_method[0]

        self.condition_fuser = ConditionFuser(fuse_method)

        self.cfg_dropout = ClassifierFreeGuidanceDropout()
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

        self.dim_out = num_tokens
        # default(dim_out, num_tokens)
        self.to_logits = nn.Linear(dim, self.dim_out, bias=False)

    def _prepare_inputs(self, input, conditions):
        audio_embed = self.project_audio(conditions["audio"][0])
        text_embed = self.project_text(conditions["text"][0])
        # conditions["audio"] = (audio_embed, conditions["audio"][1])
        # conditions["text"] = (text_embed, conditions["text"][1])

        # {
        #     "text": (text_embed, conditions["text"][1]),
        #     "audio": (audio_embed, conditions["audio"][1]),
        # }

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

    def forward(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        return_embed: bool = False,
        return_logits: bool = False,
        labels: Optional[torch.IntTensor] = None,
        cond_drop_prob: float = None,
        ignore_index: int = -100,
    ):

        assert len(inputs[0].shape) == 2, "input motion not b n"
        device, b, n = inputs[0].device, *inputs[0].shape
        cond_drop_prob = default(cond_drop_prob, self.cond_dropout)

        x = inputs[0]
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        x = self.post_emb_norm(x)
        x = self.emb_dropout(x)

        # classifier free guidance
        if cond_drop_prob > 0.0:
            conditions = self.cfg_dropout(conditions, cond_drop_prob)
        # fuse conditions

        # print(x.shape, conditions.keys())
        # print(conditions["audio"][0].shape, conditions["text"][0].shape)
        inputs_, cross_inputs_ = self._prepare_inputs((x, inputs[1]), conditions)

        x_ = inputs_[0]
        x_padding_mask = inputs_[1]
        context = cross_inputs_[0]
        context_padding_mask = cross_inputs_[1]

        # print(x_.shape, x_padding_mask.shape)
        # print(context.shape, context_padding_mask.shape)

        embed = self.transformer_blocks(
            x=x_,
            mask=x_padding_mask,
            context=context,
            context_mask=context_padding_mask,
        )
        logits = self.to_logits(embed)
        if len(self.condition_fuser.fuse2cond.get("prepend", [])) > 0:
            logits = logits[:, -n:, :]

        if return_embed:
            return logits, embed

        if not exists(labels):
            return logits

        loss = F.cross_entropy(
            rearrange(logits, "b n c -> b c n"), labels, ignore_index=ignore_index
        )

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
    probs.scatter_(2, ind, val)
    return probs


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


# noise schedules


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


# main maskgit classes


class MotionMuse(nn.Module):
    def __init__(
        self,
        config,
        noise_schedule: Callable = cosine_schedule,
        vqvae: Optional[HumanVQVAE] = None,
        # no_mask_token_prob=0.2,
    ):
        super().__init__()
        self.vqvae = vqvae.eval().freeze() if exists(vqvae) else None
        self.no_mask_token_prob = config.pop("no_mask_token_prob")
        self.transformer = Transformer(**config)

        # self.transformer = transformer

        self.mask_token_id = self.transformer.mask_token_id
        self.noise_schedule = noise_schedule

        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        # may be needed for self conditioning
        # self.no_mask_token_prob = no_mask_token_prob

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    def muse_mask(self, code_ids: torch.Tensor, ignore_index: int = -100):
        batch, seq_len, device = (
            *code_ids.shape,
            code_ids.device,
        )

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        # mask_id = self.mask_token_id
        batch_randperm = torch.rand((batch, seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask[code_ids == self.transformer.pad_token_id] = False

        # mask_id = self.transformer.mask_token_id
        labels = torch.where(mask, code_ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, self.mask_token_id, code_ids)

        return x, labels

    def bert_muse_mask(self, code_ids: torch.Tensor, ignore_index: int = -100):
        batch, seq_len, device = (
            *code_ids.shape,
            code_ids.device,
        )

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)

        # probability_matrix = torch.full(code_ids.shape, self.mlm_probability)

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        # mask_id = self.mask_token_id
        batch_randperm = torch.rand((batch, seq_len), device=device)
        batch_randperm[code_ids == self.transformer.pad_token_id] = 10.0
        batch_randperm = batch_randperm.argsort(dim=-1)

        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask[code_ids == self.transformer.pad_token_id] = False

        # mask_id = self.transformer.mask_token_id
        labels = torch.where(mask, code_ids, ignore_index)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(code_ids.shape, 0.8)).bool() & mask
        )
        code_ids[indices_replaced] = self.transformer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(code_ids.shape, 0.5)).bool()
            & mask
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.transformer.num_tokens, code_ids.shape, dtype=torch.long
        ).to(device)
        code_ids[indices_random] = random_words[indices_random]

        return code_ids, labels

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conditions: Dict[str, ConditionType],
        neg_conditions: Optional[Dict[str, ConditionType]] = None,
        duration: int = 75,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
    ):

        # begin with all image token ids masked

        device = next(self.parameters()).device

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
            print("need atleast one condition")
            return

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
        mask = torch.ones_like(ids).to(torch.bool)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        demask_fn = self.transformer.forward_with_cond_scale

        # negative prompting, as in paper

        if exists(neg_conditions):

            demask_fn = partial(
                self.transformer.forward_with_neg_prompt,
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
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(1, masked_indices, self.mask_token_id)

            logits, embed = demask_fn(
                inputs=(ids, mask),
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

            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
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
    ):
        # tokenize if needed

        motions_or_ids = inputs[0]
        input_mask = inputs[1]

        if motions_or_ids.dtype == torch.float and len(motions_or_ids.shape) == 3:
            assert exists(
                self.vqvae
            ), "human vqvae must be passed in if training from raw motions"

            with torch.no_grad():
                code_ids = self.vqvae.encode(motions_or_ids)
        else:
            assert motions_or_ids.dtype == torch.long, "code ids should be type long"
            code_ids = motions_or_ids

        code_ids = rearrange(code_ids, "b ... -> b (...)")

        x, labels = self.muse_mask(code_ids, ignore_index)

        # get loss

        ce_loss, logits = self.transformer(
            inputs=(x, input_mask),
            conditions=conditions,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            return_logits=True,
        )

        if not return_logits:
            return ce_loss

        return ce_loss, logits
