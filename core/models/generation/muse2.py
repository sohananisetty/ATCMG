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
from core.datasets.conditioner import ConditionFuser
from einops import rearrange, repeat
from torch import einsum, nn
from tqdm.auto import tqdm
from core.models.TMR.losses import InfoNCE_with_filtering
from core.models.TMR.tmr import TMR
from core.models.generation.mlm_model import MLMModel, MuseOutput

ConditionType = Tuple[torch.Tensor, torch.Tensor]
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


# @dataclass
# class MuseOutput:
#     loss: torch.Tensor = None
#     logits: torch.Tensor = None
#     embed: torch.Tensor = None
#     ce_per_codebook: List[torch.Tensor] = None


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


def make_copy(condition):
    copy_conditions = {}
    for condition_modality, (embedding, mask) in condition.items():
        copy_conditions[condition_modality] = (embedding.clone(), mask.clone())

    return copy_conditions


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


class TMRAlignLoss(nn.Module):
    def __init__(self, tmr, vqvae, threshold_selfsim=0.9) -> None:
        super().__init__()
        self.tmr = tmr
        self.vqvae = vqvae
        self.contrastive_loss = InfoNCE_with_filtering(
            threshold_selfsim=threshold_selfsim
        )

    def mean_pooling(self, token_embeddings, attention_mask):

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, inputs, conditions):

        ### inputs: codes, mask

        text_conds = conditions["text"]
        text_x_dict = {"x": text_conds[0], "mask": text_conds[1].to(torch.bool)}
        t_latents = self.tmr.encode(text_x_dict, sample_mean=False)
        # t_latents = torch.nn.functional.normalize(t_latents, dim=-1)

        body_ids = inputs[0][:, 0]

        body_motion = self.vqvae.decode(body_ids)
        mask = inputs[1]
        upsampled_motion_mask = (
            nn.functional.interpolate(mask[:, None, :].to(torch.float), scale_factor=4)
            .to(torch.bool)
            .squeeze(1)
        )

        print(body_motion.shape, upsampled_motion_mask.shape)

        motion_x_dict = {"x": body_motion, "mask": upsampled_motion_mask}
        m_latents = self.tmr.encode(motion_x_dict, sample_mean=True)
        # m_latents = torch.nn.functional.normalize(m_latents, dim=-1)

        sent_emb = self.mean_pooling(text_conds[0], text_conds[1].to(torch.bool))

        loss = self.contrastive_loss(t_latents, m_latents, sent_emb)

        return loss


class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.spatial = self.net.spatial
        self.num_codebooks = self.net.num_codebooks
        self.to_preds = (
            nn.Linear(net.dim, 1)
            if self.spatial
            else nn.ModuleList(
                [nn.Linear(net.dim, 1) for _ in range(self.num_codebooks)]
            )
        )
        # self.to_preds = nn.ModuleList(
        #     [nn.Linear(net.dim, 1) for _ in range(self.num_codebooks)]
        # )

    def forward_with_cond_scale(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        **kwargs,
    ):

        logits = self.forward(
            inputs, conditions=conditions, cond_drop_prob=0.0, **kwargs
        )

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            inputs, conditions=conditions, cond_drop_prob=1.0, **kwargs
        )

        return null_logits + (logits - null_logits) * cond_scale

    def forward_with_neg_prompt(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        neg_conditions: Dict[str, ConditionType],
        cond_scale=3.0,
        **kwargs,
    ):

        logits = self.forward(
            inputs, conditions=conditions, cond_drop_prob=0.0, **kwargs
        )

        if cond_scale == 1:
            return logits

        neg_logits = self.forward(
            inputs, conditions=neg_conditions, cond_drop_prob=0.0, **kwargs
        )

        return neg_logits + (logits - neg_logits) * cond_scale

    def _compute_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
        ignore_index=-100,
    ) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:

        B, K, T = targets.shape
        assert logits.shape == targets.shape
        # assert mask.shape == targets.shape
        ce = torch.zeros([], device=targets.device)
        ce_per_codebook: tp.List[torch.Tensor] = []
        for k in range(K):
            logits_k = logits[:, k, ...].contiguous().view(-1)  # [B x T]
            targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
            if mask is not None:
                mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
                targets_k = targets_k[mask_k]
                logits_k = logits_k[mask_k]
            q_ce = F.binary_cross_entropy_with_logits(logits_k, targets_k)
            ce += q_ce
            ce_per_codebook.append(q_ce)
        # average cross entropy across codebooks
        ce = ce / K
        return ce, ce_per_codebook

    def forward(
        self,
        inputs: ConditionTensors,
        conditions: ConditionTensors,
        labels: Optional[torch.IntTensor] = None,
        **kwargs,
    ):
        out = self.net(inputs, conditions, return_embed=True, **kwargs)
        if self.spatial:
            logits = self.to_preds(out.embed)

        else:

            logits = torch.stack(
                [self.to_preds[k](out.embed) for k in range(self.num_codebooks)], dim=1
            )

        if not exists(labels):
            return logits

        logits = rearrange(logits, "... 1 -> ...")

        if self.num_codebooks == 1:

            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(1),
                labels.squeeze(1),
            )

        else:
            loss, ce_per_codebook = self._compute_cross_entropy(
                logits,
                labels,
            )
            return loss, ce_per_codebook
        return loss, None


class MotionMuse(nn.Module):
    def __init__(
        self,
        tranformer_config,
        fuse_config,
        vqvae=None,
        tmr=None,
    ):
        super().__init__()
        # self.vqvae = vqvae.eval().freeze() if exists(vqvae) else None
        self.token_critic = None
        self.tmr_loss_fn = None
        # percentage of tokens to be [mask]ed to remain the same token, so that transformer produces better embeddings across all tokens as done in original BERT paper
        self.no_mask_token_prob = tranformer_config.pop("no_mask_token_prob")
        self.critic_loss_weight = tranformer_config.pop("critic_loss_weight")
        self.align_loss_weight = tranformer_config.pop("align_loss_weight")
        self.flatten = tranformer_config.flatten
        self.spatial = tranformer_config.spatial
        self.flatten_interleave = True
        self.self_token_critic = True if self.critic_loss_weight > 0 else False

        if tmr is not None and vqvae is not None:
            self.tmr_loss_fn = TMRAlignLoss(tmr, vqvae)

        fuse_method = fuse_config.pop("fuse_method")
        if isinstance(fuse_method, list):
            fuse_method = fuse_method[0]
        condition_fuser = ConditionFuser(fuse_method, **fuse_config)

        if (
            len(condition_fuser.fuse2cond["cross"]) > 0
            and tranformer_config.cond_dropout > 0
        ):
            assert (
                tranformer_config.custom == True
            ), "when audio is cross attention, you need custom attention"

        self.model = MLMModel(
            fuser=condition_fuser,
            **tranformer_config,
        )

        if self.self_token_critic:
            self.token_critic = SelfCritic(self.model)

        self.mask_token_id = self.model.mask_token_id
        self.noise_schedule = cosine_schedule
        self.num_codeboks = self.model.num_codebooks

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def muse_mask(self, motion_ids: torch.Tensor, ignore_index: int = -100):
        batch, K, T = motion_ids.shape
        motion_ids = motion_ids.contiguous().view(batch, 1, -1)
        seq_len = K * T
        device = motion_ids.device

        code_ids = motion_ids.clone()

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)

        num_token_masked = (seq_len * 1 * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((batch, 1 * seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask = mask.reshape(batch, 1, seq_len)
        mask[code_ids == self.model.pad_token_id] = False

        labels = torch.where(mask, code_ids, ignore_index)

        if self.no_mask_token_prob > 0.0:
            no_mask_mask = get_mask_subset_prob(mask, self.no_mask_token_prob)
            mask &= ~no_mask_mask

        x = torch.where(mask, self.mask_token_id, code_ids)
        x = x.view(batch, K, T)
        labels = labels.reshape(batch, K, T)

        return x, labels, mask.reshape(batch, K, T)

    def bert_muse_mask(self, motion_ids: torch.Tensor, ignore_index: int = -100):
        batch, K, T = motion_ids.shape
        motion_ids = motion_ids.contiguous().view(batch, 1, -1)
        seq_len = K * T
        device = motion_ids.device

        code_ids = motion_ids.clone()

        rand_time = uniform((batch,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * 1 * rand_mask_probs).round().clamp(min=1)

        # mask_id = self.mask_token_id
        batch_randperm = torch.rand((batch, 1 * seq_len), device=device).argsort(dim=-1)
        mask = batch_randperm < rearrange(num_token_masked, "b -> b 1")
        mask = mask.reshape(batch, 1, seq_len)
        mask[code_ids == self.model.pad_token_id] = False

        # mask_id = self.transformer.mask_token_id
        labels = torch.where(mask, code_ids, ignore_index)
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(code_ids.shape, 0.8)).bool().to(device) & mask
        )
        code_ids[indices_replaced] = self.mask_token_id

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

        code_ids = code_ids.view(batch, K, T)
        labels = labels.reshape(batch, K, T)

        return code_ids, labels, mask.reshape(batch, K, T)

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

    def flatten_input(self, inputs, conditions=None, interleave=False):
        motion, mask = inputs
        b, q, n = motion.shape
        offset = (
            torch.LongTensor([i * self.model.num_tokens // q for i in range(q)])[
                None, :, None
            ]
            .repeat(b, 1, n)
            .to(motion.device)
        )

        motion = motion + offset

        if interleave:
            flat_mask = mask.repeat_interleave(q, dim=1)
            flat_motion = torch.zeros_like(motion).reshape(b, -1)
            inds = [np.arange(i, n * q, q) for i in range(q)]

            for i, ind in enumerate(inds):
                flat_motion[..., ind] = motion[:, i]

        else:
            flat_mask = mask.repeat(1, q)
            flat_motion = motion.reshape(b, -1)

        if conditions is not None:
            cond = self.flatten_cond(conditions["audio"], q, interleave)
            conditions["audio"] = cond
            return (flat_motion[:, None, :], flat_mask), conditions

        return (flat_motion[:, None, :], flat_mask)

    def flatten_cond(self, inputs, q=2, interleave=False):
        cond, mask = inputs
        b, n, d = cond.shape
        cond = cond.permute(0, 2, 1)  ##b d n
        cond = cond[:, :, None].repeat(1, 1, q, 1)
        if interleave:

            flat_mask = mask.repeat_interleave(q, dim=1)
            flat_cond = torch.zeros_like(cond).reshape(b, d, -1)
            inds = [np.arange(i, n * q, q) for i in range(q)]

            for i, ind in enumerate(inds):
                flat_cond[..., ind] = cond[..., i, :]

        else:
            flat_mask = mask.repeat(1, q)
            flat_cond = cond.reshape(b, d, -1)

        flat_cond = flat_cond.permute(0, 2, 1)  ##b nq d

        return (flat_cond, flat_mask)

    def unflatten(self, ids, q=2, interleave=True):
        b, _, n = ids.shape
        l = n // q
        device = ids.device
        unflat = torch.zeros((b, q, l), dtype=torch.long, device=device)
        if interleave:
            inds = [np.arange(i, n, q) for i in range(q)]
        else:
            inds = [
                np.arange(
                    i * l,
                    (i + 1) * l,
                )
                for i in range(q)
            ]

        for i, ind in enumerate(inds):
            unflat[:, i, :] = ids[..., ind].squeeze()

        offset = (
            torch.LongTensor([i * (self.model.num_tokens // q) for i in range(q)])[
                None, :, None
            ]
            .repeat(b, 1, l)
            .to(device)
        )
        unflat = unflat - offset
        for i in range(q):
            unflat[:, i] = unflat[:, i].clamp(min=0, max=self.model.num_tokens // q - 1)

        return unflat

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        conditions: Dict[str, ConditionType],
        neg_conditions: Optional[Dict[str, ConditionType]] = None,
        prime_frames=None,
        duration_s: int = 8,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        cond_scale=3,
        critic_noise_scale=1,
        force_not_use_token_critic=False,
    ):

        has_prime = exists(prime_frames)
        if has_prime:
            prime_token_ids = prime_frames
            prime_token_length = prime_token_ids.shape[-1]

        # begin with all image token ids masked

        # assert self.num_codeboks == 1, "only 1 codebook supported  for now"
        down_sample_ratio = 4
        fps = 30

        device = next(self.parameters()).device
        duration = int(duration_s * (fps / down_sample_ratio))
        timesteps = timesteps * self.num_codeboks

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

            elif conditions.get("audio", None) is not None:
                batch_size = conditions["audio"][0].shape[0]
                conditions["audio"][0] = conditions["audio"][0][:, :seq_len]
                conditions["audio"][1] = conditions["audio"][1][:, :seq_len]
                # seq_len = int(conditions["audio"][0].shape[1] // down_sample_ratio)
                if (
                    neg_conditions is not None
                    and neg_conditions.get("audio", None) is not None
                ):
                    assert (
                        neg_conditions["audio"][0].shape[0] == batch_size
                    ), "negetive audio conditions should have same number as positive "

            else:
                rem_keys = [
                    x for x in list(conditions.keys()) if x not in ["audio", "text"]
                ]
                batch_size = conditions[rem_keys[0]][0].shape[0]
                # seq_len = int(conditions[rem_keys[0]][0].shape[1])
                if (
                    neg_conditions is not None
                    and neg_conditions.get(rem_keys[0], None) is not None
                ):
                    assert (
                        neg_conditions[rem_keys[0]][0].shape[0] == batch_size
                    ), "negetive audio conditions should have same number as positive "

        except:
            print("using null condition")
            batch_size = 1
            conditions = self.get_null_context(batch_size, device)
            cond_scale = 1

        shape = (batch_size, self.num_codeboks, seq_len)

        if self.flatten:
            q = 2
            shape = (batch_size, self.num_codeboks, seq_len * q)
            conditions = make_copy(conditions)
            if conditions["audio"] is not None:
                cond = self.flatten_cond(
                    conditions["audio"], q, self.flatten_interleave
                )
                conditions["audio"] = cond

        ids = torch.full(shape, self.mask_token_id, dtype=torch.long, device=device)
        mask = torch.ones_like(ids)[:, 0, :].to(torch.bool)
        scores = torch.zeros(shape, dtype=torch.float32, device=device)

        starting_temperature = temperature

        demask_fn = self.model.forward_with_cond_scale
        use_token_critic = exists(self.token_critic) and not force_not_use_token_critic

        if use_token_critic:
            token_critic_fn = self.token_critic.forward_with_cond_scale

        # negative prompting, as in paper

        if exists(neg_conditions):

            demask_fn = partial(
                self.model.forward_with_neg_prompt,
                neg_conditions=neg_conditions,
            )

            if use_token_critic:
                token_critic_fn = partial(
                    self.token_critic.forward_with_neg_prompt,
                    neg_conditions=neg_conditions,
                )

        for timestep, steps_until_x0 in tqdm(
            zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps)),
            ),
            total=timesteps,
        ):

            # print(steps_until_x0)
            is_last_step = steps_until_x0 == 0

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len * 1).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices

            ids = ids.scatter(-1, masked_indices, self.mask_token_id)

            # print(ids.shape, mask.shape)
            if has_prime:
                # ids = torch.cat(prime_token_ids , ids)
                ids[..., :prime_token_length] = prime_token_ids
                scores[..., :prime_token_length] = -1e5

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

            if use_token_critic and not is_last_step:
                scores = token_critic_fn(
                    inputs=(ids, mask.squeeze(1)),
                    conditions=conditions,
                    cond_scale=cond_scale,
                )

                scores = rearrange(scores, "... 1 -> ...")

                scores = scores + (
                    uniform(scores.shape, device=device) - 0.5
                ) * critic_noise_scale * (steps_until_x0 / timesteps)
            else:

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

        if self.flatten:
            ids = self.unflatten(ids, q=q, interleave=self.flatten_interleave)

        # if not exists(self.vqvae):
        return ids

        # motion_generated = self.vqvae.decode(ids)
        # return motion_generated

    def forward(
        self,
        inputs: Dict[str, ConditionType],
        conditions: Dict[str, ConditionType],
        ignore_index: int = -100,
        cond_drop_prob=None,
        quality_list=None,
        sample_temperature=None,
        train_critic=True,
    ) -> MuseOutput:

        if self.flatten:
            conditions = make_copy(conditions)
            inputs, conditions = self.flatten_input(
                inputs, conditions, self.flatten_interleave
            )

        motions_ids = inputs[0].to(torch.long)
        input_mask = inputs[1].to(torch.bool)

        B, K, T = motions_ids.shape

        x, labels, muse_mask = self.muse_mask(motions_ids, ignore_index)

        out: MuseOutput = self.model(
            inputs=(x, input_mask.clone()),
            conditions=conditions,
            labels=labels,
            cond_drop_prob=cond_drop_prob,
            ignore_index=ignore_index,
            quality_list=quality_list,
        )

        sampled_ids = gumbel_sample(
            out.logits, temperature=default(sample_temperature, random())
        )

        if self.tmr_loss_fn is not None:
            align_loss = self.tmr_loss_fn((sampled_ids, input_mask), conditions)
            out.align_loss = align_loss
            out.loss + self.align_loss_weight * align_loss

        if not exists(self.token_critic) or train_critic == False:
            return out

        # muse_mask = torch.where(x == self.mask_token_id, 1.0, 0.0).to(torch.bool)

        critic_input = torch.where(muse_mask, sampled_ids, x)
        critic_labels = (x != critic_input).float()

        bce_loss, bce_per_codebook = self.token_critic(
            inputs=(critic_input, input_mask),
            conditions=conditions,
            labels=critic_labels,
            cond_drop_prob=cond_drop_prob,
        )

        out.loss = out.loss + self.critic_loss_weight * bce_loss
        if bce_per_codebook is not None:
            for q in range(self.num_codeboks):
                out.ce_per_codebook[q] += self.critic_loss_weight * bce_per_codebook[q]

        return out


def generate_animation(
    motion_gen: MotionMuse,
    condition_provider,
    duration_s: int = 4,
    aud_file: Optional[str] = None,
    text: Optional[Union[List[str], str]] = None,
    neg_text=None,
    overlap=5,
    timesteps=24,
    use_token_critic=True,
    cond_scale=8,
    temperature=0.8,
):

    _, conditions = condition_provider(
        raw_audio=aud_file, raw_text=text, audio_max_length_s=-1
    )

    neg_conditions = None
    if neg_text is not None:
        _, neg_conditions = condition_provider(
            raw_audio=None, raw_text=text, audio_max_length_s=-1
        )
    if (
        aud_file is not None
        and (conditions["audio"][0].shape[1] // condition_provider.sampling_rate)
        < duration_s
    ):
        duration_s = conditions["audio"][0].shape[1] // condition_provider.sampling_rate

    if isinstance(text, list):
        num_iters = len(text)
    else:
        num_iters = math.ceil((duration_s * 7.5 - 30) / (30 - overlap) + 1)

    all_ids = []
    prime_frames = None
    for i in range(num_iters):
        new_conditions = {}

        if aud_file is not None:

            st = max(0, (30 - overlap) * i - overlap)
            end = st + 120
            new_conditions["audio"] = (
                conditions["audio"][0][:, st:end],
                conditions["audio"][1][:, st:end],
            )

        else:
            new_conditions["audio"] = conditions["audio"]

        if isinstance(text, list):
            new_conditions["text"] = (
                conditions["text"][0][i : i + 1],
                conditions["text"][1][i : i + 1],
            )
        else:
            new_conditions["text"] = conditions["text"]

        gen_ids = motion_gen.generate(
            conditions=new_conditions,
            neg_conditions=neg_conditions,
            prime_frames=prime_frames,
            duration_s=4,
            temperature=temperature,
            timesteps=timesteps,
            cond_scale=cond_scale,
            force_not_use_token_critic=~use_token_critic,
        )
        prime_frames = gen_ids[..., -overlap:]

        if i == 0:
            all_ids.append(gen_ids)
        else:
            all_ids.append(gen_ids[..., overlap:])

    all_ids = torch.cat(all_ids, -1)

    if isinstance(text, list):
        return all_ids
    return all_ids[..., : math.ceil(duration_s * 7.5)]
