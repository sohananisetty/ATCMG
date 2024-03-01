import math
import os
import random
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from core.models.dataclasses import MotionTokenizerParams
from core.models.utils import default
from torch.utils import data
from tqdm import tqdm


# Copied from transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [
                spec_aug_mask_idx,
                np.ones(max_num_masked_span - num_masked_span, dtype=np.int32)
                * dummy_mask_idx,
            ]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(
        batch_size, max_num_masked_span * mask_length
    )

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(
        offsets, (batch_size, max_num_masked_span, mask_length)
    ).reshape(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = (
            sequence_length - 1
        )

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask


def get_mlm_mask(
    tokenizer,
    input_ids,
    attention_mask,
    mlm_probability=0.15,
    mask_span=5,
    targets=None,
):

    if input_ids.shape[-1] < int(2.5 * mask_span):
        probability_matrix = torch.full(input_ids.shape, mlm_probability)
        masked_indices = torch.bernoulli(probability_matrix).bool()

    else:

        masked_indices = torch.BoolTensor(
            _compute_mask_indices(
                input_ids.shape,
                mlm_probability,
                mask_span,
                attention_mask=attention_mask,
            )
        )

    vocab_size = tokenizer.vocab_size

    masked_indices[input_ids == tokenizer.pad_token_id] = False
    masked_indices[input_ids == tokenizer.cls_token_id] = False

    if targets is not None:
        targets[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    if targets is not None:
        return input_ids, targets
    else:
        return input_ids


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]],
    tokenizer_params=None,
) -> Dict[str, torch.Tensor]:
    padded_motion = []
    names = []
    lens = []
    attention_masks = []

    if tokenizer_params is None:
        tokenizer_params = MotionTokenizerParams()
    max_len = max([len(sample) for sample, _ in samples])

    for motion_indices, name in samples:
        n = len(motion_indices)
        diff = max_len - n
        padded = torch.concatenate(
            (
                torch.ones(1, dtype=torch.long) * tokenizer_params.cls_token_id,
                torch.LongTensor(motion_indices),
                torch.ones(diff, dtype=torch.long) * tokenizer_params.pad_token_id,
            )
        )

        mask = torch.BoolTensor([0] + [1] * n + [0] * diff)
        padded_motion.append(padded)
        lens.append(n)
        names.append(name)
        attention_masks.append(mask)

        batch = {
            "input_ids": torch.stack(padded_motion, 0),
            "input_lengths": torch.Tensor(lens),
            "names": np.array(names),
            "attention_mask": torch.stack(attention_masks, 0),
        }

        # batch["attention_mask"] = torch.BoolTensor(_compute_mask_indices(batch["input_ids"].shape , mlm_probability, mask_span , attention_mask=batch["attention_mask"] ))
        # targets = batch["input_ids"].clone()
        # targets[~batch["attention_mask"]] = -100

        # batch["input_ids"] , batch["targets"] = mask(tokenizer_params , batch["input_ids"] , batch["attention_mask"], targets )

    return batch


class MotionBERTDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_args,
        model_args,
        fps: int = 30,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps

        data_root = dataset_args.dataset_root
        self.train_mode = dataset_args.train_mode

        # self.window_size = model_args.window_size
        self.max_motion_length = math.ceil(
            (model_args.max_motion_seconds * fps) / model_args.vqvae_downsample
        )
        self.fps = fps

        self.motion_dir = os.path.join(data_root, "indices/body")

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    # self.id_list.append(line.strip())
                    try:
                        motion = np.squeeze(
                            np.load(os.path.join(self.motion_dir, line.strip()))
                        )
                        # window_size = (
                        #     self.window_size
                        #     if self.window_size != -1
                        #     else motion.shape[0]
                        # )

                        if motion.shape[0] >= 10:
                            self.id_list.append(line.strip())
                    except:
                        continue

        print(f"Total number of motions {dataset_name}: {len(self.id_list)}")

    def __len__(self) -> int:
        return len(self.id_list)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        motion = np.squeeze(np.load(os.path.join(self.motion_dir, self.id_list[item])))
        prob = random.random()

        if self.max_motion_length < 0:
            return (
                motion,
                self.id_list[item],
            )
        else:
            if prob < 0.1:
                idx = random.randint(0, (motion).shape[0] - 1)
                motion = motion[idx : idx + 1]
            else:
                idx = random.randint(
                    0, (motion).shape[0] - min(motion.shape[0], self.max_motion_length)
                )
                motion = motion[
                    idx : idx + min(motion.shape[0], self.max_motion_length - 1)
                ]

        return (
            motion,
            self.id_list[item],
        )


def DATALoader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    sampler: torch.utils.data.Sampler = None,
    collate_fn: Optional[
        Callable[[List[Tuple[torch.Tensor, str]]], Dict[str, torch.Tensor]]
    ] = None,
) -> torch.utils.data.DataLoader:
    if collate_fn is None:
        collate_fn = simple_collate

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_loader
