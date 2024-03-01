import os
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from core.datasets.base_dataset import BaseMotionDataset
from core.models.utils import default
from torch.utils import data
from tqdm import tqdm


def splitHands(hand_data: np.array):
    # lp, rp, lr, rr, lv, rv = torch.split(hand_data, [45, 45, 90, 90, 45, 45], -1)
    lp, rp, lr, rr, lv, rv = np.split(hand_data, [45, 90, 180, 270, 315], -1)
    left_hand = np.concatenate([lp, lr, lv], -1)
    right_hand = np.concatenate([rp, rr, rv], -1)

    return left_hand, right_hand


def simple_collate(samples: List[Tuple[torch.Tensor, str]]) -> Dict[str, torch.Tensor]:
    motion_hand = []
    motion_body = []
    motion_full = []
    names = []
    lens = []
    train_mode = "full"

    for body_params, hand_params, full_params, name, length, mode in samples:
        if mode == "full":
            motion_full.append(torch.Tensor(full_params))
            motion_hand.append(torch.Tensor(hand_params))
            motion_body.append(torch.Tensor(body_params))
        elif "hand" in mode:
            motion_hand.append(torch.Tensor(hand_params))
            motion_body.append(torch.Tensor(body_params))
            motion_full.append(torch.Tensor(full_params))
        elif mode == "body":
            motion_body.append(torch.Tensor(body_params))

        names.append(name)
        lens.append(length)
        train_mode = mode

    batch = {
        "motion_hand": None,
        "motion_body": None,
        "motion_full": None,
        "names": np.array(names),
        "motion_lengths": np.array(lens),
    }
    if "hand" in train_mode or train_mode == "full":
        batch["motion_hand"] = torch.stack(motion_hand, 0)
        left_hand, right_hand = splitHands(np.array(batch["motion_hand"]))

        if train_mode == "left_hand" or train_mode == "full":
            batch["motion_left_hand"] = torch.Tensor(left_hand)
        if train_mode == "right_hand" or train_mode == "full":
            batch["motion_right_hand"] = torch.Tensor(right_hand)

    if "body" in train_mode:
        batch["motion_body"] = torch.stack(motion_body, 0)
    if "full" in train_mode:
        batch["motion_full"] = torch.stack(motion_full, 0)
        batch["motion_body"] = torch.stack(motion_body, 0)
        batch["motion_hand"] = torch.stack(motion_hand, 0)

    return batch


class MotionCollator:
    def __init__(self, mask_id: Optional[int] = None):
        self.mask_id = torch.LongTensor(([mask_id]))

    def __call__(
        self, samples: List[Tuple[torch.Tensor, str]]
    ) -> Dict[str, torch.Tensor]:
        pad_batch_inputs = []
        pad_batch_mask = []
        motion_lengths = []
        names = []
        max_len = max([lens for sample, name, lens in samples])

        for inp, name, lens in samples:
            n, d = inp.shape
            diff = max_len - n
            mask = torch.LongTensor([1] * n + [0] * diff)
            padded = torch.concatenate((torch.tensor(inp), torch.zeros((diff, d))))
            pad_batch_inputs.append(padded)
            pad_batch_mask.append(mask)
            motion_lengths.append(n)
            names.append(name)

        batch = {
            "motion": torch.stack(pad_batch_inputs, 0),
            "motion_lengths": torch.Tensor(motion_lengths),
            "motion_mask": torch.stack(pad_batch_mask, 0),
            "names": np.array(names),
        }

        return batch


class VQSMPLXMotionDataset(BaseMotionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_args,
        model_args,
        fps: int = 30,
        split: str = "train",
    ):
        super().__init__(dataset_args.dataset_root, dataset_args.train_mode)
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps

        data_root = dataset_args.dataset_root
        self.enable_var_len = dataset_args.var_len
        self.train_mode = dataset_args.train_mode

        self.window_size = model_args.window_size
        self.max_motion_length = 900
        # model_args.max_motion_seconds * fps

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.face_text_dir = os.path.join(data_root, "texts/face_texts")

        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")
        self.mean = np.load(os.path.join(data_root, "motion_data/Mean.npy"))
        self.std = np.load(os.path.join(data_root, "motion_data/Std.npy"))
        self.body_mean, self.hand_mean, self.full_mean = self.hmldata_process(self.mean)
        self.body_std, self.hand_std, self.full_std = self.hmldata_process(self.std)

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    try:
                        motion = np.load(os.path.join(self.motion_dir, line.strip()))
                        window_size = (
                            self.window_size
                            if self.window_size != -1
                            else motion.shape[0]
                        )

                        if motion.shape[0] >= window_size:
                            self.id_list.append(line.strip())
                    except:
                        continue

        print(f"Total number of motions {dataset_name}: {len(self.id_list)}")

    def __len__(self) -> int:
        return len(self.id_list)

    def mask_augment(self, motion, perc_n=0.0, perc_d=0.0):
        n, d = motion.shape
        num_masked_n = int(n * perc_n)
        num_masked_d = int(d * perc_d)

        n_ind = list(np.random.choice(np.arange(n), num_masked_n))
        d_ind = list(np.random.choice(np.arange(d), num_masked_d))

        motion[n_ind, :] = 0
        motion[:, d_ind] = 0

        return motion

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        motion = np.load(os.path.join(self.motion_dir, self.id_list[item]))
        prob = random.random()

        if self.enable_var_len and prob < 0.3:
            if self.max_motion_length < 0:
                window_size = motion.shape[0]
            else:
                window_size = np.random.randint(
                    self.window_size, min(motion.shape[0], self.max_motion_length)
                )

        else:
            if self.window_size == -1:
                window_size = (motion).shape[0]

            else:
                window_size = self.window_size

        idx = random.randint(0, (motion).shape[0] - window_size)

        motion = motion[idx : idx + window_size]
        "Z Normalization"

        if self.train_mode == "body":
            motion = (motion - self.mean) / (self.std + 1e-8)
            body_params, hand_params, full_params = self.hmldata_process(motion)
        elif "hand" in self.train_mode or self.train_mode == "full":
            # motion = (motion - self.mean) / (self.std + 1e-8)
            body_params, hand_params, full_params = self.hmldata_process(motion)
            hand_params = self.processHand(
                body_params[None, ...], hand_params, "remove"
            )
            body_params = self.transform(body_params)
            hand_params = self.transform(hand_params)
            full_params = self.transform(full_params)

        return (
            body_params,
            hand_params,
            full_params,
            self.id_list[item],
            window_size,
            self.train_mode,
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
