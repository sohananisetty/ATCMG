import codecs as cs
import itertools
import json
import math
import os
import random
from glob import glob
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import clip
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.datasets.conditioner import ConditionProvider
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from utils.motion_processing.quaternion import qinv, qrot, quaternion_to_cont6d

genre_dict = {
    "mBR": "Break",
    "mPO": "Pop",
    "mLO": "Lock",
    "mMH": "Middle_Hip-hop",
    "mLH": "LAstyle Hip-hop",
    "mHO": "House",
    "mWA": "Waack",
    "mKR": "Krump",
    "mJS": "Street_Jazz",
    "mJB": "Ballet_Jazz",
}

dataset_names_default = [
    "animation",
    "humanml",
    "perform",
    "GRAB",
    "idea400",
    "humman",
    "beat",
    "game_motion",
    "music",
    "aist",
    "fitness",
    "moyo",
    "choreomaster",
    "dance",
    "kungfu",
    "EgoBody",
    "HAA500",
]


def load_dataset(
    dataset_args,
    # dataset_root,
    dataset_names=dataset_names_default,
    # motion_min_length_s=3,
    # motion_max_length_s=10,
    # audio_rep="encodec",
    # motion_rep="full",
    split: str = "train",
    weight_scale: Optional[List[int]] = None,
):
    if weight_scale is None:
        weight_scale = [1] * len(dataset_names)
    assert len(dataset_names) == len(weight_scale), "mismatch in size"
    dataset_list = []
    weights = []
    for dataset_name in dataset_names:
        dataset_list.append(
            TranslationAudioTextDataset(
                dataset_name,
                dataset_root=dataset_args.dataset_root,
                split=split,
                motion_min_length_s=dataset_args.motion_min_length_s,
                motion_max_length_s=dataset_args.motion_max_length_s,
                audio_rep=dataset_args.audio_rep,
                motion_rep=dataset_args.motion_rep,
                hml_rep=dataset_args.hml_rep,
            )
        )

    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)

    if split != "train" or len(dataset_names) == 1:
        return concat_dataset, None, None

    for i, ds in enumerate(dataset_list):
        weights.append(
            [weight_scale[i] * concat_dataset.__len__() / (ds.__len__())] * ds.__len__()
        )

    weights = list(itertools.chain.from_iterable(weights))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights)
    )

    return concat_dataset, sampler, weights


def default(val, d):
    return val if val is not None else d


def recover_root_rot_pos(data):
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


class TranslationAudioTextDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        audio_rep: str = "encodec",
        motion_rep: str = "full",
        # hml_rep: str = "gprvc",
        motion_min_length_s=3,
        motion_max_length_s=10,
        window_size_s=None,
        sampling_rate: int = 16000,
        downsample_ratio=4,
        fps: int = 30,
        split: str = "train",
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps
        self.audio_rep = audio_rep
        self.downsample_ratio = downsample_ratio
        self.motion_rep = motion_rep

        self.window_size = (
            max(int(window_size_s * self.fps, -1))
            if window_size_s is not None
            else None
        )

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")

        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")
        self.audio_dir = os.path.join(data_root, "audio")

        if self.audio_rep in ["encodec", "librosa"]:
            self.sampling_rate = 30
        elif self.audio_rep == "clap":
            self.sampling_rate = 48000
        else:
            self.sampling_rate = int(sampling_rate)

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        self.text_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    try:
                        motion = np.load(
                            os.path.join(self.motion_dir, line.strip())
                        ).squeeze()
                        seq_len = motion.shape[0]

                        if seq_len < round(
                            default(self.window_size, self.min_motion_length)
                        ):
                            continue

                        if self.dataset_name == "humanml":
                            name_list, txt_list = self.load_humanml(line.strip())

                        else:
                            name_list, txt_list = self.load_txt(line.strip())

                        self.id_list.extend(name_list)
                        self.text_list.extend(txt_list)

                    except:
                        continue

        print(
            f"Total number of motions {dataset_name}: {len(self.id_list)} and texts {len(self.text_list)}"
        )

    def __len__(self) -> int:
        return len(self.id_list)

    def load_txt(self, name):
        name = name[:-4]
        new_name = f"{name}_0_0_0"
        name_list = []
        txt_list = []

        with open(os.path.join(self.text_dir, name + ".txt"), "r") as f:
            for line in f.readlines():
                name_list.append(new_name)
                txt_list.append(line.strip())

        return name_list, txt_list

    def load_humanml(self, name):
        name = name[:-4]
        # data_dict = {}
        name_list = []
        txt_list = []
        with open(os.path.join(self.text_dir, name + ".txt"), "r") as f:
            for index, line in enumerate(f.readlines()):
                line_split = line.strip().split("#")
                caption = line_split[0]
                tokens = line_split[1].split(" ")
                f_tag = float(line_split[2])
                to_tag = float(line_split[3])
                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                to_tag = 0.0 if np.isnan(to_tag) else to_tag
                new_name = (
                    f"{name}_{index}_{int(f_tag * self.fps)}_{int(to_tag * self.fps)}"
                )

                name_list.append(new_name)
                txt_list.append(caption)

        return name_list, txt_list

    def _select_common_start_idx(self, motion, audio, max_length_s):
        motion_s = motion.shape[0] // self.fps
        audio_s = audio.shape[0] // self.sampling_rate

        common_len_seconds = min(motion_s, audio_s)
        motion = motion[: int(common_len_seconds * self.fps)]
        audio = motion[: int(common_len_seconds * self.sampling_rate)]

        if common_len_seconds > max_length_s:
            subset_idx_motion = np.random.randint(
                0, motion.shape[0] - int(max_length_s * self.fps) + 1
            )

            mot_start_s = subset_idx_motion // self.fps
            subset_idx_audio = int(mot_start_s * self.sampling_rate)

        else:
            return 0, 0

        return subset_idx_audio, subset_idx_motion

    def get_windowed_data(self, audio_data, motion):
        if self.window_size == -1:
            mot_len_s = int(motion.shape[0] // self.fps)
            audio_len_s = mot_len_s

        else:
            mot_len_s = int(self.window_size // self.fps)
            audio_len_s = mot_len_s

        if audio_data is None:

            subset_idx_motion = random.randint(
                0, max(0, motion.shape[0] - int(mot_len_s * self.fps))
            )

        else:
            subset_idx_audio, subset_idx_motion = self._select_common_start_idx(
                motion, audio_data, mot_len_s
            )

            audio_data = audio_data[
                subset_idx_audio : subset_idx_audio
                + int(audio_len_s * self.sampling_rate)
            ]

        motion = motion[
            subset_idx_motion : subset_idx_motion + int(mot_len_s * self.fps)
        ]

        return audio_data, motion

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:

        name, ind, f_, to_ = self.id_list[item].rsplit("_", 3)
        f_, to_ = int(f_), int(to_)
        motion = (np.load(os.path.join(self.motion_dir, name + ".npy")).squeeze())[
            :, :4
        ]

        seq_len = (motion.shape[0] // self.downsample_ratio) * self.downsample_ratio
        motion = motion[:seq_len]

        text = self.text_list[item]
        try:

            if self.audio_rep in ["wav", "clap"]:

                audio_data, _ = librosa.load(
                    os.path.join(self.audio_dir, "wav", name + ".wav"),
                    sr=self.sampling_rate,
                )  # sample rate should be 48000
                audio_data = audio_data.reshape(-1, 1)  ## T 1
            elif self.audio_rep in ["encodec", "librosa"]:
                audio_data = np.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".npy")
                )

            motion_s = (motion.shape[0]) // self.fps
            audio_s = audio_data.shape[0] // self.sampling_rate

            common_len_seconds = min(motion_s, audio_s)
            motion = motion[: int((common_len_seconds * self.fps))]
            if self.motion_rep == "full":
                left_hand_motion = left_hand_motion[
                    : int((common_len_seconds * self.fps))
                ]
                right_hand_motion = right_hand_motion[
                    : int((common_len_seconds * self.fps))
                ]

            audio_data = audio_data[: int(common_len_seconds * self.sampling_rate)]

        except:
            audio_data = None

        if motion[int(f_) : math.ceil(to_)].shape[0] > default(
            self.window_size, self.min_motion_length
        ):
            motion = motion[int(f_) : math.ceil(to_)]

        if self.window_size is not None:

            audio_data, motion = self.get_windowed_data(audio_data, motion)

        final_motion = motion  ## n 1/3

        return {
            "name": name,
            "motion": final_motion,
            "text": text,
            "audio": audio_data,
        }


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]],
    conditioner: ConditionProvider,
) -> Dict[str, torch.Tensor]:
    motions = []
    texts = []
    audios = []
    names = []

    for sample in samples:
        names.append(sample["name"])
        motions.append(sample["motion"])
        texts.append(sample["text"])
        audios.append(sample["audio"])

    inputs, conditions = conditioner(
        raw_audio=audios,
        raw_motion=motions,
        raw_text=texts,
    )

    inputs["names"] = np.array(names)
    inputs["texts"] = np.array(texts)

    return inputs, conditions
