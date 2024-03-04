import codecs as cs
import itertools
import json
import os
import random
from glob import glob
from os.path import join as pjoin
from typing import Dict, List, Optional, Tuple

import clip
import librosa
import numpy as np
import torch
import torchaudio
from core import MotionRep
from core.datasets.base_dataset import BaseMotionDataset
from core.datasets.conditioner import ConditionProvider
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

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
    dataset_root,
    dataset_names=dataset_names_default,
    motion_min_length_s=3,
    motion_max_length_s=10,
    audio_rep="encodec",
    motion_rep="full",
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
            MotionAudioTextDataset(
                dataset_name,
                dataset_root=dataset_root,
                split=split,
                motion_min_length_s=motion_min_length_s,
                motion_max_length_s=motion_max_length_s,
                audio_rep=audio_rep,
                motion_rep=motion_rep,
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


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(
            f"Impossible to convert from {channels} to {target_channels}"
        )
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def default(val, d):
    return val if val is not None else d


class MotionAudioTextDataset(BaseMotionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        audio_rep: str = "encodec",
        motion_rep: str = "full",
        hml_rep: str = "gprvc",
        motion_min_length_s=2,
        motion_max_length_s=10,
        window_size=None,
        sampling_rate: int = 16000,
        fps: int = 30,
        split: str = "train",
        **kwargs,
    ):
        super().__init__(dataset_root, MotionRep(motion_rep), hml_rep)
        self.dataset_name = dataset_name
        self.split = split
        self.fps = fps
        self.audio_rep = audio_rep

        self.window_size = window_size

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.motion_dir = os.path.join(data_root, "motion_data/new_joint_vecs")
        self.audio_dir = os.path.join(data_root, "audio")

        if self.audio_rep == "encodec":
            self.sampling_rate = 50
        elif self.audio_rep == "librosa":
            self.sampling_rate = 30
        else:
            self.sampling_rate = sampling_rate

        split_file = os.path.join(data_root, f"motion_data/{split}.txt")

        self.id_list = []
        self.text_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                if dataset_name + "/" in line:
                    try:
                        motion = np.load(os.path.join(self.motion_dir, line.strip()))
                        if motion.shape[0] < default(
                            self.window_size, self.min_motion_length
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

    def load_beat(self, name):
        name = name[:-4]
        id, person_name, recording_type, start, end = name.split("_")
        if id in (list(np.arange(6, 11)) + list(np.arange(21, 31))):
            gender = "woman"
        else:
            gender = "man"

        new_name = f"{name}_0_0"
        name_list = []
        txt_list = []
        with open(
            os.path.join(
                self.text_dir.replace("semantic_labels", "body_texts"), name + ".json"
            ),
            "r",
        ) as outfile:
            frame_texts = json.load(outfile)

        emotion = frame_texts.pop("emotion")
        if emotion == "neutral":
            emotion = "a neutral tone"

        prefix = (
            f"a {gender} is giving a speech with {emotion} on "
            if recording_type == 0
            else f"a {gender} is having a conversation with {emotion} on "
        )

        items = list(frame_texts.values())

        items.insert(0, prefix)
        # sentence = (" ".join(list(dict.fromkeys(items)))).strip()
        name_list.append(new_name)
        txt_list.append(" ".join(items))

        return name_list, txt_list

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
        # print(self.id_list[item])

        name, ind, f_, to_ = self.id_list[item].rsplit("_", 3)
        f_, to_ = int(f_), int(to_)
        motion = np.load(os.path.join(self.motion_dir, name + ".npy"))
        text = self.text_list[item]
        try:

            if self.audio_rep == "wav":

                wav, sr = torchaudio.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".wav")
                )
                audio_data = np.array(convert_audio(wav, sr, self.sampling_rate, 1))
            elif self.audio_rep == "encodec" or self.audio_rep == "librosa":
                audio_data = np.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".npy")
                )

            motion_s = motion.shape[0] // self.fps
            audio_s = audio_data.shape[0] // self.sampling_rate

            common_len_seconds = min(motion_s, audio_s)
            motion = motion[: int(common_len_seconds * self.fps)]
            audio_data = audio_data[: int(common_len_seconds * self.sampling_rate)]

        except:
            audio_data = None

        if to_ * self.fps - f_ * self.fps > self.min_motion_length:
            motion = motion[f_ * self.fps : to_ * self.fps]

        processed_motion = self.get_processed_motion(
            motion, motion_rep=self.motion_rep, hml_rep=self.hml_rep
        )

        return {
            "name": name,
            "motion": processed_motion,
            "text": text,
            "audio": audio_data,
        }


class MotionIndicesAudioTextDataset(data.Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        audio_rep: str = "encodec",
        motion_rep: str = "full",
        # hml_rep: str = "gprvc",
        motion_min_length_s=2,
        motion_max_length_s=10,
        window_size=None,
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

        self.window_size = window_size

        self.min_motion_length = motion_min_length_s * fps
        self.max_motion_length = motion_max_length_s * fps

        data_root = dataset_root

        self.text_dir = os.path.join(data_root, "texts/semantic_labels")
        self.motion_dir = os.path.join(data_root, f"indices/{motion_rep}")
        self.audio_dir = os.path.join(data_root, "audio")

        if self.audio_rep == "encodec":
            self.sampling_rate = 50
        elif self.audio_rep == "librosa":
            self.sampling_rate = 30
        else:
            self.sampling_rate = sampling_rate

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
                        if motion.shape[0] * self.downsample_ratio < default(
                            self.window_size, self.min_motion_length
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

    def load_beat(self, name):
        name = name[:-4]
        id, person_name, recording_type, start, end = name.split("_")
        if id in (list(np.arange(6, 11)) + list(np.arange(21, 31))):
            gender = "woman"
        else:
            gender = "man"

        new_name = f"{name}_0_0"
        name_list = []
        txt_list = []
        with open(
            os.path.join(
                self.text_dir.replace("semantic_labels", "body_texts"), name + ".json"
            ),
            "r",
        ) as outfile:
            frame_texts = json.load(outfile)

        emotion = frame_texts.pop("emotion")
        if emotion == "neutral":
            emotion = "a neutral tone"

        prefix = (
            f"a {gender} is giving a speech with {emotion} on "
            if recording_type == 0
            else f"a {gender} is having a conversation with {emotion} on "
        )

        items = list(frame_texts.values())

        items.insert(0, prefix)
        # sentence = (" ".join(list(dict.fromkeys(items)))).strip()
        name_list.append(new_name)
        txt_list.append(" ".join(items))

        return name_list, txt_list

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, str]:
        # print(self.id_list[item])

        name, ind, f_, to_ = self.id_list[item].rsplit("_", 3)
        f_, to_ = int(f_), int(to_)
        motion = np.load(os.path.join(self.motion_dir, name + ".npy")).squeeze()
        text = self.text_list[item]
        try:

            if self.audio_rep == "wav":

                wav, sr = torchaudio.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".wav")
                )
                audio_data = np.array(convert_audio(wav, sr, self.sampling_rate, 1))
            elif self.audio_rep == "encodec" or self.audio_rep == "librosa":
                audio_data = np.load(
                    os.path.join(self.audio_dir, self.audio_rep, name + ".npy")
                )

            motion_s = (motion.shape[0]) // self.fps
            audio_s = audio_data.shape[0] // self.sampling_rate

            common_len_seconds = min(motion_s, audio_s)
            motion = motion[: int((common_len_seconds * self.fps))]
            audio_data = audio_data[: int(common_len_seconds * self.sampling_rate)]

        except:
            audio_data = None

        if to_ * self.fps - f_ * self.fps > self.min_motion_length:
            motion = motion[(f_ * self.fps) : (to_ * self.fps)]

        return {
            "name": name,
            "motion": motion.reshape(-1, 1),
            "text": text,
            "audio": audio_data,
        }


def simple_collate(
    samples: List[Tuple[torch.Tensor, str]], conditioner: ConditionProvider
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