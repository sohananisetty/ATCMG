import copy
import logging
import os
import random
import typing as tp
import warnings
from typing import Any, Dict, List, Optional, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
from core.models.utils import TorchAutocast
from torch import nn
from transformers import RobertaTokenizer, T5EncoderModel, T5Tokenizer  # type: ignore
from transformers.feature_extraction_sequence_utils import FeatureExtractionMixin
from transformers.feature_extraction_utils import BatchFeature

from core.param_dataclasses import MotionRep, TextRep, AudioRep
from .text_encoders import BERTConditioner, ClipConditioner, T5Conditioner

# from transformers import FeatureExtractionMixin

# logger = logging.get_logger(__name__)
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


def is_batched(raw):

    return bool(
        isinstance(raw, (list, tuple))
        # and (isinstance(raw[0], str) or isinstance(raw[0], np.ndarray))
    )


class ConditionProvider(nn.Module):
    """
    Provides conditioned features for text, audio, and motion.

    Args:
    - text_conditioner_name (str): The name of the text conditioner.
    - audio_conditioner_name (str): The name of the audio conditioner.
    - device (str): The device for computation (e.g., 'cuda' or 'cpu').
    - sampling_rate (int): The sampling rate for audio.
    - audio_max_length_s (int): The maximum length of audio in seconds.
    - audio_padding (str): The type of padding for audio (e.g., 'longest', 'repeat', or 'repeatpad').
    - motion_padding (str): The type of padding for motion (e.g., 'longest', 'repeat', or 'repeatpad').
    - motion_max_length_s (int): The maximum length of motion in seconds.
    - fps (int): The frame rate for motion.
    - motion_rep (str): The representation of motion (e.g., 'full' , 'body' , 'left_hand' , 'right_hand' , 'hand').
    """

    def __init__(
        self,
        text_conditioner_name: str = "t5-base",
        audio_rep: AudioRep = AudioRep.ENCODEC,
        text_rep: TextRep = TextRep.POOLED_TEXT_EMBED,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sampling_rate=16000,
        audio_max_length_s=10,
        audio_padding: str = "longest",
        motion_padding: str = "longest",
        motion_max_length_s: int = 10,
        fps: int = 30,
        motion_rep: MotionRep = MotionRep.FULL,
    ):
        super().__init__()

        self.device = device
        self.audio_padding = audio_padding

        if audio_rep == AudioRep.ENCODEC:
            self.sampling_rate = 50
        elif audio_rep == AudioRep.LIBROSA:
            self.sampling_rate = 30
        else:
            self.sampling_rate = sampling_rate

        self.audio_max_length_s = audio_max_length_s
        self.audio_max_length = audio_max_length_s * self.sampling_rate

        self.fps = fps
        self.motion_rep = motion_rep
        self.text_rep = text_rep
        self.audio_rep = audio_rep

        self.motion_padding = motion_padding
        self.motion_max_length_s = motion_max_length_s
        self.motion_max_length = motion_max_length_s * fps
        self.use_rotation = True

        self.audio_dim = 128 if audio_rep == AudioRep.ENCODEC else 35

        if "t5" in text_conditioner_name:

            self.text_encoder = T5Conditioner(text_conditioner_name, device="cuda")

        elif "bert" in text_conditioner_name:
            self.text_encoder = BERTConditioner(text_conditioner_name, device="cuda")

        else:
            self.text_encoder = ClipConditioner(text_conditioner_name, device="cuda")

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

    def _get_audio_features(
        self,
        max_length: int,
        audio_list: List[np.ndarray],
        padding="longest",
        subset_index_list=None,
    ):
        audios = []
        masks = []
        if padding == "longest":
            max_length_ = max(
                [audio.shape[0] for audio in audio_list if audio is not None],
                default=1,
            )
            max_length = min(max_length_, max_length)

        for idx, audio_feature in enumerate(audio_list):

            if audio_feature is None:
                audios.append(np.zeros((1, max_length, self.audio_dim)))
                masks.append(np.array([0] * max_length)[None, ...])

                continue

            seq_len = audio_feature.shape[0]

            if seq_len > max_length:

                overflow = seq_len - max_length
                start_idx = (
                    subset_index_list[idx]
                    if subset_index_list is not None
                    else np.random.randint(0, overflow + 1)
                )
                audio_feature = audio_feature[start_idx : start_idx + max_length]
                mask = np.array([1] * max_length)
                audios.append(audio_feature[None, ...])
                masks.append(mask[None, ...])

            else:

                if padding == "repeat":
                    n_repeat = int(max_length / seq_len)
                    audio_feature = np.stack(np.tile(audio_feature, n_repeat + 1))[
                        :max_length
                    ]
                elif padding == "repeatpad":
                    n_repeat = int(max_length / seq_len)
                    audio_feature = np.stack(np.tile(audio_feature, n_repeat))

                pad_motion = np.concatenate(
                    [
                        audio_feature,
                        np.zeros(
                            (
                                max_length - audio_feature.shape[0],
                                audio_feature.shape[-1],
                            )
                        ),
                    ]
                )
                mask = np.array(
                    [1] * audio_feature.shape[0]
                    + [0] * (max_length - audio_feature.shape[0])
                )

                audios.append(pad_motion[None, ...])
                masks.append(mask[None, ...])

        padded_audio = np.concatenate(audios, 0)
        attention_mask = np.concatenate(masks, 0)

        return padded_audio, attention_mask

    def _get_motion_features(
        self,
        motion_list: List[np.ndarray],
        max_length: int = None,
        down_sampling_factor=4,
        padding="longest",
        subset_index_list=None,
    ):
        motions = []
        masks = []

        if padding == "longest" or max_length is None:
            max_length_ = (
                max([motion.shape[0] for motion in motion_list]) // down_sampling_factor
            ) * down_sampling_factor
            max_length = min(max_length_, self.motion_max_length)

        for idx, motion in enumerate(motion_list):

            seq_len = motion.shape[0]

            if seq_len > max_length:

                overflow = seq_len - max_length
                start_idx = (
                    subset_index_list[idx]
                    if subset_index_list is not None
                    else np.random.randint(0, overflow + 1)
                )
                motion = motion[start_idx : start_idx + max_length]
                mask = np.array([1] * max_length)
                motions.append(motion[None, ...])
                masks.append(mask[None, ...])

            else:

                motion = motion[
                    : (seq_len // down_sampling_factor) * down_sampling_factor
                ]
                seq_len = (seq_len // down_sampling_factor) * down_sampling_factor
                if padding == "repeat":
                    n_repeat = int(max_length / seq_len)
                    motion = np.stack(np.tile(motion, n_repeat + 1))[:max_length]
                elif padding == "repeatpad":
                    n_repeat = int(max_length / seq_len)
                    motion = np.stack(np.tile(motion, n_repeat))

                pad_motion = np.concatenate(
                    [motion, np.zeros((max_length - motion.shape[0], motion.shape[-1]))]
                )
                mask = np.array(
                    [1] * motion.shape[0] + [0] * (max_length - motion.shape[0])
                )

                motions.append(pad_motion[None, ...])
                masks.append(mask[None, ...])

        padded_motion = np.concatenate(motions, 0)
        attention_mask = np.concatenate(masks, 0)

        return padded_motion, attention_mask

    def _get_text_features(self, raw_text: List[str]):
        tokenized = self.text_encoder.tokenize(raw_text)

        if self.text_rep == TextRep.POOLED_TEXT_EMBED:
            padded_text, text_mask = self.text_encoder.get_text_embedding(tokenized)
        else:
            padded_text, text_mask = self.text_encoder(tokenized)

        return padded_text, text_mask

    def __call__(
        self,
        raw_audio: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        raw_motion: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        raw_text: Optional[Union[str, List[str]]] = None,
        audio_padding: Optional[str] = None,
        motion_padding: Optional[str] = None,
        motion_max_length: Optional[int] = None,
        audio_max_length: Optional[int] = None,
    ) -> tp.Tuple[BatchFeature, BatchFeature]:

        audio_padding = audio_padding if audio_padding else self.audio_padding
        motion_padding = motion_padding if motion_padding else self.motion_padding

        audio_max_length = (
            audio_max_length if audio_max_length is not None else self.audio_max_length
        )
        motion_max_length = (
            motion_max_length
            if motion_max_length is not None
            else self.motion_max_length
        )

        input_features = {}
        condition_features = {}

        ### Assume both have sam e max length

        if raw_audio is None and raw_text is None:
            padded_motion, motion_mask = self._get_motion_features(
                max_length=motion_max_length,
                motion_list=raw_motion,
                padding=motion_padding,
            )
            input_features["motion"] = (
                torch.Tensor(padded_motion),
                torch.BoolTensor(motion_mask),
            )

            return BatchFeature(input_features), BatchFeature(condition_features)

        assert (
            audio_max_length // self.sampling_rate == motion_max_length // self.fps
        ), "need both max length to be same"

        max_length_s = motion_max_length // self.fps

        if not is_batched(raw_audio):
            raw_audio = [raw_audio]
        if not is_batched(raw_motion):
            raw_motion = [raw_motion]
        if not is_batched(raw_text):
            raw_text = [raw_text]

        assert len(raw_audio) == len(
            raw_motion
        ), "mismatch in number of audio and motions"

        subset_idx_motion, subset_idx_audio = None, None

        audio_list = []
        motion_list = []
        motion_idx_list = []
        audio_idx_list = []

        # if audio_present:

        for audio, motion in zip(raw_audio, raw_motion):

            subset_idx_motion = 0
            subset_idx_audio = 0

            if audio is not None:

                subset_idx_audio, subset_idx_motion = self._select_common_start_idx(
                    motion=motion, audio=audio, max_length_s=max_length_s
                )

            audio_list.append(audio)
            motion_list.append(motion)
            motion_idx_list.append(subset_idx_motion)
            audio_idx_list.append(subset_idx_audio)

        padded_audio, audio_mask = self._get_audio_features(
            max_length=audio_max_length,
            audio_list=audio_list,
            padding=audio_padding,
            subset_index_list=audio_idx_list,
        )

        padded_motion, motion_mask = self._get_motion_features(
            max_length=motion_max_length,
            motion_list=motion_list,
            padding=motion_padding,
            subset_index_list=motion_idx_list,
        )

        # tokenized = self.text_encoder.tokenize(raw_text)
        # padded_text, text_mask = self.text_encoder(tokenized)

        padded_text, text_mask = self._get_text_features(raw_text)

        padded_audio, audio_mask = torch.Tensor(padded_audio), torch.BoolTensor(
            audio_mask
        )

        condition_features["audio"] = (
            torch.Tensor(padded_audio).to(self.device),
            torch.BoolTensor(audio_mask).to(self.device),
        )
        input_features["motion"] = (
            torch.Tensor(padded_motion).to(self.device),
            torch.BoolTensor(motion_mask).to(self.device),
        )
        condition_features["text"] = (
            padded_text.to(self.device),
            text_mask.to(dtype=torch.bool, device=self.device),
        )

        return BatchFeature(input_features), BatchFeature(condition_features)


# class T5Conditioner(nn.Module):
#     """T5-based TextConditioner.

#     Args:
#         name (str): Name of the T5 model.
#         output_dim (int): Output dim of the conditioner.
#         finetune (bool): Whether to fine-tune T5 at train time.
#         device (str): Device for T5 Conditioner.
#         autocast_dtype (tp.Optional[str], optional): Autocast dtype.
#         word_dropout (float, optional): Word dropout probability.
#         normalize_text (bool, optional): Whether to apply text normalization.
#     """

#     MODELS = [
#         "t5-small",
#         "t5-base",
#         "t5-large",
#         "t5-3b",
#         "t5-11b",
#         "google/flan-t5-small",
#         "google/flan-t5-base",
#         "google/flan-t5-large",
#         "google/flan-t5-xl",
#         "google/flan-t5-xxl",
#     ]
#     MODELS_DIMS = {
#         "t5-small": 512,
#         "t5-base": 768,
#         "t5-large": 1024,
#         "t5-3b": 1024,
#         "t5-11b": 1024,
#         "google/flan-t5-small": 512,
#         "google/flan-t5-base": 768,
#         "google/flan-t5-large": 1024,
#         "google/flan-t5-3b": 1024,
#         "google/flan-t5-11b": 1024,
#     }

#     def __init__(
#         self,
#         name: str,
#         device: str,
#         autocast_dtype: tp.Optional[str] = "float32",
#         word_dropout: float = 0.0,
#     ):
#         assert (
#             name in self.MODELS
#         ), f"Unrecognized t5 model name (should in {self.MODELS})"
#         super().__init__()
#         self.dim = self.MODELS_DIMS[name]
#         self.device = device
#         self.name = name
#         self.word_dropout = word_dropout
#         if autocast_dtype is None or self.device == "cpu":
#             self.autocast = TorchAutocast(enabled=False)
#             # if self.device != "cpu":
#             #     logger.warning("T5 has no autocast, this might lead to NaN")
#         else:
#             dtype = getattr(torch, autocast_dtype)
#             assert isinstance(dtype, torch.dtype)
#             # logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
#             self.autocast = TorchAutocast(
#                 enabled=True, device_type=self.device, dtype=dtype
#             )
#         # Let's disable logging temporarily because T5 will vomit some errors otherwise.
#         # thanks https://gist.github.com/simon-weber/7853144
#         previous_level = logging.root.manager.disable
#         logging.disable(logging.ERROR)
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             try:
#                 self.t5_tokenizer = T5Tokenizer.from_pretrained(name, cache_dir="./")
#                 t5 = T5EncoderModel.from_pretrained(name, cache_dir="./").eval()
#             finally:
#                 logging.disable(previous_level)

#         # this makes sure that the t5 models is not part
#         # of the saved checkpoint
#         self.__dict__["t5"] = t5.to(device)

#     def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
#         # if current sample doesn't have a certain attribute, replace with empty string'
#         if x is not None and isinstance(x, str):
#             x = [x]
#         entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

#         empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

#         inputs = self.t5_tokenizer(entries, return_tensors="pt", padding=True).to(
#             self.device
#         )
#         # mask = inputs["attention_mask"]
#         inputs["attention_mask"][
#             empty_idx, :
#         ] = 0  # zero-out index where the input is non-existant
#         return inputs

#     def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
#         mask = inputs["attention_mask"]
#         with torch.no_grad(), self.autocast:
#             embeds = self.t5(**inputs).last_hidden_state
#         # embeds = self.output_proj(embeds.to(self.output_proj.weight))
#         embeds = embeds * mask.unsqueeze(-1)
#         return embeds, mask


class ConditionFuser(nn.Module):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["text"],
                "sum": ["bpm"],
                "cross": ["audio"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """

    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate", "cross_seperate"]

    def __init__(
        self,
        fuse2cond: tp.Dict[str, tp.List[str]],
        cross_attention_pos_emb: bool = False,
        cross_attention_pos_emb_scale: float = 1.0,
    ):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def create_sin_embedding(
        self,
        positions: torch.Tensor,
        dim: int,
        max_period: float = 10000,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create sinusoidal positional embedding, with shape `[B, T, C]`.

        Args:
            positions (torch.Tensor): LongTensor of positions.
            dim (int): Dimension of the embedding.
            max_period (float): Maximum period of the cosine/sine functions.
            dtype (torch.dtype or str): dtype to use to generate the embedding.
        Returns:
            torch.Tensor: Sinusoidal positional embedding.
        """
        # We aim for BTC format
        assert dim % 2 == 0
        half_dim = dim // 2
        positions = positions.to(dtype)
        adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(
            1, 1, -1
        )
        max_period_tensor = torch.full(
            [], max_period, device=positions.device, dtype=dtype
        )  # avoid sync point
        phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

    def forward(
        self,
        input_: tp.Dict[str, ConditionType],
        conditions: tp.Dict[str, ConditionType],
    ) -> tp.Tuple[
        ConditionType,
        tp.Optional[
            tp.Union[
                tp.Tuple[torch.Tensor, torch.Tensor],
                tp.Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
            ]
        ],
    ]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """

        input = input_[0]
        input_padding_mask = input_[1]
        B, T, _ = input.shape

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), (
            f"given conditions contain unknown attributes for fuser, "
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        )
        cross_attention_output = None
        cross_attention_mask = None

        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == "sum":
                input += cond
                input_padding_mask = input_padding_mask & cond_mask

            elif op == "input_interpolate":
                cond = einops.rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                cond_mask = F.interpolate(cond_mask, size=input.shape[1])
                input += einops.rearrange(cond, "b d t -> b t d")
                input_padding_mask = input_padding_mask & cond_mask

            elif op == "prepend":
                input = torch.cat([cond, input], dim=1)
                input_padding_mask = torch.cat([cond_mask, input_padding_mask], dim=1)

            elif op == "cross":
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat(
                        [cross_attention_output, cond], dim=1
                    )
                    cross_attention_mask = torch.cat(
                        [cross_attention_mask, cond_mask], dim=1
                    )

            elif op == "cross_seperate":
                if cross_attention_output is not None:
                    cross_attention_output[cond_type] = cond
                    cross_attention_mask[cond_type] = cond_mask
                else:
                    cross_attention_output = {cond_type: cond}
                    cross_attention_mask = {cond_type: cond_mask}
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:

            if isinstance(cross_attention_output, torch.Tensor):
                positions = torch.arange(
                    cross_attention_output.shape[1],
                    device=cross_attention_output.device,
                ).view(1, -1, 1)
                pos_emb = self.create_sin_embedding(
                    positions, cross_attention_output.shape[-1]
                )
                cross_attention_output = (
                    cross_attention_output
                    + self.cross_attention_pos_emb_scale * pos_emb
                )

            else:
                for (
                    cond_type,
                    cross_attention_condition,
                ) in cross_attention_output.items():
                    positions = torch.arange(
                        cross_attention_condition.shape[1],
                        device=cross_attention_condition.device,
                    ).view(1, -1, 1)
                    pos_emb = self.create_sin_embedding(
                        positions, cross_attention_condition.shape[-1]
                    )
                    cross_attention_output[cond_type] = (
                        cross_attention_condition
                        + self.cross_attention_pos_emb_scale * pos_emb
                    )

        return (input, input_padding_mask), (
            cross_attention_output,
            cross_attention_mask,
        )
