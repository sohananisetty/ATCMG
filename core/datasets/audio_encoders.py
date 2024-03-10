import typing as tp
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio

# from encodec import EncodecModel
# from encodec.utils import convert_audio, save_audio
from transformers import AutoProcessor, EncodecModel
from transformers.models.encodec.modeling_encodec import EncodecDecoderOutput
import librosa


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


class AudioConditionerEncodec(nn.Module):
    def __init__(self, target_bandwidth=6, target_sr=16000, device="cuda"):
        self.encoder = (
            EncodecModel.from_pretrained("facebook/encodec_24khz", cache_dir="./")
            .to(device)
            .to(eval)
        )
        self.processor = AutoProcessor.from_pretrained(
            "facebook/encodec_24khz", cache_dir="./"
        )
        self.target_sr = target_sr
        self.device = device
        self.target_bandwidth = target_bandwidth
        self.freeze()

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        def _decode_frame(
            codes: torch.Tensor, scale: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            codes = codes.transpose(0, 1)
            embeddings = self.encoder.quantizer.decode(codes)
            # if scale is not None:
            #     embeddings = embeddings * scale.view(-1, 1, 1)

            return embeddings

        chunk_length = self.encoder.config.chunk_length
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            quantized_values = _decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []
            for codes, scale in zip(audio_codes, audio_scales):

                frames = _decode_frame(frame, scale)
                decoded_frames.append(outputs)

            quantized_values = self.encoder._linear_overlap_add(
                decoded_frames, self.encoder.config.chunk_stride or 1
            )

        # truncate based on padding mask
        if padding_mask is not None:
            scale_mask = padding_mask.shape[-1] // quantized_values.shape[-1]
            padding_mask = torch.nn.functional.interpolate(
                padding_mask[:, None, :].float(), scale_factor=(1 / scale_mask)
            ).to(torch.bool)
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        return quantized_values

    def get_audio_embedding(
        self,
        path_or_wav: tp.Union[str, tp.List[str], torch.Tensor, tp.List[torch.Tensor]],
    ) -> torch.Tensor:
        if not isinstance(path_or_wav, list):
            path_or_wav = [path_or_wav]

        if not isinstance(path_or_wav[0], torch.Tensor):
            wavs = []
            for pth in path_or_wav:
                wav, sr = torchaudio.load(path_or_wav)
                audio_sample = convert_audio(
                    audio_sample, sr, self.target_sr, 1
                )  ## channel time
                wavs.append(audio_sample[0].numpy())
            inputs = self.processor(
                raw_audio=wavs,
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt",
            )  ## B chn T

        else:
            wavs = []
            for wav in path_or_wav:
                assert len(wav.shape) == 2, "should be chn T"
                wavs.append(np.array(wav[0]))
            inputs = self.processor(
                raw_audio=wavs,
                sampling_rate=self.processor.sampling_rate,
                return_tensors="pt",
            )  ## B chn T

        with torch.no_grad():
            encoder_outputs = self.encoder.encode(
                inputs["input_values"].to(self.device),
                inputs["padding_mask"].to(self.device),
                bandwidth=self.target_bandwidth,
            )
            embs = self._decode(
                encoder_outputs.audio_codes, encoder_outputs.audio_scales
            )

        return embs.permute(0, 2, 1)  ## B N d


class AudioConditionerLibrosa(nn.Module):
    def __init__(self, fps=30, device="cuda"):

        self.device = device
        self.fps = fps
        self.HOP_LENGTH = 512
        self.SR = self.fps * self.HOP_LENGTH
        self.EPS = 1e-6

    def get_audio_feat(self, data: np.ndarray):
        envelope = librosa.onset.onset_strength(data, sr=self.SR)  # (seq_len,)
        mfcc = librosa.feature.mfcc(data, sr=self.SR, n_mfcc=20).T  # (seq_len, 20)
        chroma = librosa.feature.chroma_cens(
            data, sr=self.SR, hop_length=self.HOP_LENGTH, n_chroma=12
        ).T  # (seq_len, 12)

        peak_idxs = librosa.onset.onset_detect(
            onset_envelope=envelope.flatten(), sr=self.SR, hop_length=self.HOP_LENGTH
        )
        peak_onehot = np.zeros_like(envelope, dtype=np.float32)
        peak_onehot[peak_idxs] = 1.0  # (seq_len,)

        tempo, beat_idxs = librosa.beat.beat_track(
            onset_envelope=envelope,
            sr=self.SR,
            hop_length=self.HOP_LENGTH,
            tightness=100,
        )
        beat_onehot = np.zeros_like(envelope, dtype=np.float32)
        beat_onehot[beat_idxs] = 1.0  # (seq_len,)

        audio_feature = np.concatenate(
            [
                envelope[:, None],
                mfcc,
                chroma,
                peak_onehot[:, None],
                beat_onehot[:, None],
            ],
            axis=-1,
        )
        # audio_feat[audio_name] = audio_feature

        return audio_feature

    def get_audio_embedding(
        self,
        path_or_wav: tp.Union[str, tp.List[str], torch.Tensor, tp.List[torch.Tensor]],
    ):
        if not isinstance(path_or_wav, list):
            path_or_wav = [path_or_wav]

        audio_features = []
        if not isinstance(path_or_wav[0], torch.Tensor):
            for path in path_or_wav:
                data, _ = librosa.load(path, sr=self.SR)
                audio_features.append(self.get_audio_feat(data))

        else:
            for wav in path_or_wav:
                assert len(wav.shape) == 1, "should be T"
                audio_features.append(self.get_audio_feat(data))

        embs = np.stack(audio_features, 0)

        return embs  ## B N d


# class AudioConditionerEncodec(nn.Module):
#     def __init__(self, target_bandwidth=6, target_sr=16000, device="cuda"):
#         self.model = EncodecModel.encodec_model_24khz().to(device).eval()
#         self.model.set_target_bandwidth(target_bandwidth)
#         self.target_sr = target_sr
#         self.device = device

#     def get_audio_embedding(path_or_wav: tp.Union[str, torch.Tensor]):
#         if isinstance(path_or_wav, str):
#             wav, sr = torchaudio.load(path_or_wav)
#         wav = convert_audio(wav, sr, self.target_sr, self.model.channels)
#         wav = wav.unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             encoded_frames = self.model.encode(wav)
#         embs = self.model.decode2emb(encoded_frames)

#         return embs[0].permute(1, 0)  ## N d
