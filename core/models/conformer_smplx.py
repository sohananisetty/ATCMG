from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.models.attend import Attend, Attention
from core.models.conformer import ConformerBlock
from core.models.dataclasses import (
    AttentionParams,
    ConformerParams,
    TransformerParams,
    VQVAEOutput,
)
from core.models.transformer import TransformerBlock
from core.models.utils import exists, pad_to_multiple
from core.quantization.vector_quantize import VectorQuantize
from einops import rearrange
from einops.layers.torch import Rearrange


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, "b (n s) d -> b n (s d)", s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, "b n (s d) -> b (n s) d", s=self.shorten_factor)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim_hands=256,
        input_dim_body=256,
        down_sampling_ratio=4,
        # n_heads=8,
        # dim_hands=512,
        # dim_body=512,
        # depth_hand=3,
        # depth_body=3,
        # conv_expansion_factor=2,
        # dropout=0.2,
        # conv_kernel_size=5,
        conformer_hand_params: ConformerParams = None,
        conformer_body_params: ConformerParams = None,
    ):
        super().__init__()

        self.hand_blocks = []
        self.body_blocks = []
        self.downsample_blocks_hand = []
        self.downsample_blocks = []
        self.downsample_blocks_body = []
        # self.dim_hands = dim_hands
        # self.dim_body = dim_body
        input_dim = input_dim_hands + input_dim_body

        for _ in range(int(np.log2(down_sampling_ratio))):
            self.downsample_blocks.append(
                nn.Conv1d(input_dim, input_dim, 3, stride=2, bias=False, padding=1)
            )
            self.downsample_blocks.append(nn.Dropout(p=0.2))
            self.downsample_blocks.append(nn.GroupNorm(1, input_dim))
            self.downsample_blocks.append(nn.GELU())

        self.hand_conv = nn.Conv1d(
            input_dim_hands, conformer_hand_params.attention_params.dim, 3, 1, 1
        )
        self.body_conv = nn.Conv1d(
            input_dim_body, conformer_body_params.attention_params.dim, 3, 1, 1
        )

        for _ in range(conformer_hand_params.depth):
            block1 = ConformerBlock(
                params=conformer_hand_params
                # dim=dim_hands,
                # dim_head=dim_hands // n_heads,
                # heads=n_heads,
                # conv_expansion_factor=conv_expansion_factor,
                # conv_kernel_size=conv_kernel_size,
                # attn_dropout=dropout,
                # ff_dropout=dropout,
                # conv_dropout=dropout,
            )
            self.hand_blocks.append(block1)

        for _ in range(conformer_body_params.depth):
            block2 = ConformerBlock(
                params=conformer_body_params
                # dim=dim_body,
                # dim_head=dim_body // n_heads,
                # heads=n_heads,
                # conv_expansion_factor=conv_expansion_factor,
                # conv_kernel_size=conv_kernel_size,
                # attn_dropout=dropout,
                # ff_dropout=dropout,
                # conv_dropout=dropout,
            )
            self.body_blocks.append(block2)

        self.hand_model = nn.Sequential(*self.hand_blocks)
        self.body_model = nn.Sequential(*self.body_blocks)
        self.downsample = nn.Sequential(*self.downsample_blocks)

    def forward(self, x_body, x_hand, mask=None, need_transpose=False):
        b, n, d_body = x_body.shape
        x = torch.cat((x_body, x_hand), -1)
        if need_transpose:
            x = rearrange(x, "b n d -> b d n")

        x = self.downsample(x)

        x_body = x[:, :d_body, :]
        x_hand = x[:, d_body:, :]

        x_body = self.body_conv(x_body)
        x_hand = self.hand_conv(x_hand)

        x_body = rearrange(x_body, "b d n -> b n d")
        x_hand = rearrange(x_hand, "b d n -> b n d")

        if mask is not None:
            for l in self.body_blocks:
                if isinstance(l, ConformerBlock):
                    x_body = l(x_body, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x_body = l(x_body.float())

            for l in self.hand_blocks:
                if isinstance(l, ConformerBlock):
                    x_hand = l(x_hand, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x_hand = l(x_hand.float())

            out_body = x_body
            out_hand = x_hand
            return out_hand, out_body, mask

        out_hand = self.hand_model(x_hand)
        out_body = self.body_model(x_body)
        return out_hand, out_body, mask


class EncoderSimple(nn.Module):
    def __init__(
        self,
        input_dim=256,
        down_sampling_ratio=4,
        conformer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.downsample_blocks = []
        self.dim = conformer_params.attention_params.dim
        input_dim = input_dim

        self.conv = nn.Conv1d(input_dim, self.dim, 3, 1, 1)

        for _ in range(int(np.log2(down_sampling_ratio))):
            self.downsample_blocks.append(
                nn.Conv1d(self.dim, self.dim, 3, stride=2, bias=False, padding=1)
            )
            self.downsample_blocks.append(nn.Dropout(p=0.2))
            self.downsample_blocks.append(nn.GroupNorm(self.dim // 8, self.dim))
            self.downsample_blocks.append(nn.GELU())

        for _ in range(conformer_params.depth):
            block1 = ConformerBlock(params=conformer_params)
            self.blocks.append(block1)

        self.model = nn.Sequential(*self.blocks)
        self.downsample = nn.Sequential(*self.downsample_blocks)

    def forward(self, x, mask=None, need_transpose=False):
        if need_transpose:
            x = rearrange(x, "b n d -> b d n")

        x = self.conv(x)

        x = self.downsample(x)

        x = rearrange(x, "b d n -> b n d")

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x
            return out, mask

        out = self.model(x)
        return out, mask


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim=256,
        output_dim=256,
        up_sampling_ratio=4,
        # n_heads=8,
        # dim_hands=512,
        # dim_body=512,
        # dim=768,
        # depth=3,
        # conv_expansion_factor=2,
        # dropout=0.2,
        # conv_kernel_size=5,
        conformer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []

        dim = conformer_params.attention_params.dim

        self.body_hand_fuse = nn.Conv1d(input_dim, dim, 1)

        for _ in range(int(np.log2(up_sampling_ratio)) - 1):
            self.upsample_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.upsample_blocks.append(
                nn.Conv1d(dim, dim, 3, stride=1, bias=False, padding=1)
            )
            self.upsample_blocks.append(nn.Dropout(p=0.2))
            self.upsample_blocks.append(nn.GroupNorm(dim // 8, dim))
            self.upsample_blocks.append(nn.GELU())

        self.upsample_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.upsample_blocks.append(
            nn.Conv1d(dim, output_dim, 3, stride=1, bias=False, padding=1)
        )
        self.upsample_blocks.append(nn.Dropout(p=0.2))
        self.upsample_blocks.append(nn.GELU())

        for _ in range(conformer_params.depth):
            block = ConformerBlock(
                params=conformer_params
                # dim=dim,
                # dim_head=dim // n_heads,
                # heads=n_heads,
                # conv_expansion_factor=conv_expansion_factor,
                # conv_kernel_size=conv_kernel_size,
                # attn_dropout=dropout,
                # ff_dropout=dropout,
                # conv_dropout=dropout,
            )
            self.blocks.append(block)

        self.model = nn.Sequential(*self.blocks)
        self.up_sample = nn.Sequential(*self.upsample_blocks)

    def forward(self, x, mask=None, need_transpose=False):
        # x: b n d

        x = rearrange(x, "b n d -> b d n")
        x = self.body_hand_fuse(x)
        x = rearrange(x, "b d n -> b n d")

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

            if need_transpose:
                out = rearrange(out, "b n d -> b d n")

            out = self.up_sample(out)
            out = rearrange(x, "b d n -> b n d")
            return out, mask

        out = self.model(x)
        if need_transpose:
            out = rearrange(out, "b n d -> b d n")

        out = self.up_sample(out)
        out = rearrange(out, "b d n -> b n d")

        return out


class DecoderSimple(nn.Module):
    def __init__(
        self,
        input_dim=256,
        up_sampling_ratio=4,
        conformer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.dim = conformer_params.attention_params.dim

        for _ in range(int(np.log2(up_sampling_ratio)) - 1):
            self.upsample_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.upsample_blocks.append(
                nn.Conv1d(self.dim, self.dim, 3, stride=1, bias=False, padding=1)
            )
            self.upsample_blocks.append(nn.Dropout(p=0.2))
            self.upsample_blocks.append(nn.GroupNorm(self.dim // 8, self.dim))
            self.upsample_blocks.append(nn.GELU())

        self.upsample_blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.upsample_blocks.append(
            nn.Conv1d(self.dim, input_dim, 3, stride=1, bias=False, padding=1)
        )
        self.upsample_blocks.append(nn.Dropout(p=0.2))
        self.upsample_blocks.append(nn.GELU())

        for _ in range(conformer_params.depth):
            block = ConformerBlock(params=conformer_params)
            self.blocks.append(block)

        self.model = nn.Sequential(*self.blocks)
        self.up_sample = nn.Sequential(*self.upsample_blocks)

    def forward(self, x, mask=None, need_transpose=False):
        # x: b n d

        x = rearrange(x, "b n d -> b d n")
        x = rearrange(x, "b d n -> b n d")

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

            if need_transpose:
                out = rearrange(out, "b n d -> b d n")

            out = self.up_sample(out)
            out = rearrange(x, "b d n -> b n d")
            return out, mask

        out = self.model(x)
        if need_transpose:
            out = rearrange(out, "b n d -> b d n")

        out = self.up_sample(out)
        out = rearrange(out, "b d n -> b n d")

        return out


class DecoderTransfomer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        up_sampling_ratio=4,
        transfomer_params: TransformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.dim = transfomer_params.attention_params.dim

        for _ in range(int(np.log2(up_sampling_ratio))):
            block = nn.Sequential(
                TransformerBlock(
                    transfomer_params.attention_params, transfomer_params.depth
                ),
                LinearUpsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.blocks.append(nn.Linear(self.dim, input_dim))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, TransformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

        else:
            out = self.model(x)

        return out


class EncoderTransfomer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        down_sampling_ratio=4,
        transfomer_params: TransformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.updown = down_sampling_ratio
        self.dim = transfomer_params.attention_params.dim

        self.blocks.append(nn.Linear(input_dim, self.dim))

        for _ in range(int(np.log2(down_sampling_ratio))):
            block = nn.Sequential(
                TransformerBlock(
                    transfomer_params.attention_params, transfomer_params.depth
                ),
                LinearDownsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d
        x = pad_to_multiple(x, self.updown, dim=-2)

        if mask is not None:
            mask = pad_to_multiple(mask, self.updown, dim=-1, value=False)
            for l in self.blocks:
                if isinstance(l, TransformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x
            return out, mask

        else:
            out = self.model(x)

        return out


class DecoderConformer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        up_sampling_ratio=4,
        transfomer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.dim = transfomer_params.attention_params.dim

        for _ in range(int(np.log2(up_sampling_ratio))):
            block = nn.Sequential(
                ConformerBlock(
                    transfomer_params.attention_params, transfomer_params.depth
                ),
                LinearUpsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.blocks.append(nn.Linear(self.dim, input_dim))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d

        if mask is not None:
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x

        else:
            out = self.model(x)

        return out


class EncoderConformer(nn.Module):
    def __init__(
        self,
        input_dim=256,
        down_sampling_ratio=4,
        transfomer_params: ConformerParams = None,
    ):
        super().__init__()

        self.blocks = []
        self.upsample_blocks = []
        self.updown = down_sampling_ratio
        self.dim = transfomer_params.attention_params.dim

        self.blocks.append(nn.Linear(input_dim, self.dim))

        for _ in range(int(np.log2(down_sampling_ratio))):
            block = nn.Sequential(
                ConformerBlock(
                    transfomer_params.attention_params, transfomer_params.depth
                ),
                LinearDownsample(self.dim, 2),
            )
            self.blocks.append(block)

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x, mask=None):
        # x: b n d
        x = pad_to_multiple(x, self.updown, dim=-2)

        if mask is not None:
            mask = pad_to_multiple(mask, self.updown, dim=-1, value=False)
            for l in self.blocks:
                if isinstance(l, ConformerBlock):
                    x = l(x, mask.bool()).float()
                    mask = torch.nn.functional.max_pool1d(
                        mask.float(), 3, stride=2, padding=1
                    )
                else:
                    x = l(x.float())

            out = x
            return out, mask

        else:
            out = self.model(x)

        return out


class ConformerVQMotionModel(nn.Module):
    """Audio Motion VQGAN model."""

    def __init__(self, args):
        """Initializer for VQGANModel.

        Args:
        config: `VQGANModel` instance.
        is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(ConformerVQMotionModel, self).__init__()

        self.input_dim_body = args.motion_dim_body
        self.input_dim_hands = args.motion_dim_hand

        conformer_hand_params = ConformerParams(
            attention_params=AttentionParams(
                dim=args.dim_hands, heads=args.heads, dropout=args.dropout
            ),
            conv_kernel_size=args.conv_kernel_size,
            conv_expansion_factor=2,
            conv_dropout=args.dropout,
            ff_dropout=args.dropout,
            depth=args.depth_hand,
        )

        conformer_body_params = ConformerParams(
            AttentionParams(dim=args.dim_body, heads=args.heads, dropout=args.dropout),
            conv_kernel_size=args.conv_kernel_size,
            conv_expansion_factor=2,
            conv_dropout=args.dropout,
            ff_dropout=args.dropout,
            depth=args.depth_body,
        )
        conformer_decoder_params = ConformerParams(
            AttentionParams(dim=args.dim, heads=args.heads, dropout=args.dropout),
            conv_kernel_size=args.conv_kernel_size,
            conv_expansion_factor=2,
            conv_dropout=args.dropout,
            ff_dropout=args.dropout,
            depth=args.depth,
        )

        self.motionEncoder = Encoder(
            input_dim_hands=args.motion_dim_hand,
            input_dim_body=args.motion_dim_body,
            down_sampling_ratio=args.down_sampling_ratio,
            conformer_body_params=conformer_body_params,
            conformer_hand_params=conformer_hand_params,
        )

        self.motionDecoder = Decoder(
            input_dim=conformer_hand_params.attention_params.dim
            + conformer_body_params.attention_params.dim,
            output_dim=args.motion_dim_body + args.motion_dim_hand,
            up_sampling_ratio=args.down_sampling_ratio,
            conformer_params=conformer_decoder_params,
        )

        self.vq_hands = VectorQuantize(
            dim=conformer_hand_params.attention_params.dim,
            codebook_dim=args.codebook_dim_hands,
            codebook_size=args.codebook_size_hands,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=100,
            threshold_ema_dead_code=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.2,
            affine_param=True,
            sync_update_v=0.2,
            sync_codebook=False,
        )

        self.vq_body = VectorQuantize(
            dim=conformer_body_params.attention_params.dim,
            codebook_dim=args.codebook_dim_body,
            codebook_size=args.codebook_size_body,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=100,
            threshold_ema_dead_code=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.2,
            affine_param=True,
            sync_update_v=0.2,
            sync_codebook=False,
        )

    def load(self, path=None, pkg=None):
        if pkg is None:
            assert path is not None
            pkg = torch.load(str(path), map_location="cuda")
        self.vq_hands._codebook.batch_mean = pkg["model"][
            "vq_hands._codebook.batch_mean"
        ]
        self.vq_hands._codebook.batch_variance = pkg["model"][
            "vq_hands._codebook.batch_variance"
        ]

        self.vq_body._codebook.batch_mean = pkg["model"]["vq_body._codebook.batch_mean"]
        self.vq_body._codebook.batch_variance = pkg["model"][
            "vq_body._codebook.batch_variance"
        ]
        self.load_state_dict(pkg["model"])

    def forward(
        self,
        motion_input_body: torch.Tensor,
        motion_input_hands: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rearrange_output=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict sequences from inputs.

        This is a single forward pass that been used during training.

        Args:
                inputs: Input dict of tensors. The dict should contains
                `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension])

        Returns:
                Final output after the cross modal transformer. A tensor with shape
                [batch_size, motion_seq_length, motion_feature_dimension]
        """
        # Computes motion features.
        # motion_input_body = motion[:, :, : self.input_dim_body]  # b n d
        # motion_input_hands = motion[:, :, : self.input_dim_hands]

        (
            embed_motion_features_hand,
            embed_motion_features_body,
            mask_downsampled,
        ) = self.motionEncoder(
            motion_input_body, motion_input_hands, mask, True
        )  # b n d

        ##codebook
        quantized_enc_hand, indices_hand, commit_loss_hand = self.vq_hands(
            embed_motion_features_hand, mask_downsampled
        )
        quantized_enc_body, indices_body, commit_loss_body = self.vq_body(
            embed_motion_features_body, mask_downsampled
        )

        quantized_enc_motion = torch.cat((quantized_enc_body, quantized_enc_hand), -1)

        indices = torch.cat((indices_body[..., None], indices_hand[..., None]), -1)

        # b n d , b n/4 , q

        ## decoder
        decoded_motion_features = self.motionDecoder(
            quantized_enc_motion, mask, True
        )  # b n d

        if rearrange_output:
            pred_motion_body = decoded_motion_features[..., : self.input_dim_body]
            pred_motion_hand = decoded_motion_features[..., self.input_dim_body :]
            root, pos, rots, vels, foot = torch.split(
                pred_motion_body, [4, 21 * 3, 21 * 6, 21 * 3, 4], -1
            )
            pos2, rots2, vels2 = torch.split(
                pred_motion_hand, [30 * 3, 30 * 6, 30 * 3], -1
            )
            decoded_motion_features = torch.cat(
                [root, pos, pos2, rots, rots2, vels, vels2, foot], -1
            )

        return (
            decoded_motion_features,
            indices,
            commit_loss_body.sum(),
            commit_loss_hand.sum(),
        )

    def encode(self, motion_input, mask=None):
        with torch.no_grad():
            motion_input_body = motion_input[:, :, : self.input_dim_body]  # b n d
            motion_input_hands = motion_input[:, :, : self.input_dim_hands]

            (
                embed_motion_features_hand,
                embed_motion_features_body,
                mask_downsampled,
            ) = self.motionEncoder(
                motion_input_body, motion_input_hands, mask, True
            )  # b n d

            ##codebook
            quantized_enc_hand, indices_hand, commit_loss_hand = self.vq_hands(
                embed_motion_features_hand, mask_downsampled
            )
            quantized_enc_body, indices_body, commit_loss_body = self.vq_body(
                embed_motion_features_body, mask_downsampled
            )

            quantized_enc_motion = torch.cat(
                (quantized_enc_body, quantized_enc_hand), -1
            )
            indices = torch.cat((indices_body[..., None], indices_hand[..., None]), -1)
            return indices

    def decode(self, indices, mask=None):
        b, n, bh = indices.shape
        indices_body = indices[..., 0]
        indices_hand = indices[..., 1]
        with torch.no_grad():
            quantized_enc_hand = self.vq_hands.get_codes_from_indices(
                indices_hand
            ).reshape(b, n, -1)
            quantized_enc_body = self.vq_body.get_codes_from_indices(
                indices_body
            ).reshape(b, n, -1)
            quantized_enc_motion = torch.cat(
                (quantized_enc_body, quantized_enc_hand), -1
            )
            decoded_motion_features = self.motionDecoder(
                quantized_enc_motion, mask, True
            )  # b n d
            return quantized_enc_motion, decoded_motion_features


class ConformerVQMotionModelSimple(nn.Module):
    """Audio Motion VQGAN model."""

    def __init__(self, args):
        """Initializer for VQGANModel.

        Args:
        config: `VQGANModel` instance.
        is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(ConformerVQMotionModelSimple, self).__init__()

        self.dim = args.dim
        self.input_dim = args.motion_dim

        conformer_params = ConformerParams(
            attention_params=AttentionParams(
                dim=args.dim, heads=args.heads, dropout=args.dropout
            ),
            conv_kernel_size=args.conv_kernel_size,
            conv_expansion_factor=2,
            conv_dropout=args.dropout,
            ff_dropout=args.dropout,
            depth=args.depth,
        )

        self.motionEncoder = EncoderSimple(
            input_dim=args.motion_dim,
            down_sampling_ratio=args.down_sampling_ratio,
            conformer_params=conformer_params,
        )

        self.motionDecoder = DecoderSimple(
            input_dim=args.motion_dim,
            up_sampling_ratio=args.down_sampling_ratio,
            conformer_params=conformer_params,
        )

        self.vq = VectorQuantize(
            dim=args.dim,
            codebook_dim=args.codebook_dim,
            codebook_size=args.codebook_size,  # codebook size
            kmeans_init=True,  # set to True
            kmeans_iters=100,
            threshold_ema_dead_code=10,
            stochastic_sample_codes=True,
            sample_codebook_temp=0.2,
            affine_param=True,
            sync_update_v=0.2,
            sync_codebook=False,
        )

    def load(self, path):
        pkg = torch.load(str(path), map_location="cuda")
        self.vq._codebook.batch_mean = pkg["model"]["vq._codebook.batch_mean"]
        self.vq._codebook.batch_variance = pkg["model"]["vq._codebook.batch_variance"]
        self.load_state_dict(pkg["model"])

    def forward(
        self, motion: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> VQVAEOutput:
        """Predict sequences from inputs.

        This is a single forward pass that been used during training.

        Args:
                inputs: Input dict of tensors. The dict should contains
                `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension])

        Returns:
                Final output after the cross modal transformer. A tensor with shape
                [batch_size, motion_seq_length, motion_feature_dimension]
        """
        # Computes motion features.
        motion_input = motion  # b n d

        embed_motion_features, mask_downsampled = self.motionEncoder(
            motion_input, mask, True
        )  # b n d

        ##codebook
        quantized_enc_motion, indices, commit_loss = self.vq(
            embed_motion_features, mask_downsampled
        )

        # b n d , b n/4 , q

        ## decoder
        decoded_motion_features = self.motionDecoder(
            quantized_enc_motion, mask, True
        )  # b n d

        # print(commit_loss.shape)
        # commit_loss = torch.Tensor([1]).to("cuda")

        return VQVAEOutput(
            decoded_motion=decoded_motion_features,
            indices=indices,
            commit_loss=commit_loss.sum(),
        )

    def encode(self, motion_input, mask=None):
        with torch.no_grad():
            embed_motion_features, mask_downsampled = self.motionEncoder(
                motion_input, mask, True
            )  # b n d

            ##codebook
            quantized_enc_motion, indices, commit_loss = self.vq(
                embed_motion_features, mask_downsampled
            )
            return indices

    def decode(self, indices, mask=None):
        with torch.no_grad():
            quantized = self.vq.get_codes_from_indices(indices).reshape(
                indices.shape[0], -1, self.dim
            )
            decoded_motion_features = self.motionDecoder(quantized, mask, True)  # b n d
            return quantized, decoded_motion_features
