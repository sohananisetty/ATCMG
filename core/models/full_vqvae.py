import os
import random

import torch
import torch.nn as nn
from core import (AttentionParams, ConformerParams, PositionalEmbeddingParams,
                  PositionalEmbeddingType, TransformerParams, VQVAEOutput)
from core.models.resnetVQ.encdec import Decoder, Encoder
from core.models.resnetVQ.quantizer import QuantizeEMAReset
from core.models.resnetVQ.residual_vq import ResidualVQ
from core.models.resnetVQ.resnet import Resnet1D
from core.models.resnetVQ.vqvae import HumanVQVAE
from core.models.transformer_modules import (DecoderConformer,
                                             DecoderTransfomer,
                                             EncoderConformer,
                                             EncoderTransfomer)
from core.models.transformer_template import TransformerBlock


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=263,
        nb_code=1024,
        code_dim=512,
        down_sampling_ratio=4,
        transfomer_params=None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code

        self.project_in = (
            nn.Linear(input_dim, transfomer_params.attention_params.dim)
            if transfomer_params.attention_params.dim != input_dim
            else nn.Identity()
        )

        self.encoder = TransformerBlock(
            transfomer_params.attention_params, transfomer_params.depth
        )
        self.decoder = DecoderTransfomer(
            output_dim, down_sampling_ratio, transfomer_params
        )
        self.quantizer = QuantizeEMAReset(
            nb_code, code_dim, code_dim, transfomer_params.attention_params.dim
        )

    def forward(self, x, sample_codebook_temp=0.0):
        # Encode
        x_encoder = self.encoder(x)

        ## quantization
        x_quantized, code_idx, loss, perplexity = self.quantizer(
            x_encoder, return_idx=True, temperature=sample_codebook_temp
        )

        ## decoder
        x_decoder = self.decoder(x_quantized)

        return VQVAEOutput(
            decoded_motion=x_decoder,
            indices=code_idx,
            commit_loss=loss.sum(),
            perplexity=perplexity,
        )

    def encode(self, x, sample_codebook_temp=0.0):
        N, T, d = x.shape
        x_encoder = self.encoder(x)
        x_encoder = self.quantizer.project_in(
            x_encoder.contiguous().view(-1, x_encoder.shape[-1])
        )  # (NT, C)
        code_idx = self.quantizer.quantize(
            x_encoder, sample_codebook_temp=sample_codebook_temp
        )
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward_decoder(self, code_idx):
        x_d = self.quantizer.project_out(self.quantizer.dequantize(code_idx))

        # decoder
        x_decoder = self.decoder(x_d)
        return x_decoder


class HumanRVQVAE(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.args = args
        self.initializeEncoders()

        self.nb_joints = args.nb_joints
        self.sample_codebook_temp = args.sample_codebook_temp
        transfomer_params = TransformerParams(
            attention_params=AttentionParams(
                dim=args.dim, heads=args.heads, dropout=args.dropout
            ),
            depth=args.depth,
        )

        self.rvqvae = VQVAE(
            input_dim=args.codebook_dim_hands + args.codebook_dim_body,
            output_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_sampling_ratio=args.down_sampling_ratio,
            transfomer_params=transfomer_params,
        )

    def initializeEncoders(self):
        self.left_hand_model = HumanVQVAE(self.args.cfg_left.vqvae)
        self.left_hand_model.load(
            os.path.join(self.args.cfg_left.output_dir, "vqvae_motion.pt")
        )

        self.right_hand_model = HumanVQVAE(self.args.cfg_right.vqvae)
        self.right_hand_model.load(
            os.path.join(self.args.cfg_right.output_dir, "vqvae_motion.pt")
        )

        self.body_model = HumanVQVAE(self.args.cfg_body.vqvae)
        self.body_model.load(
            os.path.join(self.args.cfg_body.output_dir, "vqvae_motion.pt")
        )

        self.body_model.freeze()
        self.left_hand_model.freeze()
        self.right_hand_model.freeze()

    def load(self, path):
        pkg = torch.load(path, map_location="cuda")
        self.load_state_dict(pkg["model"])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, motion_encodings):
        # gt_motion_left_hand = batch["motion_left_hand"]
        # gt_motion_right_hand = batch["motion_right_hand"]
        # gt_motion_body = batch["motion_body"]

        return self.rvqvae(
            motion_encodings, sample_codebook_temp=self.sample_codebook_temp
        )
