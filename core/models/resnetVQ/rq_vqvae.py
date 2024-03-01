import os
import random

import torch
import torch.nn as nn
from core import VQVAEOutput
from core.models.resnetVQ.encdec import Decoder, Encoder
from core.models.resnetVQ.quantizer import QuantizeEMAReset
from core.models.resnetVQ.residual_vq import ResidualVQ
from core.models.resnetVQ.resnet import Resnet1D
from core.models.resnetVQ.vqvae import HumanVQVAE


class RVQVAE(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        output_dim=263,
        nb_code=1024,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        num_quantizers=2,
        quantize_dropout_prob=0.2,
        activation="relu",
        shared_codebook=False,
        norm=None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code

        # if input_dim == code_dim:
        #     self.encoder = nn.Identity()
        # else:
        #     self.encoder = nn.Sequential(
        #         nn.Conv1d(input_dim, code_dim, 3, 1, 1),
        #         # Resnet1D(code_dim, 3, dilation_growth_rate),
        #     )

        Encoder(
            input_dim,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        # nn.Linear(input_dim, code_dim)
        self.decoder = Decoder(
            output_dim,
            code_dim,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )
        rvqvae_config = {
            "num_quantizers": num_quantizers,
            "shared_codebook": shared_codebook,
            "quantize_dropout_prob": quantize_dropout_prob,
            "quantize_dropout_cutoff_index": 0,
            "nb_code": nb_code,
            "code_dim": code_dim,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        quantized_out, code_idx, all_codes = self.quantizer.quantize(
            x_encoder, return_latent=True
        )
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return quantized_out, code_idx, all_codes

    def forward(self, x, sample_codebook_temp=0.5):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)

        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(
            x_encoder, sample_codebook_temp=sample_codebook_temp
        )

        # print(code_idx[0, :, 1])
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return VQVAEOutput(
            decoded_motion=x_out,
            indices=code_idx,
            commit_loss=commit_loss.sum(),
            perplexity=perplexity,
        )

        # return x_out, commit_loss, perplexity

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_decoder = self.decoder(x)
        x_out = self.postprocess(x_decoder)
        return x_out


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

        self.rvqvae = RVQVAE(
            input_dim=args.codebook_dim_hands + args.codebook_dim_body,
            output_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_t=args.down_sampling_ratio // 2,
            stride_t=args.down_sampling_ratio // 2,
            width=args.dim,
            depth=args.depth,
            dilation_growth_rate=3,
            num_quantizers=args.num_quantizers,
            quantize_dropout_prob=args.quantize_dropout_prob,
            activation="relu",
            shared_codebook=args.shared_codebook,
            norm=None,
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

    # def encode(self, batch):
    #     gt_motion_left_hand = batch["motion_left_hand"]
    #     gt_motion_right_hand = batch["motion_right_hand"]
    #     gt_motion_body = batch["motion_body"]

    #     motion_encodings = self.getEncodings(
    #         gt_motion_body, gt_motion_left_hand, gt_motion_right_hand
    #     )
    #     # b, t, c = motion.size()
    #     quantized_out, code_idx, all_codes = self.rvqvae.encode(
    #         motion_encodings, sample_codebook_temp=self.sample_codebook_temp
    #     )  # (N, T)
    #     return quantized_out, code_idx, all_codes

    # def getEncodings(self, gt_motion_body, gt_motion_left_hand, gt_motion_right_hand):
    #     indb = self.body_model.encode(gt_motion_body)
    #     indhr = self.right_hand_model.encode(gt_motion_right_hand)
    #     indhl = self.left_hand_model.encode(gt_motion_left_hand)
    #     enc_b = self.body_model.vqvae.quantizer.get_codebook_entry(indb)
    #     enc_hr = self.right_hand_model.vqvae.quantizer.get_codebook_entry(indhr)
    #     enc_hl = self.left_hand_model.vqvae.quantizer.get_codebook_entry(indhl)
    #     enc = torch.cat([enc_b, enc_hr, enc_hl], -1)
    #     return enc

    # def getFinalMotions(
    #     self, gt_motion_body, gt_motion_left_hand, gt_motion_right_hand
    # ):
    #     body_out = self.body_model(gt_motion_body)
    #     right_out = self.right_hand_model(gt_motion_right_hand)
    #     left_out = self.left_hand_model(gt_motion_left_hand)
    #     # enc_b = self.body_model.vqvae.quantizer.get_codebook_entry(indb)
    #     # enc_hr = self.right_hand_model.vqvae.quantizer.get_codebook_entry(indhr)
    #     # enc_hl = self.left_hand_model.vqvae.quantizer.get_codebook_entry(indhl)
    #     # enc = torch.cat([enc_b, enc_hr, enc_hl], -1)
    #     return enc

    def forward(self, batch):
        gt_motion_left_hand = batch["motion_left_hand"]
        gt_motion_right_hand = batch["motion_right_hand"]
        gt_motion_body = batch["motion_body"]

        motion_encodings = self.getEncodings(
            gt_motion_body, gt_motion_left_hand, gt_motion_right_hand
        )

        return self.rvqvae(
            motion_encodings, sample_codebook_temp=self.sample_codebook_temp
        )

    # def forward_decoder(self, indices):
    #     # indices shape 'b n q'
    #     x_out = self.rvqvae.forward_decoder(indices)
    #     return x_out


class LengthEstimator(nn.Module):
    def __init__(self, input_size, output_size):
        super(LengthEstimator, self).__init__()
        nd = 512
        self.output = nn.Sequential(
            nn.Linear(input_size, nd),
            nn.LayerNorm(nd),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd, nd // 2),
            nn.LayerNorm(nd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(nd // 2, nd // 4),
            nn.LayerNorm(nd // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nd // 4, output_size),
        )

        self.output.apply(self.__init_weights)

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_emb):
        return self.output(text_emb)
