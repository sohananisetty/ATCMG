import torch
import torch.nn as nn
from core.models.dataclasses import VQVAEOutput
from core.models.resnetVQ.encdec import Decoder, Encoder
from core.models.resnetVQ.quantizer import QuantizeEMAReset


class VQVAE(nn.Module):
    def __init__(
        self,
        input_dim=263,
        nb_code=1024,
        code_dim=512,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.encoder = Encoder(
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
        self.decoder = Decoder(
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
        self.quantizer = QuantizeEMAReset(nb_code, code_dim, code_dim)

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
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def forward(self, x, mask=None, temperature=0.0):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        if mask is not None:

            downsampled_motion_mask = torch.nn.functional.max_pool1d(
                mask.float(),
                1,
                stride=4,
            )
            x_encoder = x_encoder * downsampled_motion_mask[:, None, :]

        ## quantization
        x_quantized, code_idx, loss, perplexity = self.quantizer(
            x_encoder, return_idx=True, temperature=temperature
        )

        if mask is not None:

            x_quantized = x_quantized * downsampled_motion_mask[:, None, :]

        ## decoder
        x_decoder = self.decoder(x_quantized)

        if mask is not None:

            x_decoder = x_decoder * mask[:, None, :]

        x_out = self.postprocess(x_decoder)
        return VQVAEOutput(
            decoded_motion=x_out,
            indices=code_idx,
            commit_loss=loss.sum(),
            perplexity=perplexity,
            quantized_motion=x_quantized.clone().permute(0, 2, 1),
        )

    # x_out, loss, perplexity, code_idx

    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


class HumanVQVAE(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()

        self.nb_joints = args.nb_joints
        self.vqvae = VQVAE(
            input_dim=args.motion_dim,
            nb_code=args.codebook_size,
            code_dim=args.codebook_dim,
            down_t=args.down_sampling_ratio // 2,
            stride_t=args.down_sampling_ratio // 2,
            width=args.dim,
            depth=args.depth,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

    def load(self, path):
        pkg = torch.load(path, map_location="cuda")
        self.load_state_dict(pkg["model"])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, motion):
        b, t, c = motion.size()
        code_idx = self.vqvae.encode(motion)  # (N, T)
        return code_idx

    def forward(self, motion, mask=None):
        return self.vqvae(motion, mask)

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
