import os

import numpy as np
import torch
import torch.nn as nn
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from einops import rearrange

# from utils import compute_trajectory, export_ply_trajectory, normalize

# from .base_model import BaseModel
# from .fk import ForwardKinematicsLayer
# from .residual_blocks import ResidualBlock, SkeletonResidual, residual_ratio
from core.models.skeleton import (
    SkeletonConv,
    SkeletonPool,
    build_edge_topology,
    find_neighbor,
)


def residual_ratio(k):
    return 1 / (k + 1)


class BatchStatistics(nn.Module):
    def __init__(self, affine=-1):
        super(BatchStatistics, self).__init__()
        self.affine = nn.Sequential() if affine == -1 else Affine(affine)
        self.loss = 0

    def clear_loss(self):
        self.loss = 0

    def compute_loss(self, input):
        input_flat = input.view(input.size(1), input.numel() // input.size(1))
        mu = input_flat.mean(1)
        logvar = (input_flat.pow(2).mean(1) - mu.pow(2)).sqrt().log()

        self.loss = mu.pow(2).mean() + logvar.pow(2).mean()

    def forward(self, input):
        self.compute_loss(input)
        return self.affine(input)


class Affine(nn.Module):
    def __init__(self, num_parameters, scale=True, bias=True, scale_init=1.0):
        super(Affine, self).__init__()
        if scale:
            self.scale = nn.Parameter(torch.ones(num_parameters) * scale_init)
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(num_parameters))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        output = input
        if self.scale is not None:
            scale = self.scale.unsqueeze(0)
            while scale.dim() < input.dim():
                scale = scale.unsqueeze(2)
        output = output.mul(scale)

        if self.bias is not None:
            bias = self.bias.unsqueeze(0)
            while bias.dim() < input.dim():
                bias = bias.unsqueeze(2)
        output += bias

        return output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        residual_ratio,
        activation,
        batch_statistics=False,
        last_layer=False,
    ):
        super(ResidualBlock, self).__init__()

        self.residual_ratio = residual_ratio
        self.shortcut_ratio = 1 - residual_ratio

        residual = []
        residual.append(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        )
        if batch_statistics:
            residual.append(BatchStatistics(out_channels))
        if not last_layer:
            residual.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        self.residual = nn.Sequential(*residual)

        self.shortcut = nn.Sequential(
            nn.AvgPool1d(kernel_size=2) if stride == 2 else nn.Sequential(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            (
                BatchStatistics(out_channels)
                if (in_channels != out_channels and batch_statistics is True)
                else nn.Sequential()
            ),
        )

    def forward(self, input):
        return self.residual(input).mul(self.residual_ratio) + self.shortcut(input).mul(
            self.shortcut_ratio
        )


class SkeletonResidual(nn.Module):
    def __init__(
        self,
        topology,
        neighbour_list,
        joint_num,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        padding_mode,
        bias,
        extra_conv,
        pooling_mode,
        activation,
        last_pool,
    ):
        super(SkeletonResidual, self).__init__()

        kernel_even = False if kernel_size % 2 else True

        seq = []
        for _ in range(extra_conv):
            # (T, J, D) => (T, J, D)
            seq.append(
                SkeletonConv(
                    neighbour_list,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    joint_num=joint_num,
                    kernel_size=kernel_size - 1 if kernel_even else kernel_size,
                    stride=1,
                    padding=padding,
                    padding_mode=padding_mode,
                    bias=bias,
                )
            )
            seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        # (T, J, D) => (T/2, J, 2D)
        seq.append(
            SkeletonConv(
                neighbour_list,
                in_channels=in_channels,
                out_channels=out_channels,
                joint_num=joint_num,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
                add_offset=False,
            )
        )
        seq.append(nn.GroupNorm(8, out_channels))  # FIXME: REMEMBER TO CHANGE BACK !!!
        self.residual = nn.Sequential(*seq)

        # (T, J, D) => (T/2, J, 2D)
        self.shortcut = SkeletonConv(
            neighbour_list,
            in_channels=in_channels,
            out_channels=out_channels,
            joint_num=joint_num,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=True,
            add_offset=False,
        )

        seq = []
        # (T/2, J, 2D) => (T/2, J', 2D)
        pool = SkeletonPool(
            edges=topology,
            pooling_mode=pooling_mode,
            channels_per_edge=out_channels // len(neighbour_list),
            last_pool=last_pool,
        )
        if len(pool.pooling_list) != pool.edge_num:
            seq.append(pool)
        seq.append(nn.PReLU() if activation == "relu" else nn.Tanh())
        self.common = nn.Sequential(*seq)

    def forward(self, input):
        output = self.residual(input) + self.shortcut(input)

        return self.common(output)


class Predictor(nn.Module):
    def __init__(
        self,
        channel_base=6,
        out_channels=12,
        skeleton_dist=1,
        activation="relu",
        kernel_size=15,
        num_layers=3,
        use_residual_blocks=True,
        skeleton_pool="mean",
        padding_mode="reflect",
        extra_conv=0,
    ):
        super().__init__()

        parents_body = [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
        ]
        topology = build_edge_topology(parents_body)

        self.topologies = [topology]
        self.channel_base = [channel_base]
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        # self.args = args
        # self.convs = []

        kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        bias = True

        for _ in range(num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(num_layers):
            seq = []
            neighbour_list = find_neighbor(self.topologies[i], skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(
                edges=self.topologies[i],
                pooling_mode=skeleton_pool,
                channels_per_edge=out_channels // len(neighbour_list),
                last_pool=last_pool,
            )

            if use_residual_blocks:
                # (T, J, D) => (T, J', 2D)
                seq.append(
                    SkeletonResidual(
                        self.topologies[i],
                        neighbour_list,
                        joint_num=self.edge_num[i],
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode,
                        bias=bias,
                        extra_conv=extra_conv,
                        pooling_mode=skeleton_pool,
                        activation=activation,
                        last_pool=last_pool,
                    )
                )
            else:
                for _ in range(extra_conv):
                    # (T, J, D) => (T, J, D)
                    seq.append(
                        SkeletonConv(
                            neighbour_list,
                            in_channels=in_channels,
                            out_channels=in_channels,
                            joint_num=self.edge_num[i],
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                            padding_mode=padding_mode,
                            bias=bias,
                        )
                    )
                    seq.append(nn.PReLU())
                # (T, J, D) => (T, J, 2D)
                seq.append(
                    SkeletonConv(
                        neighbour_list,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        joint_num=self.edge_num[i],
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        padding_mode=padding_mode,
                        bias=bias,
                        add_offset=False,
                        in_offset_channel=3
                        * self.channel_base[i]
                        // self.channel_base[0],
                    )
                )
                # self.convs.append(seq[-1])

                seq.append(pool)
                seq.append(nn.PReLU())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

        in_channels = self.channel_base[-1] * len(self.pooling_list[-1])
        out_channels = out_channels  # root orient (6) + root vel (3) + root height (1) + contacts (8)

        self.global_motion = nn.Sequential(
            ResidualBlock(
                in_channels=in_channels,
                out_channels=512,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(1),
                activation=activation,
            ),
            ResidualBlock(
                in_channels=512,
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(2),
                activation=activation,
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(3),
                activation=activation,
            ),
            ResidualBlock(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(4),
                activation=activation,
                last_layer=True,
            ),
        )

    def forward(self, input):
        feature = input
        for layer in self.layers:
            feature = layer(feature)

        return self.global_motion(feature)


def FeedForward(
    dim,
    mult=4,
    dropout=0.1,
):
    """https://arxiv.org/abs/2110.09456"""

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.LayerNorm(inner_dim, bias=False),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Predictor2(nn.Module):
    def __init__(
        self,
        channel_base=6,
        out_channels=8,
        attn_dim=128,
        activation="relu",
        kernel_size=15,
        num_layers=3,
        num_joints=21,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        bias = True

        self.in_project = nn.Linear(channel_base, attn_dim)
        self.norm_sa = nn.LayerNorm(attn_dim, eps=1e-5, bias=False)
        self.norm_out = nn.LayerNorm(attn_dim, eps=1e-5, bias=False)

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.MultiheadAttention(
                            embed_dim=attn_dim,
                            num_heads=4,
                            dropout=0.1,
                            bias=False,
                            batch_first=True,
                        ),
                        FeedForward(dim=attn_dim, mult=4, dropout=0.1),
                    ]
                )
            )

        self.global_motion = nn.Sequential(
            ResidualBlock(
                in_channels=attn_dim * num_joints,
                out_channels=512,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(1),
                activation=activation,
            ),
            ResidualBlock(
                in_channels=512,
                out_channels=256,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(2),
                activation=activation,
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=128,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(3),
                activation=activation,
            ),
            ResidualBlock(
                in_channels=128,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                residual_ratio=residual_ratio(4),
                activation=activation,
                last_layer=True,
            ),
        )

    def load(self, path):
        pkg = torch.load(path, map_location="cuda")
        print(f"loading model with {pkg['steps']} steps and {pkg['total_loss']} loss")
        self.load_state_dict(pkg["model"])

    def forward(self, input, mask=None):
        x = input
        x = self.in_project(x)
        b, n, j, d = x.shape
        x = rearrange(x, "b n j d -> (b n) j d", b=b, n=n, j=j, d=d)
        for attn, ff1 in self.layers:
            x = self.norm_sa(x)

            x = (
                attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=mask,
                    need_weights=False,
                )[0]
                + x
            )
            x = self.norm_out(ff1(x) + x)
        x = rearrange(x, "(b n) j d -> b (j d) n", b=b, n=n, j=j, d=d)

        x = self.global_motion(x)

        return x.permute(0, 2, 1)
