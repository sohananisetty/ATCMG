import torch.nn as nn
import torch.nn.functional as F
from core.models.resnetVQ.resnet import Resnet1D
import numpy as np
import torch
from utils.motion_processing.quaternion import qinv, qrot, quaternion_to_cont6d
import utils.rotation_conversions as geometry


class TCN(nn.Module):
    def __init__(
        self,
        in_dim=2,
        dim=64,
        out_dim=2,
        depth=2,
        k=[7, 3],
        norm=None,
        dilation=3,
    ) -> None:
        super().__init__()

        # self.model = nn.Sequential(
        #     nn.Conv1d(in_dim, dim, k[0], 1, "same"),
        #     # nn.Dropout1d(0.2),
        #     (
        #         Resnet1D(
        #             dim,
        #             depth,
        #             dilation_growth_rate=dilation,
        #             activation="gelu",
        #             norm=norm,
        #         )
        #         if depth != 0
        #         else nn.Identity()
        #     ),
        #     # nn.Dropout1d(0.2),
        #     nn.Conv1d(dim, out_dim, k[1], 1, "same"),
        # )

        self.model = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode="reflect", **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get("dilation", 1)
        stride = kwargs.get("stride", 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0), mode=self.pad_mode)
        return self.conv(x)


class ResidualUnit(nn.Module):
    def __init__(
        self, chan_in, chan_out, dilation=1, kernel_size=7, pad_mode="reflect"
    ):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(
                chan_in, chan_out, kernel_size, dilation=dilation, pad_mode=pad_mode
            ),
            nn.ELU(),
            CausalConv1d(chan_out, chan_out, 1, pad_mode=pad_mode),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x) + x


class CausalTCN(nn.Module):
    def __init__(
        self,
        in_dim=2,
        dim=64,
        out_dim=2,
        depth=2,
        k=[7, 3],
        dilation=3,
    ) -> None:
        super().__init__()

        self.blocks = [CausalConv1d(in_dim, dim, k[0])]

        self.blocks.extend(
            [
                ResidualUnit(dim, dim, dilation=dilation ** (d // 2))
                for d in range(depth)
            ]
        )

        self.blocks.append(CausalConv1d(dim, out_dim, k[1]))

        self.model = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.model(x)


class Traj2Orient(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.output_dim = config.output_dim
        if config.causal:
            self.model = CausalTCN(
                in_dim=config.input_dim,
                dim=config.dim,
                out_dim=config.output_dim,
                k=config.k,
                depth=config.depth,
                dilation=config.dilation,
            )
        else:
            self.model = TCN(
                in_dim=config.input_dim,
                dim=config.dim,
                out_dim=config.output_dim,
                k=config.k,
                depth=config.depth,
                dilation=config.dilation,
            )

        if config.loss_fnc == "l1":
            self.loss_fnc = nn.L1Loss()
        elif config.loss_fnc == "l2":
            self.loss_fnc = nn.MSELoss()
        elif config.loss_fnc == "l1_smooth":
            self.loss_fnc = nn.SmoothL1Loss()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def predict(self, traj):
        ##traj b n 2
        rel_pos = self.process_input(torch.Tensor(traj).to(self.model.device))
        msk = torch.ones_like(rel_pos)[..., 0]
        with torch.no_grad():
            pred_orient = self((rel_pos, msk))

        return pred_orient

    def process_input(self, traj):
        rel_pos = torch.zeros_like(traj)
        rel_pos[:, 1:] = traj[:, 1:] - traj[:, :-1]

        return rel_pos

    def forward(self, inputs):

        ##x (b n d , b n)

        motion = inputs[0].to(self.device)
        mask = inputs[1].to(torch.bool).to(self.device)
        r_rot = motion[..., :4]
        r_pos = motion[..., 4:]
        r_pos = r_pos[..., [0, 2]]

        if self.output_dim == 1:
            r_rot = geometry.quaternion_to_axis_angle(r_rot)[..., 1:2]
        elif self.output_dim == 2:
            r_rot = geometry.quaternion_to_matrix(r_rot)[..., [0, 0], [0, 2]]
        elif self.output_dim == 6:
            r_rot = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(r_rot))

        y = r_rot

        x = self.process_input(r_pos)

        # x = x.permute(0, 2, 1)
        x = self.model(x)
        # x = x.permute(0, 2, 1)

        if mask is not None:
            x = x * mask[..., None]

        loss = self.loss_fnc(x, y)
        return loss, x
