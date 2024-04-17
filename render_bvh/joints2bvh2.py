import render_bvh.Animation as Animation
import numpy as np
import os
from render_bvh.InverseKinematics import (
    BasicInverseKinematics,
    BasicJacobianIK,
    InverseKinematics,
)
from render_bvh.Quaternions import Quaternions
import render_bvh.BVH_mod as BVH
from render_bvh.remove_fs import *

# from utils.plot_script import plot_3d_motion
# from render_bvh.utils import paramUtil
# from render_bvh.common.skeleton import Skeleton
import torch

from torch import nn
from render_bvh.utils.quat import ik_rot, between, fk, ik
from tqdm import tqdm

smplx_full_kinematic_chain = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
    [20, 22, 23, 24],
    [20, 25, 26, 27],
    [20, 28, 29, 30],
    [20, 31, 32, 33],
    [20, 34, 35, 36],
    [21, 37, 38, 39],
    [21, 40, 41, 42],
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 49, 50, 51],
]


def get_grot(glb, parent, offset):
    root_quat = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(glb.shape[0], axis=0)[:, None]
    local_pos = glb[:, 1:] - glb[:, parent[1:]]
    norm_offset = offset[1:] / np.linalg.norm(offset[1:], axis=-1, keepdims=True)
    norm_lpos = local_pos / np.linalg.norm(local_pos, axis=-1, keepdims=True)
    grot = between(norm_offset, norm_lpos)
    grot = np.concatenate((root_quat, grot), axis=1)
    grot /= np.linalg.norm(grot, axis=-1, keepdims=True)
    return grot


class Joint2BVHConvertor:
    def __init__(self, path=None):
        path = "./render_bvh/data/smplx_full_template.bvh" if path is None else path
        self.template = BVH.load(path, need_quater=True)

        self.re_order = [
            0,
            1,
            4,
            7,
            10,
            2,
            5,
            8,
            11,
            3,
            6,
            9,
            12,
            15,
            13,
            16,
            18,
            20,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            14,
            17,
            19,
            21,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
        ]

        self.re_order_inv = [
            0,
            1,
            5,
            9,
            2,
            6,
            10,
            3,
            7,
            11,
            4,
            8,
            12,
            14,
            33,
            13,
            15,
            34,
            16,
            35,
            17,
            36,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
        ]

        self.end_points = [
            4,
            8,
            13,
            17,
            21,
        ]  # left_foot right_foot head left_wrist right_wrist

        self.template_offset = self.template.offsets.copy()

        self.parents = [
            -1,
            0,
            1,
            2,
            3,
            0,
            5,
            6,
            7,
            0,
            9,
            10,
            11,
            12,
            11,
            14,
            15,
            16,
            17,
            18,
            19,
            17,
            21,
            22,
            17,
            24,
            25,
            17,
            27,
            28,
            17,
            30,
            31,
            11,
            33,
            34,
            35,
            36,
            37,
            38,
            36,
            40,
            41,
            36,
            43,
            44,
            36,
            46,
            47,
            36,
            49,
            50,
        ]

    def convert(self, positions, filename, iterations=10, foot_ik=True):
        """
        Convert the SMPL joint positions to Mocap BVH
        :param positions: (N, 52, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        """
        positions = positions[:, self.re_order]
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions.shape[0], axis=-0)
        new_anim.positions[:, 0] = positions[:, 0]

        if foot_ik:
            positions = remove_fs(
                positions,
                None,
                fid_l=(3, 4),
                fid_r=(7, 8),
                interp_length=5,
                force_on_floor=True,
            )
        ik_solver = BasicInverseKinematics(
            new_anim, positions, iterations=iterations, silent=True
        )
        new_anim = ik_solver()

        # BVH.save(filename, new_anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = Animation.positions_global(new_anim)[:, self.re_order_inv]
        if filename is not None:
            BVH.save(
                filename,
                new_anim,
                names=new_anim.names,
                frametime=1 / 30,
                order="zyx",
                quater=True,
            )
        return new_anim, glb

    def convert_sgd(self, positions, filename, iterations=100, foot_ik=True):
        """
        Convert the SMPL joint positions to Mocap BVH

        :param positions: (N, 22, 3)
        :param filename: Save path for resulting BVH
        :param iterations: iterations for optimizing rotations, 10 is usually enough
        :param foot_ik: whether to enfore foot inverse kinematics, removing foot slide issue.
        :return:
        """

        ## Positional Foot locking ##
        glb = positions[:, self.re_order]

        if foot_ik:
            glb = remove_fs(
                glb,
                None,
                fid_l=(3, 4),
                fid_r=(7, 8),
                interp_length=2,
                force_on_floor=True,
            )

        ## Fit BVH ##
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(glb.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(glb.shape[0], axis=-0)
        new_anim.positions[:, 0] = glb[:, 0]
        anim = new_anim.copy()

        rot = torch.tensor(anim.rotations.qs, dtype=torch.float)
        pos = torch.tensor(anim.positions[:, 0, :], dtype=torch.float)
        offset = torch.tensor(anim.offsets, dtype=torch.float)

        glb = torch.tensor(glb, dtype=torch.float)
        ik_solver = InverseKinematics(rot, pos, offset, anim.parents, glb)
        print("Fixing foot contact using IK...")
        for i in tqdm(range(iterations)):
            mse = ik_solver.step()
            # print(i, mse)

        rotations = ik_solver.rotations.detach().cpu()
        norm = torch.norm(rotations, dim=-1, keepdim=True)
        rotations /= norm

        anim.rotations = Quaternions(rotations.numpy())
        anim.rotations[:, self.end_points] = Quaternions.id(
            (anim.rotations.shape[0], len(self.end_points))
        )
        anim.positions[:, 0, :] = ik_solver.position.detach().cpu().numpy()
        if filename is not None:
            BVH.save(
                filename,
                anim,
                names=new_anim.names,
                frametime=1 / 20,
                order="zyx",
                quater=True,
            )
        # BVH.save(filename[:-3] + 'bvh', anim, names=new_anim.names, frametime=1 / 20, order='zyx', quater=True)
        glb = Animation.positions_global(anim)[:, self.re_order_inv]
        return anim, glb


if __name__ == "__main__":
    # file = 'batch0_sample13_repeat0_len196.npy'
    # file = 'batch2_sample10_repeat0_len156.npy'
    # file = 'batch2_sample13_repeat0_len196.npy' #line #57 new_anim.positions = lpos #new_anim.positions[0:1].repeat(positions.shape[0], axis=-0) #TODO, figure out why it's important
    # file = 'batch1_sample12_repeat0_len196.npy' #hard case karate
    # file = 'batch1_sample14_repeat0_len180.npy'
    # file = 'batch0_sample3_repeat0_len192.npy'
    # file = 'batch1_sample4_repeat0_len136.npy'

    # file = 'batch0_sample0_repeat0_len152.npy'
    # path = f'/Users/yuxuanmu/project/MaskMIT/demo/cond4_topkr0.9_ts18_tau1.0_s1009/joints/{file}'
    # joints = np.load(path)
    # converter = Joint2BVHConvertor()
    # new_anim = converter.convert(joints, './gen_L196.mp4', foot_ik=True)

    folder = "/Users/yuxuanmu/project/MaskMIT/demo/cond4_topkr0.9_ts18_tau1.0_s1009"
    files = os.listdir(os.path.join(folder, "joints"))
    files = [f for f in files if "repeat" in f]
    converter = Joint2BVHConvertor()
    for f in tqdm(files):
        joints = np.load(os.path.join(folder, "joints", f))
        converter.convert(
            joints,
            os.path.join(folder, "ik_animations", f"ik_{f}".replace("npy", "mp4")),
            foot_ik=True,
        )
