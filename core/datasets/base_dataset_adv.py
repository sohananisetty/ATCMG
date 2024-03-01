import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from core.models.utils import default
from torch.utils import data
from tqdm import tqdm
from utils.motion_processing.quaternion import qinv, qrot, quaternion_to_cont6d

from .kinematics import getSkeleton
from core import MotionRep

from core import Motion

# @dataclass
# class Motion:
#     motion_rep: MotionRep = MotionRep.FULL
#     hml_rep: str = "gprvc"
#     root_params: Optional[Union[np.ndarray, torch.Tensor]] = None
#     positions: Optional[Union[np.ndarray, torch.Tensor]] = None
#     rotations: Optional[Union[np.ndarray, torch.Tensor]] = None
#     velocity: Optional[Union[np.ndarray, torch.Tensor]] = None
#     contact: Optional[Union[np.ndarray, torch.Tensor]] = None

#     @property
#     def device(self):
#         for prm in [
#             self.root_params,
#             self.positions,
#             self.rotations,
#             self.velocity,
#             self.contact,
#         ]:
#             try:
#                 return prm.device
#             except:
#                 continue

#         return "cpu"

#     def __len__(self) -> int:
#         for prm in [
#             self.root_params,
#             self.positions,
#             self.rotations,
#             self.velocity,
#             self.contact,
#         ]:
#             try:
#                 return prm.shape[0]
#             except:
#                 continue

#         return 0

#     @property
#     def nb_joints(self):
#         if self.motion_rep == MotionRep.FULL:
#             return 52
#         elif self.motion_rep == MotionRep.BODY:
#             return 22
#         elif self.motion_rep == MotionRep.HAND:
#             return 30
#         elif self.motion_rep == MotionRep.LEFT_HAND:
#             return 15
#         elif self.motion_rep == MotionRep.RIGHT_HAND:
#             return 15

#     @property
#     def dtype(self):

#         if any(isinstance(value, np.ndarray) for value in self.__dict__.values()):
#             return np.ndarray
#         elif any(isinstance(value, torch.Tensor) for value in self.__dict__.values()):
#             return torch.Tensor

#     def numpy(self):
#         """
#         inplace convert to np array

#         """
#         self.root_params = (
#             self.root_params.numpy()
#             if isinstance(self.root_params, torch.Tensor)
#             else self.root_params
#         )
#         self.positions = (
#             self.positions.numpy()
#             if isinstance(self.positions, torch.Tensor)
#             else self.positions
#         )
#         self.rotations = (
#             self.rotations.numpy()
#             if isinstance(self.rotations, torch.Tensor)
#             else self.rotations
#         )
#         self.velocity = (
#             self.velocity.numpy()
#             if isinstance(self.velocity, torch.Tensor)
#             else self.velocity
#         )
#         self.contact = (
#             self.contact.numpy()
#             if isinstance(self.contact, torch.Tensor)
#             else self.contact
#         )

#     def tensor(self):
#         """
#         inplace convert to torch.Tensor

#         """

#         self.root_params = (
#             torch.tensor(self.root_params)
#             if isinstance(self.root_params, np.ndarray)
#             else self.root_params
#         )
#         self.positions = (
#             torch.tensor(self.positions)
#             if isinstance(self.positions, np.ndarray)
#             else self.positions
#         )
#         self.rotations = (
#             torch.tensor(self.rotations)
#             if isinstance(self.rotations, np.ndarray)
#             else self.rotations
#         )
#         self.velocity = (
#             torch.tensor(self.velocity)
#             if isinstance(self.velocity, np.ndarray)
#             else self.velocity
#         )
#         self.contact = (
#             torch.tensor(self.contact)
#             if isinstance(self.contact, np.ndarray)
#             else self.contact
#         )

#     def __call__(self) -> Union[np.ndarray, torch.Tensor]:

#         params = []
#         if "g" in self.hml_rep:
#             assert self.root_params is not None, f"need root for {self.hml_rep}"
#             params.append(self.root_params)

#         if "p" in self.hml_rep:
#             assert self.positions is not None, f"need positions for {self.hml_rep}"
#             params.append(self.positions)

#         if "r" in self.hml_rep:
#             assert self.rotations is not None, f"need rotation for {self.hml_rep}"
#             params.append(self.rotations)

#         if "v" in self.hml_rep:
#             assert self.velocity is not None, f"need velocities for {self.hml_rep}"
#             params.append(self.velocity)

#         if "c" in self.hml_rep:
#             assert self.contact is not None, f"need contact for {self.hml_rep}"
#             params.append(self.contact)

#         if isinstance(params[0], np.ndarray):
#             return np.concatenate(params, -1)
#         else:
#             return torch.cat(params, -1)

#     def __add__(self, motion2):
#         assert (
#             self.motion_rep != MotionRep.FULL and motion2.motion_rep != self.motion_rep
#         ), "already full"

#         # assert self.motion_rep == MotionRep.BODY, "only body + hand"

#         assert self.hml_rep.replace("g", "").replace(
#             "c", ""
#         ) == motion2.hml_rep.replace("g", "").replace(
#             "c", ""
#         ), f"{self.hml_rep} and {motion2.hml_rep} not compatible"

#         combo_motion = Motion(hml_rep=max(self.hml_rep, motion2.hml_rep, key=len))

#         if "g" in self.hml_rep:
#             assert self.root_params is not None, f"need root for {self.hml_rep} "
#             combo_motion.root_params = self.root_params

#         if "p" in self.hml_rep and "p" in motion2.hml_rep:
#             concat_func = (
#                 np.concatenate if isinstance(self.positions, np.ndarray) else torch.cat
#             )
#             assert (
#                 self.positions is not None and motion2.positions is not None
#             ), f"need positions for {self.hml_rep}"

#             combo_motion.positions = concat_func(
#                 [self.positions, motion2.positions], -1
#             )

#         if "r" in self.hml_rep and "r" in motion2.hml_rep:
#             concat_func = (
#                 np.concatenate if isinstance(self.positions, np.ndarray) else torch.cat
#             )
#             assert (
#                 self.rotations is not None and motion2.rotations is not None
#             ), f"need positions for {self.hml_rep} {motion2.hml_rep}"
#             combo_motion.positions = concat_func(
#                 [self.rotations, motion2.rotations], -1
#             )

#         if "v" in self.hml_rep and "v" in motion2.hml_rep:

#             assert (
#                 self.velocity is not None and motion2.velocity is not None
#             ), f"need positions for {self.hml_rep} {motion2.hml_rep}"

#             concat_func = (
#                 np.concatenate if isinstance(self.positions, np.ndarray) else torch.cat
#             )
#             combo_motion.positions = concat_func([self.velocity, motion2.velocity], -1)

#         if "c" in self.hml_rep:
#             assert self.contact is not None, f"need contact for {self.hml_rep}"
#             combo_motion.contact = self.contact

#     def transform(self, mean, std):

#         if isinstance(self.dtype, torch.Tensor):
#             mean.tensor().to(self.device)
#             std.tensor().to(self.device)

#         for prm in [
#             "root_params",
#             "positions",
#             "rotations",
#             "velocity",
#             "contact",
#         ]:

#             try:
#                 trns = (getattr(self, prm) - getattr(mean, prm)) / (
#                     getattr(std, prm) + 1e-8
#                 )
#                 setattr(self, prm, trns)
#             except:
#                 continue

#     def inv_transform(self, mean, std):

#         if isinstance(self.dtype, torch.Tensor):
#             mean.tensor().to(self.device)
#             std.tensor().to(self.device)

#         for prm in [
#             "root_params",
#             "positions",
#             "rotations",
#             "velocity",
#             "contact",
#         ]:

#             try:
#                 trns = getattr(self, prm) * (getattr(std, prm) - 1e-8) + getattr(
#                     mean, prm
#                 )

#                 setattr(self, prm, trns)
#             except:
#                 continue

#     def __getitem__(self, idx: slice) -> "Motion":
#         return Motion(
#             root_params=self.root_params[idx] if self.root_params is not None else None,
#             positions=self.positions[idx] if self.positions is not None else None,
#             rotations=self.rotations[idx] if self.rotations is not None else None,
#             velocity=self.velocity[idx] if self.velocity is not None else None,
#             contact=self.contact[idx] if self.contact is not None else None,
#         )

#     def __setitem__(
#         self, idx: Union[slice, int], value: Union[List, np.ndarray, torch.Tensor]
#     ):
#         if isinstance(idx, slice):
#             start, stop, step = idx.start, idx.stop, idx.step
#             prm_indices = [
#                 (idx, prm)
#                 for idx, prm in enumerate(
#                     [
#                         self.root_params,
#                         self.positions,
#                         self.rotations,
#                         self.velocity,
#                         self.contact,
#                     ]
#                 )
#                 if prm is not None
#             ]
#             for idx, prm in prm_indices:
#                 prm[start:stop:step] = value[start:stop:step]
#         else:
#             prm_indices = [
#                 (idx, prm)
#                 for idx, prm in enumerate(
#                     [
#                         self.root_params,
#                         self.positions,
#                         self.rotations,
#                         self.velocity,
#                         self.contact,
#                     ]
#                 )
#                 if prm is not None
#             ]
#             for idx, prm in prm_indices:
#                 prm[idx] = value[idx]


class Motion2Positions:

    def __init__(self, data_root, motion_rep) -> None:
        self.skeleton = getSkeleton(
            os.path.join(data_root, "motion_data/000021_pos.npy"),
            motion_rep=motion_rep.value,
        )

    def recover_root_rot_pos(self, data: Motion):
        rot_vel = data.root_params[..., 0]
        r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
        """Get Y-axis rotation from rotation velocity"""
        r_rot_ang[..., 1:] = rot_vel[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

        r_rot_quat = torch.zeros(data.root_params.shape[:-1] + (4,)).to(data.device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

        r_pos = torch.zeros(data.root_params.shape[:-1] + (3,)).to(data.device)
        r_pos[..., 1:, [0, 2]] = data.root_params[..., :-1, 1:3]
        """Add Y-axis rotation to root position"""
        r_pos = qrot(qinv(r_rot_quat), r_pos)

        r_pos = torch.cumsum(r_pos, dim=-2)

        r_pos[..., 1] = data.root_params[..., 3]
        return r_rot_quat, r_pos

    def recover_from_rot(self, data: Motion):

        data.tensor()

        joints_num = data.nb_joints
        r_rot_quat, r_pos = self.recover_root_rot_pos(data)
        r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

        cont6d_params = data.rotations
        cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
        cont6d_params = cont6d_params.view(-1, joints_num, 6)

        positions = self.skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

        return positions

    def recover_from_ric(self, data: Motion):
        data.tensor()

        joints_num = data.nb_joints
        if joints_num == 22 or joints_num == 52:
            r_rot_quat, r_pos = self.recover_root_rot_pos(data)
            positions = data.positions
            positions = positions.view(positions.shape[:-1] + (-1, 3)).to(torch.float32)

            """Add Y-axis rotation to local joints"""
            positions = qrot(
                qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
                positions,
            )

            """Add root XZ to joints"""
            positions[..., 0] += r_pos[..., 0:1]
            positions[..., 2] += r_pos[..., 2:3]

            """Concate root and joints"""
            positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        else:
            positions = data.positions.view(positions.shape[:-1] + (-1, 3)).float()

        return positions


def splitHands(hand_data: Motion) -> Tuple[Motion, Motion]:

    left_hand = Motion(MotionRep.LEFT_HAND, hml_rep=hand_data.hml_rep)
    right_hand = Motion(MotionRep.RIGHT_HAND, hml_rep=hand_data.hml_rep)
    hml_rep = hand_data.hml_rep
    if "p" in hml_rep:
        left_hand.positions = hand_data.positions[..., :45]
        right_hand.positions = hand_data.positions[..., 45:]
    if "r" in hml_rep:
        left_hand.rotations = hand_data.rotations[..., :90]
        right_hand.rotations = hand_data.rotations[..., 90:]
    if "v" in hml_rep:
        left_hand.velocity = hand_data.velocity[..., :45]
        right_hand.velocity = hand_data.velocity[..., 45:]

    return left_hand, right_hand


class BaseMotionDataset(ABC, data.Dataset):
    def __init__(
        self,
        dataset_root,
        motion_rep=MotionRep.FULL,
        hml_rep="gprvc",
    ) -> None:
        """Initializes the BaseMotionDataset class.

        Args:
            dataset_root (str): The root directory of the dataset.
            train_mode (str, optional): The training motion_rep. Defaults to "full".
            use_rotation (bool, optional): Whether to use rotation. Defaults to True.
        """

        self.motion_rep = motion_rep
        self.hml_rep = hml_rep

        ## Making sure hand does not have global params

        if self.motion_rep in [
            MotionRep.HAND,
            MotionRep.LEFT_HAND,
            MotionRep.RIGHT_HAND,
        ]:
            self.hml_rep = self.hml_rep.replace("g", "").replace("c", "")

        self.data_root = dataset_root

        self.mean = np.load(os.path.join(self.data_root, "motion_data/Mean_cl.npy"))
        self.std = np.load(os.path.join(self.data_root, "motion_data/Std_cl.npy"))
        self.body_mean, self.hand_mean, self.full_mean = self.hmldata_process(self.mean)
        self.body_std, self.hand_std, self.full_std = self.hmldata_process(self.std)

        self.left_hand_mean, self.right_hand_mean = splitHands(self.hand_mean)
        self.left_hand_std, self.right_hand_std = splitHands(self.hand_std)

        self.mot2pos = Motion2Positions(self.data_root, self.motion_rep)

    def inv_transform(self, data: Motion) -> Motion:
        """Inverse transforms the data.

        Args:
            data (torch.Tensor): The input data.
            train_mode (Optional[str], optional): The training motion_rep. Defaults to None.

        Returns:
            torch.Tensor: The inverse-transformed data.
        """
        motion_rep = data.motion_rep

        if motion_rep == MotionRep.FULL:
            data.inv_transform(self.mean, self.std)
        elif motion_rep == MotionRep.HAND:
            data.inv_transform(self.hand_mean, self.hand_std)
        elif motion_rep == MotionRep.BODY:
            data.inv_transform(self.body_mean, self.body_std)

        elif motion_rep == MotionRep.LEFT_HAND:
            data.inv_transform(self.left_hand_mean, self.left_hand_std)
        elif motion_rep == MotionRep.RIGHT_HAND:
            data.inv_transform(self.right_hand_mean, self.right_hand_std)

        return data

    def transform(self, data: Motion) -> Motion:
        motion_rep = data.motion_rep

        if motion_rep == MotionRep.FULL:
            data.transform(self.mean, self.std)
        elif motion_rep == MotionRep.HAND:
            data.transform(self.hand_mean, self.hand_std)
        elif motion_rep == MotionRep.BODY:
            data.transform(self.body_mean, self.body_std)

        elif motion_rep == MotionRep.LEFT_HAND:
            data.transform(self.left_hand_mean, self.left_hand_std)
        elif motion_rep == MotionRep.RIGHT_HAND:
            data.transform(self.right_hand_mean, self.right_hand_std)

        return data

    def hmldata_process(
        self,
        hml_data: np.array,
        joint_num: int = 52,
        body_joints: int = 22,
        hand_joints: int = 30,
        hml_rep: Optional[str] = None,
    ):
        """Processes the HML data.

        Args:
            hml_data (np.array): The input HML data.
            joint_num (int, optional): The number of joints. Defaults to 52.
            body_joints (int, optional): The number of body joints. Defaults to 22.
            hand_joints (int, optional): The number of hand joints. Defaults to 30.

        Returns:
            tuple: The processed data.
        """

        if hml_rep is None:
            hml_rep = self.hml_rep

        split_seq = np.cumsum(
            [4, (joint_num - 1) * 3, (joint_num - 1) * 6, joint_num * 3, 4]
        )

        root_params, local_pos, local_rots, local_vels, foot = np.split(
            hml_data, split_seq, -1
        )[:-1]

        local_pos_body, local_pos_hand = np.split(
            local_pos, np.cumsum([(body_joints - 1) * 3, hand_joints * 3]), -1
        )[:-1]
        local_rots_body, local_rots_hand = np.split(
            local_rots, np.cumsum([(body_joints - 1) * 6, hand_joints * 6]), -1
        )[:-1]
        local_vel_body, local_vel_hand = np.split(
            local_vels, np.cumsum([(body_joints) * 3, hand_joints * 3]), -1
        )[:-1]

        body_motion = Motion(
            motion_rep=MotionRep.BODY,
            hml_rep=hml_rep,
            root_params=root_params if "g" in hml_rep else None,
            positions=local_pos_body if "p" in hml_rep else None,
            rotations=local_rots_body if "r" in hml_rep else None,
            velocity=local_vel_body if "v" in hml_rep else None,
            contact=foot if "c" in hml_rep else None,
        )
        hand_motion = Motion(
            motion_rep=MotionRep.HAND,
            hml_rep=hml_rep.replace("g", "").replace("c", ""),
            positions=local_pos_hand if "p" in hml_rep else None,
            rotations=local_rots_hand if "r" in hml_rep else None,
            velocity=local_vel_hand if "v" in hml_rep else None,
        )

        full_motion = Motion(
            motion_rep=MotionRep.FULL,
            root_params=root_params if "g" in hml_rep else None,
            positions=local_pos if "p" in hml_rep else None,
            rotations=local_rots if "r" in hml_rep else None,
            velocity=local_vels if "v" in hml_rep else None,
            contact=foot if "c" in hml_rep else None,
        )

        return body_motion, hand_motion, full_motion

    def processHand(
        self, body: Motion, hand: Union[Motion, List[Motion]], mode="remove"
    ) -> Motion:
        l_wrist_pos_param = body.positions[..., 19 * 3 : 20 * 3].reshape(-1, 1, 3)
        r_wrist_pos_param = body.positions[..., 20 * 3 : 21 * 3].reshape(-1, 1, 3)

        def joinHands(hands: List[Motion]) -> Motion:
            l_hand, r_hand = hands[0], hands[1]
            hand = l_hand + r_hand
            hand.motion_rep = MotionRep.HAND
            hand.hml_rep = body.hml_rep.replace("g", "").replace("c", "")

            # hand = Motion(
            #     motion_rep=MotionRep.HAND,
            #     hml_rep=body.hml_rep.replace("g", "").replace("c", ""),
            #     positions=torch.cat(l_hand.positions, r_hand.positions),
            #     rotations=torch.cat(l_hand.rotations, r_hand.rotations),
            #     velocity=torch.cat(l_hand.velocity, r_hand.velocity),
            # )

            return hand

        if isinstance(hand, list):
            hand = joinHands(hand)

        finger_param = hand.positions.reshape(-1, 30, 3)
        if mode == "remove":
            finger_param_left = finger_param[:, :15, :] - l_wrist_pos_param
            finger_param_right = finger_param[:, 15:, :] - r_wrist_pos_param
        elif mode == "add":
            finger_param_left = finger_param[:, :15, :] + l_wrist_pos_param
            finger_param_right = finger_param[:, 15:, :] + r_wrist_pos_param
        hand.positions[..., :45] = finger_param_left.reshape(
            hand.positions.shape[:-1] + (45,)
        )
        hand.positions[..., 45:90] = finger_param_right.reshape(
            hand.positions.shape[:-1] + (45,)
        )

        return hand

    def joinBodyHands(self, body: Motion, hand: Union[Motion, List[Motion]]) -> Motion:
        if isinstance(hand, list):
            hand = self.processHand(body, hand, "add")

        full_params = body + hand

        return full_params

    def toFullJointRepresentation(self, body: Motion, left: Motion, right: Motion):
        left_inv = self.inv_transform(left)
        right_inv = self.inv_transform(right)
        body_inv = self.inv_transform(body)
        processed_hand = self.processHand(body_inv, [left_inv, right_inv], "add")
        joined_motion = self.joinBodyHands(body_inv, processed_hand)
        return joined_motion

    def get_processed_motion(
        self,
        motion: Union[np.ndarray, torch.Tensor],
        motion_rep: MotionRep = MotionRep.FULL,
        hml_rep: str = "gprvc",
    ) -> Motion:

        if motion_rep == MotionRep.FULL:

            body_params, hand_params, full_params = self.hmldata_process(
                motion, hml_rep=hml_rep
            )
            full_params = self.transform(full_params)
            return full_params

        elif motion_rep == MotionRep.BODY:

            body_params, hand_params, full_params = self.hmldata_process(
                motion, hml_rep=hml_rep
            )
            body_params = self.transform(body_params)
            return body_params

        elif motion_rep in [
            MotionRep.HAND,
            MotionRep.LEFT_HAND,
            MotionRep.RIGHT_HAND,
        ]:

            body_params, hand_params, full_params = self.hmldata_process(
                motion, hml_rep=hml_rep
            )
            hand_params = self.processHand(body_params, hand_params, "remove")
            hand_params = self.transform(hand_params)

            if motion_rep == MotionRep.HAND:
                return hand_params

            left_hand, right_hand = splitHands(hand_params)
            if motion_rep == MotionRep.LEFT_HAND:
                return left_hand
            else:
                return right_hand

        # elif motion_rep == MotionRep.HAND:
        #     body_params, hand_params, full_params = self.hmldata_process(
        #         motion, hml_rep=hml_rep
        #     )
        #     hand_params = self.processHand(body_params, hand_params, "remove")
        #     hand_params = self.transform(hand_params)

        #     selected_motion = hand_params

        # elif motion_rep == MotionRep.LEFT_HAND:
        #     body_params, hand_params, full_params = self.hmldata_process(
        #         motion, hml_rep=hml_rep
        #     )
        #     hand_params = self.processHand(body_params, hand_params, "remove")
        #     hand_params = self.transform(hand_params)
        #     left_hand, right_hand = splitHands(hand_params)

        #     selected_motion = left_hand

        # elif motion_rep == MotionRep.RIGHT_HAND:
        #     body_params, hand_params, full_params = self.hmldata_process(
        #         motion, hml_rep=hml_rep
        #     )
        #     hand_params = self.processHand(body_params, hand_params, "remove")
        #     hand_params = self.transform(hand_params)
        #     left_hand, right_hand = splitHands(hand_params)

        #     selected_motion = right_hand
        # return selected_motion

    def to_xyz(self, motion: Motion, hml_rep=None) -> torch.Tensor:
        hml_rep = self.hml_rep if hml_rep is None else hml_rep

        if "p" not in hml_rep:
            xyz = self.mot2pos.recover_from_rot(motion)
        else:
            xyz = self.mot2pos.recover_from_ric(motion)

        return xyz

    def findAllFile(self, base):
        file_path = []
        for root, ds, fs in os.walk(base, followlinks=True):
            for f in fs:
                fullname = os.path.join(root, f)
                file_path.append(fullname)
        return file_path

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("not implemented")

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError("not implemented")
