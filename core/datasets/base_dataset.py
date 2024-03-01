import os
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from core.models.utils import default
from torch.utils import data
from tqdm import tqdm
from utils.motion_processing.skeleton import getSkeleton


def splitHands(hand_data: np.array):
    # lp, rp, lr, rr, lv, rv = torch.split(hand_data, [45, 45, 90, 90, 45, 45], -1)
    lp, rp, lr, rr, lv, rv = np.split(hand_data, [45, 90, 180, 270, 315], -1)
    left_hand = np.concatenate([lp, lr, lv], -1)
    right_hand = np.concatenate([rp, rr, rv], -1)

    return left_hand, right_hand


class BaseMotionDataset(ABC, data.Dataset):
    def __init__(self, dataset_root, train_mode="full", use_rotation=True) -> None:
        """Initializes the BaseMotionDataset class.

        Args:
            dataset_root (str): The root directory of the dataset.
            train_mode (str, optional): The training mode. Defaults to "full".
            use_rotation (bool, optional): Whether to use rotation. Defaults to True.
        """

        self.use_rotation = use_rotation
        self.train_mode = train_mode
        self.data_root = dataset_root

        self.mean = np.load(os.path.join(self.data_root, "motion_data/Mean_cl.npy"))
        self.std = np.load(os.path.join(self.data_root, "motion_data/Std_cl.npy"))
        self.body_mean, self.hand_mean, self.full_mean = self.hmldata_process(self.mean)
        self.body_std, self.hand_std, self.full_std = self.hmldata_process(self.std)

        self.left_hand_mean, self.right_hand_mean = splitHands(self.hand_mean)
        self.left_hand_std, self.right_hand_std = splitHands(self.hand_std)

        self.skeleton = getSkeleton(
            os.path.join(self.data_root, "motion_data/000021_pos.npy"), self.train_mode
        )

    def inv_transform(self, data: torch.Tensor, train_mode=None) -> torch.Tensor:
        """Inverse transforms the data.

        Args:
            data (torch.Tensor): The input data.
            train_mode (Optional[str], optional): The training mode. Defaults to None.

        Returns:
            torch.Tensor: The inverse-transformed data.
        """
        train_mode = default(train_mode, self.train_mode)
        # body_mean, hand_mean, full_mean = self.hmldata_process(self.mean)
        # body_std, hand_std, full_std = self.hmldata_process(self.std)

        assert (
            data.shape[-1] == self.body_mean.shape[-1]
            or data.shape[-1] == self.hand_mean.shape[-1]
            or data.shape[-1] == self.full_mean.shape[-1]
            or data.shape[-1] == self.left_hand_mean.shape[-1]
        ), f"shape mismatch between input data {data.shape} and means {self.body_mean.shape} {self.hand_mean.shape} {self.full_mean.shape}"

        if data.shape[-1] == self.body_mean.shape[-1]:
            return data * (
                torch.Tensor(self.body_std).to(data.device) - 1e-8
            ) + torch.Tensor(self.body_mean).to(data.device)

        elif data.shape[-1] == self.hand_mean.shape[-1]:
            return data * (
                torch.Tensor(self.hand_std).to(data.device) - 1e-8
            ) + torch.Tensor(self.hand_mean).to(data.device)

        elif data.shape[-1] == self.full_mean.shape[-1]:
            return data * (
                torch.Tensor(self.full_std).to(data.device) - 1e-8
            ) + torch.Tensor(self.full_mean).to(data.device)

        elif data.shape[-1] == self.left_hand_mean.shape[-1]:
            if train_mode == "left_hand":
                return data * (
                    torch.Tensor(self.left_hand_std).to(data.device) - 1e-8
                ) + torch.Tensor(self.left_hand_mean).to(data.device)
            if train_mode == "right_hand":
                return data * (
                    torch.Tensor(self.right_hand_std).to(data.device) - 1e-8
                ) + torch.Tensor(self.right_hand_mean).to(data.device)

    def transform(self, data: np.array) -> np.array:
        assert (
            data.shape[-1] == self.body_mean.shape[-1]
            or data.shape[-1] == self.hand_mean.shape[-1]
            or data.shape[-1] == self.full_mean.shape[-1]
        ), f"shape mismatch between input data {data.shape} and means {self.body_mean.shape} {self.hand_mean.shape} {self.full_mean.shape}"

        if data.shape[-1] == self.body_mean.shape[-1]:
            return (data - self.body_mean) / (self.body_std + 1e-8)

        elif data.shape[-1] == self.hand_mean.shape[-1]:
            return (data - self.hand_mean) / (self.hand_std + 1e-8)

        elif data.shape[-1] == self.full_mean.shape[-1]:
            return (data - self.full_mean) / (self.full_std + 1e-8)

    def hmldata_process(
        self,
        hml_data: np.array,
        joint_num=52,
        body_joints=22,
        hand_joints=30,
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

        root_params = hml_data[..., :4]
        local_pos = hml_data[..., 4 : 4 + (joint_num - 1) * 3]
        local_rots = hml_data[
            ..., 4 + (joint_num - 1) * 3 : 4 + (joint_num - 1) * 3 + (joint_num - 1) * 6
        ]
        local_vels = hml_data[
            ...,
            4
            + (joint_num - 1) * 3
            + (joint_num - 1) * 6 : 4
            + (joint_num - 1) * 3
            + (joint_num - 1) * 6
            + joint_num * 3,
        ]
        foot = hml_data[..., -4:]

        local_rots_body = local_rots[..., : (body_joints - 1) * 6]
        local_rots_hand = local_rots[..., -hand_joints * 6 :]

        local_pos_body = local_pos[..., : (body_joints - 1) * 3]
        local_pos_hand = local_pos[..., -hand_joints * 3 :]

        local_vel_body = local_vels[..., : (body_joints) * 3]
        local_vel_hand = local_vels[..., -hand_joints * 3 :]

        if self.use_rotation:
            body_params = np.concatenate(
                [root_params, local_pos_body, local_rots_body, local_vel_body, foot], -1
            )
            hand_params = np.concatenate(
                [local_pos_hand, local_rots_hand, local_vel_hand], -1
            )

        if not self.use_rotation:
            body_params = np.concatenate(
                [root_params, local_pos_body, local_vel_body, foot], -1
            )
            hand_params = np.concatenate([local_pos_hand, local_vel_hand], -1)

            hml_data = np.concatenate(
                [
                    hml_data[..., : 4 + (joint_num - 1) * 3],
                    hml_data[..., 4 + (joint_num - 1) * 3 + (joint_num - 1) * 6 :],
                ],
                -1,
            )

        return body_params, hand_params, hml_data

    def processHand(self, body, hand, mode="remove"):
        l_wrist_pos_param = body[..., 4 + 19 * 3 : 4 + 20 * 3].reshape(-1, 1, 3)
        r_wrist_pos_param = body[..., 4 + 20 * 3 : 4 + 21 * 3].reshape(-1, 1, 3)

        def joinHands(hands: list):
            l_hand, r_hand = hands[0], hands[1]
            lp, lr, lv = torch.split(l_hand, [45, 90, 45], -1)
            rp, rr, rv = torch.split(r_hand, [45, 90, 45], -1)
            hand = torch.cat([lp, rp, lr, rr, lv, rv], -1)
            return hand

        if isinstance(hand, list):
            hand = joinHands(hand)

        finger_param = hand[..., :90].reshape(-1, 30, 3)
        if mode == "remove":
            finger_param_left = finger_param[:, :15, :] - l_wrist_pos_param
            finger_param_right = finger_param[:, 15:, :] - r_wrist_pos_param
        elif mode == "add":
            finger_param_left = finger_param[:, :15, :] + l_wrist_pos_param
            finger_param_right = finger_param[:, 15:, :] + r_wrist_pos_param
        hand[..., :45] = finger_param_left.reshape(hand.shape[:-1] + (45,))
        hand[..., 45:90] = finger_param_right.reshape(hand.shape[:-1] + (45,))

        return hand

    def joinBodyHands(self, body: torch.Tensor, hand) -> torch.Tensor:
        if isinstance(hand, list):
            hand = self.processHand(body, hand, "add")
        root_params = body[..., :4]
        body_pos = body[..., 4 : 4 + (22 - 1) * 3]
        body_vels = body[
            ...,
            4 + (22 - 1) * 3 + (22 - 1) * 6 : 4 + (22 - 1) * 3 + (22 - 1) * 6 + 22 * 3,
        ]
        foot = body[..., -4:]
        hand_pos = hand[..., 0:90]
        hand_vels = hand[..., 270:360]

        if self.use_rotation:
            body_rots = body[..., 4 + (22 - 1) * 3 : 4 + (22 - 1) * 3 + (22 - 1) * 6]
            hand_rots = hand[..., 90:270]

            full_params = torch.cat(
                [
                    root_params,
                    body_pos,
                    hand_pos,
                    body_rots,
                    hand_rots,
                    body_vels,
                    hand_vels,
                    foot,
                ],
                -1,
            )
        elif not self.use_rotation:
            full_params = torch.cat(
                [root_params, body_pos, hand_pos, body_vels, hand_vels, foot], -1
            )

        return full_params

    def toFullJointRepresentation(self, body, left, right):
        left_inv = self.inv_transform(left, "left_hand").squeeze().float()
        right_inv = self.inv_transform(right, "right_hand").squeeze().float()
        body_inv = self.inv_transform(body).squeeze().float()
        processed_hand = self.processHand(body_inv, [left_inv, right_inv], "add")
        joined_motion = self.joinBodyHands(body_inv, processed_hand)
        return joined_motion

    def get_processed_motion(self, motion, motion_rep="full", hml_rep="gprvc"):

        if motion_rep == "body":
            # motion = (motion - self.mean) / (self.std + 1e-8)
            motion = self.transform(motion)
            body_params, hand_params, full_params = self.hmldata_process(motion)
            selected_motion = body_params

        elif "hand" in motion_rep or motion_rep == "full":
            # motion = (motion - self.mean) / (self.std + 1e-8)
            body_params, hand_params, full_params = self.hmldata_process(motion)
            hand_params = self.processHand(
                body_params[None, ...], hand_params, "remove"
            )
            body_params = self.transform(body_params)
            hand_params = self.transform(hand_params)
            full_params = self.transform(full_params)

            selected_motion = hand_params if "hand" in self.motion_rep else full_params

        return selected_motion

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
