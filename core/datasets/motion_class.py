from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from core import MotionRep


@dataclass
class Motion:
    motion_rep: MotionRep = MotionRep.FULL
    hml_rep: str = "gprvc"
    root_params: Optional[Union[np.ndarray, torch.Tensor]] = None
    positions: Optional[Union[np.ndarray, torch.Tensor]] = None
    rotations: Optional[Union[np.ndarray, torch.Tensor]] = None
    velocity: Optional[Union[np.ndarray, torch.Tensor]] = None
    contact: Optional[Union[np.ndarray, torch.Tensor]] = None

    @property
    def device(self):
        for prm in [
            self.root_params,
            self.positions,
            self.rotations,
            self.velocity,
            self.contact,
        ]:
            try:
                return prm.device
            except:
                continue

        return "cpu"

    @property
    def nb_joints(self):
        if self.motion_rep == MotionRep.FULL:
            return 52
        elif self.motion_rep == MotionRep.BODY:
            return 22
        elif self.motion_rep == MotionRep.HAND:
            return 30
        elif self.motion_rep == MotionRep.LEFT_HAND:
            return 15
        elif self.motion_rep == MotionRep.RIGHT_HAND:
            return 15

    @property
    def dtype(self):

        if any(isinstance(value, np.ndarray) for value in self.__dict__.values()):
            return np.ndarray
        elif any(isinstance(value, torch.Tensor) for value in self.__dict__.values()):
            return torch.Tensor

    def numpy(self):
        """
        inplace convert to np array

        """
        self.root_params = (
            self.root_params.numpy()
            if isinstance(self.root_params, torch.Tensor)
            else self.root_params
        )
        self.positions = (
            self.positions.numpy()
            if isinstance(self.positions, torch.Tensor)
            else self.positions
        )
        self.rotations = (
            self.rotations.numpy()
            if isinstance(self.rotations, torch.Tensor)
            else self.rotations
        )
        self.velocity = (
            self.velocity.numpy()
            if isinstance(self.velocity, torch.Tensor)
            else self.velocity
        )
        self.contact = (
            self.contact.numpy()
            if isinstance(self.contact, torch.Tensor)
            else self.contact
        )

    def tensor(self):
        """
        inplace convert to torch.Tensor

        """

        self.root_params = (
            torch.tensor(self.root_params)
            if isinstance(self.root_params, np.ndarray)
            else self.root_params
        )
        self.positions = (
            torch.tensor(self.positions)
            if isinstance(self.positions, np.ndarray)
            else self.positions
        )
        self.rotations = (
            torch.tensor(self.rotations)
            if isinstance(self.rotations, np.ndarray)
            else self.rotations
        )
        self.velocity = (
            torch.tensor(self.velocity)
            if isinstance(self.velocity, np.ndarray)
            else self.velocity
        )
        self.contact = (
            torch.tensor(self.contact)
            if isinstance(self.contact, np.ndarray)
            else self.contact
        )

    def transform(self, mean, std):

        if isinstance(self.dtype, torch.Tensor):
            mean.tensor().to(self.device)
            std.tensor().to(self.device)

        for prm in [
            "root_params",
            "positions",
            "rotations",
            "velocity",
            "contact",
        ]:

            try:
                trns = (getattr(self, prm) - getattr(mean, prm)) / (
                    getattr(std, prm) + 1e-8
                )
                setattr(self, prm, trns)
            except:
                continue

    def inv_transform(self, mean, std):

        if isinstance(self.dtype, torch.Tensor):
            mean.tensor().to(self.device)
            std.tensor().to(self.device)

        for prm in [
            "root_params",
            "positions",
            "rotations",
            "velocity",
            "contact",
        ]:

            try:
                trns = getattr(self, prm) * (getattr(std, prm) - 1e-8) + getattr(
                    mean, prm
                )

                setattr(self, prm, trns)
            except:
                continue

    def __call__(self) -> Union[np.ndarray, torch.Tensor]:

        params = []
        if "g" in self.hml_rep:
            assert self.root_params is not None, f"need root for {self.hml_rep}"
            params.append(self.root_params)

        if "p" in self.hml_rep:
            assert self.positions is not None, f"need positions for {self.hml_rep}"
            params.append(self.positions)

        if "r" in self.hml_rep:
            assert self.rotations is not None, f"need rotation for {self.hml_rep}"
            params.append(self.rotations)

        if "v" in self.hml_rep:
            assert self.velocity is not None, f"need velocities for {self.hml_rep}"
            params.append(self.velocity)

        if "c" in self.hml_rep:
            assert self.contact is not None, f"need contact for {self.hml_rep}"
            params.append(self.contact)

        if isinstance(params[0], np.ndarray):
            return np.concatenate(params, -1)
        else:
            return torch.cat(params, -1)

    def __add__(self, motion2):
        assert (
            self.motion_rep != MotionRep.FULL and motion2.motion_rep != self.motion_rep
        ), "already full"

        # assert self.motion_rep == MotionRep.BODY, "only body + hand"

        assert self.hml_rep.replace("g", "").replace(
            "c", ""
        ) == motion2.hml_rep.replace("g", "").replace(
            "c", ""
        ), f"{self.hml_rep} and {motion2.hml_rep} not compatible"

        combo_motion = Motion(hml_rep=max(self.hml_rep, motion2.hml_rep, key=len))

        if "g" in self.hml_rep:
            assert self.root_params is not None, f"need root for {self.hml_rep} "
            combo_motion.root_params = self.root_params

        if "p" in self.hml_rep and "p" in motion2.hml_rep:
            concat_func = (
                np.concatenate if isinstance(self.positions, np.ndarray) else torch.cat
            )
            assert (
                self.positions is not None and motion2.positions is not None
            ), f"need positions for {self.hml_rep}"

            combo_motion.positions = concat_func(
                [self.positions, motion2.positions], -1
            )

        if "r" in self.hml_rep and "r" in motion2.hml_rep:
            concat_func = (
                np.concatenate if isinstance(self.positions, np.ndarray) else torch.cat
            )
            assert (
                self.rotations is not None and motion2.rotations is not None
            ), f"need positions for {self.hml_rep} {motion2.hml_rep}"
            combo_motion.positions = concat_func(
                [self.rotations, motion2.rotations], -1
            )

        if "v" in self.hml_rep and "v" in motion2.hml_rep:

            assert (
                self.velocity is not None and motion2.velocity is not None
            ), f"need positions for {self.hml_rep} {motion2.hml_rep}"

            concat_func = (
                np.concatenate if isinstance(self.positions, np.ndarray) else torch.cat
            )
            combo_motion.positions = concat_func([self.velocity, motion2.velocity], -1)

        if "c" in self.hml_rep:
            assert self.contact is not None, f"need contact for {self.hml_rep}"
            combo_motion.contact = self.contact

    def __len__(self) -> int:
        for prm in [
            self.root_params,
            self.positions,
            self.rotations,
            self.velocity,
            self.contact,
        ]:
            try:
                return prm.shape[0]
            except:
                continue

        return 0

    def __getitem__(self, idx: slice) -> "Motion":
        return Motion(
            motion_rep=self.motion_rep,
            hml_rep=self.hml_rep,
            root_params=self.root_params[idx] if self.root_params is not None else None,
            positions=self.positions[idx] if self.positions is not None else None,
            rotations=self.rotations[idx] if self.rotations is not None else None,
            velocity=self.velocity[idx] if self.velocity is not None else None,
            contact=self.contact[idx] if self.contact is not None else None,
        )

    # def __setitem__(
    #     self, idx: Union[slice, int], value: Union[List, np.ndarray, torch.Tensor]
    # ):
    #     if isinstance(idx, slice):
    #         start, stop, step = idx.start, idx.stop, idx.step
    #         prm_indices = [
    #             (idx, prm)
    #             for idx, prm in enumerate(
    #                 [
    #                     self.root_params,
    #                     self.positions,
    #                     self.rotations,
    #                     self.velocity,
    #                     self.contact,
    #                 ]
    #             )
    #             if prm is not None
    #         ]
    #         for idx, prm in prm_indices:
    #             prm[start:stop:step] = value[start:stop:step]
    #     else:
    #         prm_indices = [
    #             (idx, prm)
    #             for idx, prm in enumerate(
    #                 [
    #                     self.root_params,
    #                     self.positions,
    #                     self.rotations,
    #                     self.velocity,
    #                     self.contact,
    #                 ]
    #             )
    #             if prm is not None
    #         ]
    #         for idx, prm in prm_indices:
    #             prm[idx] = value[idx]
