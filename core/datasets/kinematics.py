import numpy as np
import scipy.ndimage.filters as filters
import torch
from utils.motion_processing.quaternion import (
    cont6d_to_matrix,
    cont6d_to_matrix_np,
    qbetween_np,
    qinv_np,
    qmul,
    qmul_np,
    qrot,
    qrot_np,
)

hand_joints_id = [i for i in range(22, 52)]
body_joints_id = [i for i in range(0, 22)]
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
r_hip, l_hip = 2, 1
joints_num = 52


smplx_raw_body_offsets = np.array(
    [
        [0, 0, 0],  # pelvis
        [1, 0, 0],  # left_hip
        [-1, 0, 0],  # right_hip
        [0, 1, 0],  # spine1
        [0, -1, 0],  # left_knee
        [0, -1, 0],  # right_knee
        [0, 1, 0],  # spine2
        [0, -1, 0],  # left_ankle
        [0, -1, 0],  # right_ankle
        [0, 1, 0],  # spine3
        [0, 0, 1],  # left_foot
        [0, 0, 1],  # right_foot
        [0, 1, 0],  # neck
        [1, 0, 0],  # left_collar
        [-1, 0, 0],  # right_collar
        [0, 0, 1],  # head
        [0, -1, 0],  # left_shoulder
        [0, -1, 0],  # right_shoulder
        [0, -1, 0],  # left_elbow
        [0, -1, 0],  # right_elbow
        [0, -1, 0],  # left_wrist
        [0, -1, 0],  # right_wrist
    ]
)

smplx_raw_hand_offsets = np.array(
    [
        [0, -1, 0],  # left_index1
        [0, -1, 0],  # left_index2
        [0, -1, 0],  # left_index3
        [0, -1, 0],  # left_middle1
        [0, -1, 0],  # left_middle2
        [0, -1, 0],  # left_middle3
        [0, -1, 0],  # left_pinky1
        [0, -1, 0],  # left_pinky2
        [0, -1, 0],  # left_pinky3
        [0, -1, 0],  # left_ring1
        [0, -1, 0],  # left_ring2
        [0, -1, 0],  # left_ring3
        [0, -1, 0],  # left_thumb1
        [0, -1, 0],  # left_thumb2
        [0, -1, 0],  # left_thumb3
        [0, -1, 0],  # right_index1
        [0, -1, 0],  # right_index2
        [0, -1, 0],  # right_index3
        [0, -1, 0],  # right_middle1
        [0, -1, 0],  # right_middle2
        [0, -1, 0],  # right_middle3
        [0, -1, 0],  # right_pinky1
        [0, -1, 0],  # right_pinky2
        [0, -1, 0],  # right_pinky3
        [0, -1, 0],  # right_ring1
        [0, -1, 0],  # right_ring2
        [0, -1, 0],  # right_ring3
        [0, -1, 0],  # right_thumb1
        [0, -1, 0],  # right_thumb2
        [0, -1, 0],  # right_thumb3
    ]
)


smplx_full_raw_offsets = np.concatenate(
    (smplx_raw_body_offsets, smplx_raw_hand_offsets), axis=0
)
# pelvis ---> left_hip ---> left_knee ---> left_ankle ---> left_foot 0, 1, 4, 7, 10
# pelvis --> right_hip ---> right_knee ---> right_ankle ---> right_foot 0, 2, 5, 8, 11
# pelvis ---> spine1 ---> spine2 ---> spine3 ---> neck ---> head 0, 3, 6, 9, 12, 15
# spine3 ---> left_collar ---> left_shoulder --->  left_elbow ---> left_wrist 9, 13, 16, 18, 20
# spine3 ---> right_collar ---> right_shoulder --->  right_elbow ---> right_wrist  9, 14, 17, 19, 21

smplx_kinematic_chain = [
    [0, 1, 4, 7, 10],
    [0, 2, 5, 8, 11],
    [0, 3, 6, 9, 12, 15],
    [9, 13, 16, 18, 20],
    [9, 14, 17, 19, 21],
]

# left_wrist ---> left_index1 ---> left_index2 --->left_index3
# left_wrist ---> left_middle1 ---> left_middle2 --->left_middle3
# left_wrist ---> left_pinky1 ---> left_pinky2 --->left_pinky3
# left_wrist ---> left_ring1 ---> left_ring2 --->left_ring3
# left_wrist ---> left_thumb1 ---> left_thumb2 --->left_thumb3


smplx_left_hand_chain = [
    [20, 22, 23, 24],
    [20, 25, 26, 27],
    [20, 28, 29, 30],
    [20, 31, 32, 33],
    [20, 34, 35, 36],
]

# right_wrist ---> right_index1 ---> right_index2 --->right_index3
# right_wrist ---> right_middle1 ---> right_middle2 --->right_middle3
# right_wrist ---> right_pinky1 ---> right_pinky2 --->right_pinky3
# right_wrist ---> right_ring1 ---> right_ring2 --->right_ring3
# right_wrist ---> right_thumb1 ---> right_thumb2 --->right_thumb3
smplx_right_hand_chain = [
    [21, 37, 38, 39],
    [21, 40, 41, 42],
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 49, 50, 51],
]


smplx_only_one_hand_chain = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
]

smplx_only_both_hand_chain = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
    [18, 19, 20],
    [21, 22, 23],
    [24, 25, 26],
    [27, 28, 29],
]

smplx_full_kinematic_chain = (
    smplx_kinematic_chain + smplx_left_hand_chain + smplx_right_hand_chain
)

smplx_hand_kinematic_chain = smplx_left_hand_chain + smplx_right_hand_chain


class Skeleton(object):
    def __init__(self, offset, kinematic_tree, device="cpu"):
        self.device = device
        self._raw_offset_np = offset.numpy()
        self._raw_offset = offset.clone().detach().to(device).float()
        self._kinematic_tree = kinematic_tree
        self._offset = None
        self._parents = [0] * len(self._raw_offset)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j - 1]

    def njoints(self):
        return len(self._raw_offset)

    def offset(self):
        return self._offset

    def set_offset(self, offsets):
        self._offset = offsets.clone().detach().to(self.device).float()

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._parents

    # joints (batch_size, joints_num, 3)
    def get_offsets_joints_batch(self, joints):
        assert len(joints.shape) == 3
        _offsets = self._raw_offset.expand(joints.shape[0], -1, -1).clone()
        for i in range(1, self._raw_offset.shape[0]):
            _offsets[:, i] = (
                torch.norm(joints[:, i] - joints[:, self._parents[i]], p=2, dim=1)[
                    :, None
                ]
                * _offsets[:, i]
            )

        self._offset = _offsets.detach()
        return _offsets

    # joints (joints_num, 3)
    def get_offsets_joints(self, joints):
        assert len(joints.shape) == 2
        _offsets = self._raw_offset.clone()
        for i in range(1, self._raw_offset.shape[0]):
            # print(joints.shape)
            _offsets[i] = (
                torch.norm(joints[i] - joints[self._parents[i]], p=2, dim=0)
                * _offsets[i]
            )

        self._offset = _offsets.detach()
        return _offsets

    # face_joint_idx should follow the order of right hip, left hip, right shoulder, left shoulder
    # joints (batch_size, joints_num, 3)
    def inverse_kinematics_np(self, joints, face_joint_idx, smooth_forward=False):
        assert len(face_joint_idx) == 4
        """Get Forward Direction"""
        # l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
        r_hip, l_hip, sdr_r, sdr_l = face_joint_idx
        across1 = joints[:, r_hip] - joints[:, l_hip]
        across2 = joints[:, sdr_r] - joints[:, sdr_l]
        across = across1 + across2
        across = across / np.sqrt((across**2).sum(axis=-1))[:, np.newaxis]
        # print(across1.shape, across2.shape)

        # forward (batch_size, 3)
        forward = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        if smooth_forward:
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode="nearest")
            # forward (batch_size, 3)
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[..., np.newaxis]

        """Get Root Rotation"""
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        root_quat = qbetween_np(forward, target)

        """Inverse Kinematics"""
        # quat_params (batch_size, joints_num, 4)
        # print(joints.shape[:-1])
        quat_params = np.zeros(joints.shape[:-1] + (4,))
        # print(quat_params.shape)
        root_quat[0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        quat_params[:, 0] = root_quat
        # quat_params[0, 0] = np.array([[1.0, 0.0, 0.0, 0.0]])
        for chain in self._kinematic_tree:
            R = root_quat
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_offset_np[chain[j + 1]][np.newaxis, ...].repeat(
                    len(joints), axis=0
                )
                # print(u.shape)
                # (batch, 3)
                v = joints[:, chain[j + 1]] - joints[:, chain[j]]
                v = v / np.sqrt((v**2).sum(axis=-1))[:, np.newaxis]
                # print(u.shape, v.shape)
                rot_u_v = qbetween_np(u, v)

                R_loc = qmul_np(qinv_np(R), rot_u_v)

                quat_params[:, chain[j + 1], :] = R_loc
                R = qmul_np(R, R_loc)

        return quat_params

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics(
        self, quat_params, root_pos, skel_joints=None, do_root_R=True
    ):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        joints = torch.zeros(quat_params.shape[:-1] + (3,)).to(self.device)
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = (
                    torch.tensor([[1.0, 0.0, 0.0, 0.0]])
                    .expand(len(quat_params), -1)
                    .detach()
                    .to(self.device)
                )
            for i in range(1, len(chain)):
                R = qmul(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    # Be sure root joint is at the beginning of kinematic chains
    def forward_kinematics_np(
        self, quat_params, root_pos, skel_joints=None, do_root_R=True
    ):
        # quat_params (batch_size, joints_num, 4)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(quat_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(quat_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                R = quat_params[:, 0]
            else:
                R = np.array([[1.0, 0.0, 0.0, 0.0]]).repeat(len(quat_params), axis=0)
            for i in range(1, len(chain)):
                R = qmul_np(R, quat_params[:, chain[i]])
                offset_vec = offsets[:, chain[i]]
                joints[:, chain[i]] = qrot_np(R, offset_vec) + joints[:, chain[i - 1]]
        return joints

    def forward_kinematics_cont6d_np(
        self, cont6d_params, root_pos, skel_joints=None, do_root_R=True
    ):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        offsets = offsets.numpy()
        joints = np.zeros(cont6d_params.shape[:-1] + (3,))
        joints[:, 0] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix_np(cont6d_params[:, 0])
            else:
                matR = np.eye(3)[np.newaxis, :].repeat(len(cont6d_params), axis=0)
            for i in range(1, len(chain)):
                matR = np.matmul(matR, cont6d_to_matrix_np(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]][..., np.newaxis]
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = (
                    np.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
                )
        return joints

    def forward_kinematics_cont6d(
        self, cont6d_params, root_pos, skel_joints=None, do_root_R=True
    ):
        # cont6d_params (batch_size, joints_num, 6)
        # joints (batch_size, joints_num, 3)
        # root_pos (batch_size, 3)
        if skel_joints is not None:
            # skel_joints = torch.from_numpy(skel_joints)
            offsets = self.get_offsets_joints_batch(skel_joints)
        if len(self._offset.shape) == 2:
            offsets = self._offset.expand(cont6d_params.shape[0], -1, -1)
        joints = torch.zeros(cont6d_params.shape[:-1] + (3,)).to(cont6d_params.device)
        joints[..., 0, :] = root_pos
        for chain in self._kinematic_tree:
            if do_root_R:
                matR = cont6d_to_matrix(cont6d_params[:, 0])
            else:
                matR = (
                    torch.eye(3)
                    .expand((len(cont6d_params), -1, -1))
                    .detach()
                    .to(cont6d_params.device)
                )
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, cont6d_to_matrix(cont6d_params[:, chain[i]]))
                offset_vec = offsets[:, chain[i]].unsqueeze(-1)
                # print(matR.shape, offset_vec.shape)
                joints[:, chain[i]] = (
                    torch.matmul(matR, offset_vec).squeeze(-1) + joints[:, chain[i - 1]]
                )
        return joints


def getSkeleton(
    example_data_path="/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/core/datasets/data/000021_pos.npy",
    motion_rep="full",
):

    example_data = torch.Tensor(np.load(example_data_path))

    if motion_rep == "body":
        example_data = example_data[:, body_joints_id, :]
        kinematic_chain = smplx_kinematic_chain
        n_raw_offsets = torch.from_numpy(smplx_raw_body_offsets)

    elif "hand" in motion_rep:
        if motion_rep == "left_hand":
            example_data = example_data[:, hand_joints_id[:15], :]
            kinematic_chain = smplx_only_one_hand_chain
            n_raw_offsets = torch.from_numpy(smplx_raw_hand_offsets[:15])
        elif motion_rep == "right_hand":
            example_data = example_data[:, hand_joints_id[15:], :]
            kinematic_chain = smplx_only_one_hand_chain
            n_raw_offsets = torch.from_numpy(smplx_raw_hand_offsets[15:])
        else:
            example_data = example_data[:, hand_joints_id, :]
            kinematic_chain = smplx_only_both_hand_chain
            n_raw_offsets = torch.from_numpy(smplx_raw_hand_offsets)
    elif motion_rep == "full":
        if example_data.shape[1] != 52:
            example_data = example_data[:, body_joints_id + hand_joints_id, :]
        kinematic_chain = smplx_full_kinematic_chain
        n_raw_offsets = torch.from_numpy(smplx_full_raw_offsets)

    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    return tgt_skel
