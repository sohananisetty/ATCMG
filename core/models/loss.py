import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.rotation_conversions as geometry
from core import MotionRep
from core.models.smpl.body_model import BodyModel


class ReConsLoss(nn.Module):
    def __init__(
        self,
        recons_loss: str = "l1_smooth",
        use_geodesic_loss: bool = False,
        use_simple_loss=True,
        nb_joints: int = 52,
        hml_rep: str = "gprvc",
        motion_rep: MotionRep = MotionRep.FULL,
    ):
        super().__init__()

        if recons_loss == "l1":
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == "l2":
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == "l1_smooth":
            self.Loss = torch.nn.SmoothL1Loss()

        self.use_geodesic_loss = use_geodesic_loss

        self.geodesic_loss = GeodesicLoss()

        # 4 global motion associated to root
        # 12 motion ( 3 positions ,6 rot6d, 3 vel xyz, )
        # 4 foot contact
        self.nb_joints = nb_joints
        self.hml_rep = hml_rep
        self.motion_rep = motion_rep
        self.use_simple_loss = use_simple_loss

        split_seq = []

        if "g" in hml_rep:
            split_seq.append(4)
        if "p" in hml_rep:
            if motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL:
                split_seq.append((self.nb_joints - 1) * 3)
            else:
                split_seq.append((self.nb_joints) * 3)
        if "r" in hml_rep:
            if motion_rep == MotionRep.BODY or motion_rep == MotionRep.FULL:
                split_seq.append((self.nb_joints - 1) * 6)
            else:
                split_seq.append((self.nb_joints) * 6)
        if "v" in hml_rep:
            split_seq.append(self.nb_joints * 3)
        if "c" in hml_rep:
            split_seq.append(4)

        self.split_seq = split_seq

    def get_c_from_v(self, vel_param):
        fid_r, fid_l = [8, 11], [7, 10]
        vel = vel_param.contiguous().view(vel_param.shape[:2] + (self.nb_joints, 3))
        pred_cl = torch.sum(vel[:, :, fid_l] ** 2, dim=-1) - 0.002
        pred_cr = torch.sum(vel[:, :, fid_r] ** 2, dim=-1) - 0.002
        pred_c = 1e3 * torch.cat([pred_cl, pred_cr], -1)
        return pred_c

    def forward(self, motion_pred, motion_gt, mask=None):

        if self.use_simple_loss:
            return self.Loss(motion_pred, motion_gt)

        hml_rep = self.hml_rep

        loss = 0
        params_pred = torch.split(motion_pred, self.split_seq, -1)
        params_gt = torch.split(motion_gt, self.split_seq, -1)

        if mask is not None:
            mask_split = torch.split(mask, self.split_seq, -1)[0]

        for indx, rep in enumerate(list(hml_rep)):

            mp = params_pred[indx]
            mg = params_gt[indx]

            if mask is not None:
                msk = mask_split[indx]
                mp = mp * msk[..., None]
                mg = mg * msk[..., None]

            if rep == "r" and self.use_geodesic_loss:

                if (
                    self.motion_rep == MotionRep.BODY
                    or self.motion_rep == MotionRep.FULL
                ):
                    nb_joints = self.nb_joints - 1
                else:
                    nb_joints = self.nb_joints

                mp = geometry.rotation_6d_to_matrix(
                    mp.view(-1, nb_joints, 6).contiguous()
                ).view(-1, 3, 3)
                mg = geometry.rotation_6d_to_matrix(
                    mg.view(-1, nb_joints, 6).contiguous()
                ).view(-1, 3, 3)

                loss += self.geodesic_loss(mp, mg)

            if rep == "v":

                if "c" in hml_rep:
                    pred_c = self.get_c_from_v(mp)
                    loss += nn.functional.binary_cross_entropy_with_logits(
                        pred_c, params_gt[-1]
                    )
                loss += self.Loss(mp, mg)

            else:

                loss += self.Loss(mp, mg)

        return loss


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2, epsilon=1e-7):
        """Compute the geodesic distance between two rotation matrices.
        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).
        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.permute(0, 2, 1))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        # cos = (m.diagonal(dim1=-2, dim2=-1).sum(-1) -1) /2
        # cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        # cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
        cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction="mean"):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss


class InfoNceLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        batch_size = z_i.shape[0]
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        negatives_mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool, device=z_i.device)
        ).float()

        nominator = torch.exp(positives / self.temperature)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss


class CLIPLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature))

    def cross_entropy(self, preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)

        logits = (z_i @ z_j.T) / self.temperature
        images_similarity = z_j @ z_j.T
        texts_similarity = z_i @ z_i.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = self.cross_entropy(logits, targets, reduction="none")
        images_loss = self.cross_entropy(logits.T, targets.T, reduction="none")
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()
