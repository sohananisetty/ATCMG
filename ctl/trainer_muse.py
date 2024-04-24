import itertools
import os
import random
from collections import Counter
from functools import partial
from math import sqrt
from pathlib import Path
from time import time

import numpy as np
import torch
import transformers
import utils.vis_utils.plot_3d_global as plot_3d
import wandb
from configs.config import get_cfg_defaults as vqvae_get_cfg_defaults
from configs.config_t2m import get_cfg_defaults as muse_get_cfg_defaults
from core import AudioRep, MotionRep, MotionTokenizerParams, TextRep
from core.datasets.conditioner import ConditionProvider
from core.datasets.multimodal_dataset import load_dataset_gen, simple_collate
from core.models.generation.muse import MotionMuse
from core.models.generation.muse2 import MotionMuse as MotionMuse2

from core.models.resnetVQ.vqvae import HumanVQVAE
from core.models.utils import get_obj_from_str, instantiate_from_config
from core.optimizer import get_optimizer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from utils.motion_processing.hml_process import recover_from_ric, recover_root_rot_pos
from yacs.config import CfgNode
from core.models.TMR.tmr import TMR
from configs.config_tmr import get_cfg_defaults as get_cfg_defaults_tmr


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


dataset_qualities = {
    "animation": 0,
    "humanml": 1,
    "perform": 0,
    "GRAB": 1,
    "idea400": 1,
    "humman": 0,
    "beat": 1,
    "game_motion": 0,
    "music": 0,
    "aist": 1,
    "fitness": 0,
    "moyo": 1,
    "choreomaster": 1,
    "dance": 0,
    "kungfu": 0,
    "EgoBody": 0,
    # "HAA500": 1.0,
}


# main trainer class


class MotionMuseTrainer(nn.Module):
    def __init__(
        self,
        args: CfgNode,
    ):
        super().__init__()
        self.model_name = args.model_name

        transformers.set_seed(42)

        self.args = args
        self.training_args = args.train
        self.dataset_args = args.dataset
        self.model_args = args.motion_generator

        self.motion_rep = self.dataset_args.motion_rep
        self.hml_rep = self.dataset_args.hml_rep

        self.num_train_steps = self.training_args.num_train_iters
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        target = self.model_args.pop("target")
        fuse_config = self.args.fuser
        vqvae_config = self.args.vqvae

        self.use_align = self.model_args.align_loss_weight > 0

        self.load_vqvae(vqvae_config)

        print(self.model_args)

        if self.use_align:
            self.load_tmr(self.args.tmr)

            self.motion_muse = MotionMuse2(
                self.model_args, fuse_config, self.body_model, self.tmr
            ).to(self.device)

        else:

            self.motion_muse = MotionMuse2(self.model_args, fuse_config).to(self.device)
        total = sum(p.numel() for p in self.motion_muse.parameters() if p.requires_grad)
        print("Total training params: %.2fM" % (total / 1e6))

        # if self.motion_rep == "hand":
        #     self.body_muse = self.getBodyMuse(nme="motion_muse_film")

        self.register_buffer("steps", torch.Tensor([0]))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        self.optim = get_optimizer(
            self.motion_muse.parameters(),
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )
        condition_provider = ConditionProvider(
            text_conditioner_name=self.dataset_args.text_conditioner_name,
            motion_rep=MotionRep(self.dataset_args.motion_rep),
            audio_rep=AudioRep(self.dataset_args.audio_rep),
            text_rep=TextRep(self.dataset_args.text_rep),
            motion_padding=self.dataset_args.motion_padding,
            audio_padding=self.dataset_args.audio_padding,
            motion_max_length_s=self.dataset_args.motion_max_length_s,
            audio_max_length_s=self.dataset_args.motion_max_length_s,
            pad_id=MotionTokenizerParams(self.model_args.num_tokens).pad_token_id,
            fps=self.dataset_args.fps / self.dataset_args.down_sampling_ratio,
        )

        # dataset_names = {
        #     "animation": 0.5,
        #     "humanml": 3.5,
        #     "perform": 0.6,
        #     "GRAB": 1.0,
        #     "idea400": 2.0,
        #     "humman": 0.5,
        #     "beat": 2.5,
        #     "game_motion": 0.8,
        #     "music": 0.5,
        #     "aist": 2.0,
        #     "fitness": 1.0,
        #     "moyo": 1.5,
        #     "choreomaster": 2.5,
        #     "dance": 1.0,
        #     "kungfu": 1.0,
        #     "EgoBody": 0.5,
        #     # "HAA500": 1.0,
        # }

        dataset_names = {
            "animation": 0.4,
            "humanml": 1.5,
            "perform": 0.5,
            "GRAB": 1.0,
            "idea400": 1.0,
            "humman": 0.5,
            "beat": 1.2,
            "game_motion": 0.8,
            "music": 0.8,
            "aist": 1.5,
            "fitness": 0.7,
            "moyo": 1.0,
            "choreomaster": 2.5,
            "dance": 0.5,
            "kungfu": 0.5,
            "EgoBody": 0.5,
            # "HAA500": 1.0,
        }

        train_ds, sampler_train, weights_train = load_dataset_gen(
            dataset_args=self.dataset_args,
            split="train",
            dataset_names=list(dataset_names.keys()),
            weight_scale=list(dataset_names.values()),
        )
        test_ds, _, _ = load_dataset_gen(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="test",
        )

        self.render_ds, _, _ = load_dataset_gen(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="render",
        )

        self.print(
            f"training with training {len(train_ds)} and test dataset of {len(test_ds)} and render {len(self.render_ds)} samples"
        )

        # dataloader
        self.dl = DataLoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=partial(
                simple_collate, conditioner=condition_provider, permute=True
            ),
            drop_last=False,
            # pin_memory=False,
            # num_workers=2,
        )
        self.valid_dl = DataLoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=partial(
                simple_collate, conditioner=condition_provider, permute=True
            ),
            drop_last=False,
            # pin_memory=False,
            # num_workers=2,
        )

        self.render_dl = DataLoader(
            self.render_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(
                simple_collate, conditioner=condition_provider, permute=True
            ),
        )

        self.dl_iter = cycle(self.dl)
        # self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = self.training_args.save_steps
        self.log_losses_every = self.training_args.logging_steps
        self.evaluate_every = self.training_args.evaluate_every
        self.calc_metrics_every = self.training_args.evaluate_every
        self.wandb_every = self.training_args.wandb_every

        # if self.is_main:
        wandb.login()
        wandb.init(project=self.model_name)

    def load_tmr(self, tmr_args):

        if tmr_args.config is not None:
            tmr_cfg = get_cfg_defaults_tmr()
            tmr_cfg.merge_from_file(tmr_args.config)
            tmr_cfg.freeze()
            tmr_parms = tmr_cfg.tmr

            _ = tmr_parms.pop("target")
            motion_encoder = (
                instantiate_from_config(tmr_cfg.motion_encoder).to(self.device).eval()
            )
            text_encoder = (
                instantiate_from_config(tmr_cfg.text_encoder).to(self.device).eval()
            )
            motion_decoder = (
                instantiate_from_config(tmr_cfg.motion_decoder).to(self.device).eval()
            )
            self.tmr = (
                TMR(
                    motion_encoder,
                    text_encoder,
                    motion_decoder,
                    lr=tmr_cfg.train.learning_rate,
                    **tmr_parms,
                )
                .to(self.device)
                .eval()
            )
            pkg = torch.load(os.path.join(tmr_cfg.output_dir, "tmr.pt"))
            self.tmr.load_state_dict(pkg["model"])
            self.tmr.freeze()

            self.tmr_cfg = tmr_cfg
        else:
            self.tmr = None
            self.tmr_cfg = None

        # return tmr, tmr_cfg

    def load_vqvae(self, vqvae_args):

        if vqvae_args.body_config is not None:

            self.body_cfg = vqvae_get_cfg_defaults()
            self.body_cfg.merge_from_file(vqvae_args.body_config)
            self.body_model = (
                instantiate_from_config(self.body_cfg.vqvae).to(self.device).eval()
            )
            self.body_model.load(
                os.path.join(self.body_cfg.output_dir, "vqvae_motion.pt")
            )
            self.body_model.freeze()

        else:
            self.body_model = None

        if vqvae_args.left_hand_config is not None:
            self.left_cfg = vqvae_get_cfg_defaults()
            self.left_cfg.merge_from_file(vqvae_args.left_hand_config)
            self.left_hand_model = (
                instantiate_from_config(self.left_cfg.vqvae).to(self.device).eval()
            )
            self.left_hand_model.load(
                os.path.join(self.left_cfg.output_dir, "vqvae_motion.pt")
            )
            self.left_hand_model.freeze()
        else:
            self.left_hand_model = None

        if vqvae_args.right_hand_config is not None:
            self.right_cfg = vqvae_get_cfg_defaults()
            self.right_cfg.merge_from_file(vqvae_args.right_hand_config)
            self.right_hand_model = (
                instantiate_from_config(self.right_cfg.vqvae).to(self.device).eval()
            )
            self.right_hand_model.load(
                os.path.join(self.right_cfg.output_dir, "vqvae_motion.pt")
            )
            self.right_hand_model.freeze()
        else:
            self.right_hand_model = None

    def print(self, msg):
        # self.accelerator.print(msg)
        print(msg)

    @property
    def device(self):
        return torch.device("cuda")

    def save(self, path, loss=None):
        pkg = dict(
            model=self.motion_muse.state_dict(),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
            config=dict(self.args),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cuda")
        self.motion_muse.load_state_dict(pkg["model"])

        self.optim.load_state_dict(pkg["optim"])
        self.steps = pkg["steps"]
        self.best_loss = pkg["total_loss"]

    def to_device(self, batch):
        for k in batch.keys():
            try:
                batch[k] = batch[k].to(self.device)
            except:
                continue

        return batch

    def train_step(self):
        steps = int(self.steps.item())

        self.motion_muse = self.motion_muse.train()

        # logs

        logs = {}

        for _ in range(self.grad_accum_every):
            inputs, conditions = next(self.dl_iter)

            inputs = self.to_device(inputs)
            conditions = self.to_device(conditions)

            # inputs["motion"][0] b n 1

            motions = inputs["motion"][0].to(torch.long)
            motion_mask = inputs["motion"][1]
            quality_list = torch.LongTensor(
                [dataset_qualities[nm.split("/")[0]] for nm in inputs["names"]]
            ).to(motions.device)

            # out_body = self.body_muse(
            #     (inputs["motion"][0][:, :1, :], motion_mask),
            #     conditions,
            #     train_critic=False,
            # )
            # conditions["body"] = (out_body.embed, motion_mask)
            # _ = conditions.pop("audio")
            # _ = conditions.pop("text")

            out = self.motion_muse(
                (motions, motion_mask),
                conditions,
                quality_list=quality_list,
            )
            loss = out.loss
            if out.ce_per_codebook is not None and len(out.ce_per_codebook) == 3:
                loss = (
                    loss
                    + 1.5 * out.ce_per_codebook[0]
                    + out.ce_per_codebook[1]
                    + out.ce_per_codebook[2]
                )

            loss = loss / self.grad_accum_every

            loss.backward()

            if self.use_align:
                accum_log(
                    logs,
                    dict(
                        loss=loss.detach().cpu(),
                        align_loss=out.align_loss.detach().cpu(),
                    ),
                )
            else:

                accum_log(
                    logs,
                    dict(loss=loss.detach().cpu()),
                )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: model cross entropy loss: {logs['loss'].float():.3} "

        if self.use_align:
            losses_str += f"align loss {logs['align_loss'].float():.3}"

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"motion_muse.{steps}.pt"
            )
            self.save(model_path, logs["loss"])
            print(float(logs["loss"]), self.best_loss)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(self.output_dir, f"motion_muse.pt")
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        if steps % self.evaluate_every == 0:
            self.validation_step()
            self.sample_render_hmlvec(os.path.join(self.output_dir, "samples"))

        self.steps += 1
        return logs

    def validation_step(self):
        self.motion_muse.eval()
        val_loss_ae = {}

        self.print(f"validation start")
        cnt = 0

        with torch.no_grad():
            for inputs, conditions in tqdm(
                (self.valid_dl),
                position=0,
                leave=True,
            ):

                inputs = self.to_device(inputs)
                conditions = self.to_device(conditions)
                loss_dict = {}

                motions = inputs["motion"][0].to(torch.long)
                motion_mask = inputs["motion"][1]

                out = self.motion_muse((motions, motion_mask), conditions)
                loss = out.loss
                if out.ce_per_codebook is not None and len(out.ce_per_codebook) == 3:
                    loss = (
                        loss
                        + 2 * out.ce_per_codebook[0]
                        + out.ce_per_codebook[1]
                        + out.ce_per_codebook[2]
                    )

                loss_dict["loss"] = loss.detach().cpu()

                if self.use_align:
                    loss_dict["align_loss"] = out.align_loss.detach().cpu()

                for key, value in loss_dict.items():
                    if key in val_loss_ae:
                        val_loss_ae[key] += value
                    else:
                        val_loss_ae[key] = value

                    cnt += 1

        for key in val_loss_ae.keys():
            val_loss_ae[key] = val_loss_ae[key] / cnt

        for key, value in val_loss_ae.items():
            wandb.log({f"val_ce_loss/{key}": value})

            print(f"val/{key}", value)

        self.motion_muse.train()

    def bkn_to_motion(self, codes, dset):
        # codes b k n

        k = codes.shape[1]
        mrep = dset.motion_rep

        if k == 1:
            if mrep == MotionRep("body"):

                body_inds = codes[:, 0]
                body_motion = self.body_model.decode(body_inds[0:1]).detach().cpu()

                if self.dataset_args.remove_translation:
                    z = torch.zeros(
                        body_motion.shape[:-1] + (2,),
                        dtype=body_motion.dtype,
                        device=body_motion.device,
                    )
                    body_motion = torch.cat(
                        [body_motion[..., 0:1], z, body_motion[..., 1:]], -1
                    )

                body_M = dset.toMotion(
                    body_motion[0],
                    motion_rep=MotionRep("body"),
                    hml_rep=self.body_cfg.dataset.hml_rep,
                )

                return body_M

            elif mrep == MotionRep("left_hand"):

                left_inds = codes[:, 0]
                left_motion = self.left_hand_model.decode(left_inds[0:1]).detach().cpu()
                left_M = dset.toMotion(
                    left_motion[0],
                    motion_rep=MotionRep(self.left_cfg.dataset.motion_rep),
                    hml_rep=self.left_cfg.dataset.hml_rep,
                )
                return left_M

            elif mrep == MotionRep("right_hand"):
                right_inds = codes[:, 0]
                right_motion = (
                    self.right_hand_model.decode(right_inds[0:1]).detach().cpu()
                )
                right_M = dset.toMotion(
                    right_motion[0],
                    motion_rep=MotionRep(self.right_cfg.dataset.motion_rep),
                    hml_rep=self.right_cfg.dataset.hml_rep,
                )
                return right_M

        if k == 2:
            left_inds = codes[:, 0]
            right_inds = codes[:, 1]

            left_motion = self.left_hand_model.decode(left_inds[0:1]).detach().cpu()
            right_motion = self.right_hand_model.decode(right_inds[0:1]).detach().cpu()

            left_M = dset.toMotion(
                left_motion[0],
                motion_rep=MotionRep(self.left_cfg.dataset.motion_rep),
                hml_rep=self.left_cfg.dataset.hml_rep,
            )
            right_M = dset.toMotion(
                right_motion[0],
                motion_rep=MotionRep(self.right_cfg.dataset.motion_rep),
                hml_rep=self.right_cfg.dataset.hml_rep,
            )
            hand_M = left_M + right_M
            hand_M.motion_rep = MotionRep.HAND
            hand_M.hml_rep = "".join(
                [i for i in left_M.hml_rep if i in right_M.hml_rep]
            )
            return hand_M

        if k == 3:
            left_inds = codes[:, 1]
            right_inds = codes[:, 2]
            body_inds = codes[:, 0]
            body_motion = self.body_model.decode(body_inds[0:1]).detach().cpu()

            if self.body_cfg.dataset.remove_translation:
                z = torch.zeros(
                    body_motion.shape[:-1] + (2,),
                    dtype=body_motion.dtype,
                    device=body_motion.device,
                )
                body_motion = torch.cat(
                    [body_motion[..., 0:1], z, body_motion[..., 1:]], -1
                )

            left_motion = self.left_hand_model.decode(left_inds[0:1]).detach().cpu()
            right_motion = self.right_hand_model.decode(right_inds[0:1]).detach().cpu()

            body_M = dset.toMotion(
                body_motion[0],
                motion_rep=MotionRep("body"),
                hml_rep=self.body_cfg.dataset.hml_rep,
            )

            left_M = dset.toMotion(
                left_motion[0],
                motion_rep=MotionRep("left_hand"),
                hml_rep=self.left_cfg.dataset.hml_rep,
            )
            right_M = dset.toMotion(
                right_motion[0],
                motion_rep=MotionRep("right_hand"),
                hml_rep=self.right_cfg.dataset.hml_rep,
            )
            full_M = dset.to_full_joint_representation(body_M, left_M, right_M)
            return full_M

    def sample_render_hmlvec(self, save_path):
        save_file = os.path.join(save_path, f"{int(self.steps.item())}")
        os.makedirs(save_file, exist_ok=True)

        # assert self.render_dl.batch_size == 1 , "Batch size for rendering should be 1!"

        dataset_lens = self.render_ds.cumulative_sizes
        self.motion_muse.eval()
        print(f"render start")
        with torch.no_grad():
            for idx, (inputs, conditions) in tqdm(
                enumerate(self.render_dl),
            ):
                name = str(inputs["names"][0])
                curr_dataset_idx = np.searchsorted(dataset_lens, idx + 1)
                dset = self.render_ds.datasets[curr_dataset_idx]

                inputs = self.to_device(inputs)
                conditions = self.to_device(conditions)

                motions = inputs["motion"][0].to(torch.long)  ###  1 k n
                motion_mask = inputs["motion"][1]
                gt_len = int(sum(motion_mask[0]))
                gt_len_s = round(
                    gt_len
                    / (self.dataset_args.fps / self.dataset_args.down_sampling_ratio)
                )

                gt_motion = self.bkn_to_motion(motions[..., :gt_len], dset)

                gen_ids = self.motion_muse.generate(conditions, duration_s=gt_len_s)

                gen_motion = self.bkn_to_motion(gen_ids, dset)

                dset.render_hml(
                    gt_motion,
                    os.path.join(
                        save_file, os.path.basename(name).split(".")[0] + "_gt.gif"
                    ),
                    # zero_trans=True,
                    # zero_orient=True,
                )

                dset.render_hml(
                    gen_motion,
                    os.path.join(
                        save_file, os.path.basename(name).split(".")[0] + "_gen.gif"
                    ),
                    # zero_trans=True,
                    # zero_orient=True,
                )

        self.motion_muse.train()

    def train(self, resume=False):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_dir = self.args.output_dir
            save_path = os.path.join(save_dir, "motion_muse.pt")
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()

        self.print("training complete")


# if __name__ == "__main__":
#     nme = "motion_translation"
#     path = f"/srv/hays-lab/scratch/sanisetty3/music_motion/motion_translation/checkpoints/{nme}/{nme}.yaml"
#     cfg = get_cfg_defaults()
#     print("loading config from:", path)
#     cfg.merge_from_file(path)
#     cfg.freeze()
#     print("output_dir: ", cfg.output_dir)

#     trainer = TranslationTransformerTrainer(
#         args=cfg,
#     ).cuda()

#     trainer.train(cfg.train.resume)
