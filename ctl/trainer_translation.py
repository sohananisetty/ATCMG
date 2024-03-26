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
from core.datasets.conditioner import ConditionProvider
from core.optimizer import get_optimizer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from utils.motion_processing.hml_process import recover_from_ric, recover_root_rot_pos
from yacs.config import CfgNode
from core.models.generation.translation_transformer import TranslationTransformer
from core import AudioRep, MotionRep, MotionTokenizerParams, TextRep
from core.datasets.translation_dataset import (
    TranslationAudioTextDataset,
    load_dataset,
    simple_collate,
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


# main trainer class


class TranslationTransformerTrainer(nn.Module):
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
        self.model_args = args.translation_transformer

        self.num_train_steps = self.training_args.num_train_iters
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        target = self.model_args.pop("target")
        fuse_config = self.args.fuser

        self.translation_transformer = TranslationTransformer(
            self.model_args, fuse_config
        ).to(self.device)

        self.register_buffer("steps", torch.Tensor([0]))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        self.optim = get_optimizer(
            self.translation_transformer.parameters(),
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        dataset_names = {
            "animation": 0.8,
            "humanml": 3.5,
            "perform": 0.6,
            "GRAB": 1.0,
            "idea400": 1.5,
            "humman": 0.5,
            "beat": 2.0,
            "game_motion": 0.8,
            "music": 0.5,
            "aist": 1.5,
            "fitness": 1.0,
            "moyo": 1.0,
            "choreomaster": 2.0,
            "dance": 1.0,
            "kungfu": 1.0,
            "EgoBody": 0.5,
            # "HAA500": 1.0,
        }

        train_ds, sampler_train, weights_train = load_dataset(
            dataset_args=self.dataset_args,
            split="train",
            dataset_names=list(dataset_names.keys()),
            weight_scale=list(dataset_names.values()),
        )
        test_ds, _, _ = load_dataset(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="test",
        )

        self.render_ds, _, _ = load_dataset(
            dataset_names=list(dataset_names.keys()),
            dataset_args=self.dataset_args,
            split="render",
        )
        self.print(
            f"training with training {len(train_ds)} and test dataset of and {len(test_ds)} samples"
        )

        # dataloader

        condition_provider = ConditionProvider(
            text_conditioner_name=self.dataset_args.text_conditioner_name,
            motion_rep=MotionRep(self.dataset_args.motion_rep),
            audio_rep=AudioRep(self.dataset_args.audio_rep),
            text_rep=TextRep(self.dataset_args.text_rep),
            motion_padding=self.dataset_args.motion_padding,
            audio_padding=self.dataset_args.audio_padding,
            motion_max_length_s=self.dataset_args.motion_max_length_s,
            audio_max_length_s=self.dataset_args.motion_max_length_s,
            fps=self.dataset_args.fps,
        )

        self.dl = DataLoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=partial(simple_collate, conditioner=condition_provider),
            # pin_memory=False,
            # num_workers=2,
        )
        self.valid_dl = DataLoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=partial(simple_collate, conditioner=condition_provider),
            # pin_memory=False,
            # num_workers=2,
        )

        self.render_dl = DataLoader(
            self.render_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(simple_collate, conditioner=condition_provider),
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

    def print(self, msg):
        # self.accelerator.print(msg)
        print(msg)

    @property
    def device(self):
        return torch.device("cuda")

    def save(self, path, loss=None):
        pkg = dict(
            model=self.translation_transformer.state_dict(),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cuda")
        self.translation_transformer.load_state_dict(pkg["model"])

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

        self.translation_transformer = self.translation_transformer.train()

        # logs

        logs = {}

        for _ in range(self.grad_accum_every):
            inputs, conditions = next(self.dl_iter)

            inputs = self.to_device(inputs)
            conditions = self.to_device(conditions)

            loss, pred_orient = self.translation_transformer(
                inputs["motion"], conditions
            )
            loss = loss / self.grad_accum_every

            loss.backward()

            accum_log(
                logs,
                dict(
                    loss=loss.detach().cpu(),
                ),
            )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: model total loss: {logs['loss'].float():.3}"

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"translation_transformer.{steps}.pt"
            )
            self.save(model_path)
            print(float(logs["loss"]), self.best_loss)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(
                    self.output_dir, f"translation_transformer.pt"
                )
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        if steps % self.evaluate_every == 0:
            self.validation_step()
            # self.sample_render_hmlvec(os.path.join(self.output_dir, "samples"))

        self.steps += 1
        return logs

    def validation_step(self):
        self.translation_transformer.eval()
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

                loss, pred_orient = self.translation_transformer(
                    inputs["motion"], conditions
                )

                loss_dict["loss"] = loss.detach().cpu()

                for key, value in loss_dict.items():
                    if key in val_loss_ae:
                        val_loss_ae[key] += value
                    else:
                        val_loss_ae[key] = value

                    cnt += 1

        for key in val_loss_ae.keys():
            val_loss_ae[key] = val_loss_ae[key] / cnt

        for key, value in val_loss_ae.items():
            wandb.log({f"val_loss/{key}": value})
            print(f"val_loss/{key}", value)

        self.translation_transformer.train()

    def sample_render_hmlvec(self, save_path):
        save_file = os.path.join(save_path, f"{int(self.steps.item())}")
        os.makedirs(save_file, exist_ok=True)

        # assert self.render_dl.batch_size == 1 , "Batch size for rendering should be 1!"

        dataset_lens = self.render_ds.cumulative_sizes
        self.translation_transformer.eval()
        print(f"render start")
        with torch.no_grad():
            for idx, (inputs, conditions) in tqdm(
                enumerate(self.render_dl),
            ):
                name = str(inputs["names"][0])
                curr_dataset_idx = np.searchsorted(dataset_lens, idx + 1)

                inputs = self.to_device(inputs)
                conditions = self.to_device(conditions)

                output = self.translation_transformer(inputs, conditions)

                gt_motion = inputs["motion"][0].clone()
                gt_mask = inputs["motion"][1].clone().cpu()
                motion_len = min(sum(gt_mask[0]), 300)
                pred_motion = inputs["motion"][0].clone()
                pred_motion[..., :4] = output.pred_motion

                gt_motion = (
                    self.render_ds.datasets[curr_dataset_idx]
                    .inv_transform(gt_motion.cpu())
                    .squeeze()
                    .float()
                )
                pred_motion = (
                    self.render_ds.datasets[curr_dataset_idx]
                    .inv_transform(pred_motion.cpu())
                    .squeeze()
                    .float()
                )

                gt_motion_xyz = recover_from_ric(gt_motion, self.nb_joints)
                pred_motion_xyz = recover_from_ric(pred_motion, self.nb_joints)

                plot_3d.render(
                    gt_motion_xyz.numpy().squeeze()[:motion_len],
                    os.path.join(
                        save_file, os.path.basename(name).split(".")[0] + "_gt.gif"
                    ),
                )
                plot_3d.render(
                    pred_motion_xyz.numpy().squeeze()[:motion_len],
                    os.path.join(
                        save_file, os.path.basename(name).split(".")[0] + "_pred.gif"
                    ),
                )

        self.translation_transformer.train()

    def train(self, resume=False):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_dir = self.args.output_dir
            save_path = os.path.join(save_dir, "translation_transformer.pt")
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()

        self.print("training complete")
