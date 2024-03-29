import itertools
import os
import random
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import transformers
import utils.vis_utils.plot_3d_global as plot_3d
import wandb
from core.datasets.dataset_loading_utils import load_dataset
from core.datasets.motion_bert_dataset import (DATALoader, MotionBERTDataset,
                                               get_mlm_mask)
from core.models.loss import ReConsLoss
from core.models.resnetVQ.vqvae import HumanVQVAE
from core.models.utils import instantiate_from_config
from core.optimizer import get_optimizer
from torch import nn
from tqdm import tqdm
from transformers import AdamW, get_scheduler
from utils.motion_processing.hml_process import recover_from_ric
from yacs.config import CfgNode


def exists(val):
    return val is not None


def noop(*args, **kwargs):
    pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


# auto data to module keyword argument routing functions


def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


# main trainer class


class MotionBERTrainer(nn.Module):
    def __init__(
        self,
        args: CfgNode,
    ):
        super().__init__()
        self.model_name = args.bert_model_name

        transformers.set_seed(42)

        self.args = args
        self.bert_args = args.bert
        self.training_args = args.train
        self.dataset_args = args.dataset
        self.dataset_name = args.dataset.dataset_name
        self.num_train_steps = self.training_args.num_train_iters
        self.num_stages = self.training_args.num_stages
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.register_buffer("steps", torch.Tensor([0]))
        print(self.bert_args)

        self.bert_model = instantiate_from_config(self.bert_args).to(self.device)
        # self.vqvae_model = (
        #     instantiate_from_config(self.args.vqvae_cfg.vqvae).to(self.device).eval()
        # )
        # self.vqvaae_model.load(
        #     os.path.join(self.args.vqvae_cfg.output_dir, "vqvae_motion.pt")
        # )

        total = sum(p.numel() for p in self.bert_model.parameters() if p.requires_grad)
        print("Total training params: %.2fM" % (total / 1e6))

        self.grad_accum_every = self.training_args.gradient_accumulation_steps

        # self.loss_fnc = ReConsLoss(
        #     self.bert_args.recons_loss,
        #     self.bert_args.nb_joints,
        # )

        self.optim = get_optimizer(
            self.bert_model.parameters(),
            lr=self.training_args.learning_rate,
            wd=self.training_args.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=self.num_train_steps,
        )

        self.max_grad_norm = self.training_args.max_grad_norm

        train_ds, sampler_train, weights_train = load_dataset(
            dataset_args=self.dataset_args,
            model_args=self.bert_args,
            split="train",
            dataset_class=MotionBERTDataset,
        )
        test_ds, _, _ = load_dataset(
            dataset_args=self.dataset_args,
            model_args=self.bert_args,
            split="test",
            dataset_class=MotionBERTDataset,
        )
        self.render_ds, _, _ = load_dataset(
            dataset_args=self.dataset_args,
            model_args=self.bert_args,
            split="render",
            dataset_class=MotionBERTDataset,
        )

        # if self.is_main:
        self.print(
            f"training with training {len(train_ds)} and test dataset of  and  {len(test_ds)} samples and render of  {len(self.render_ds)}"
        )

        # dataloader

        self.dl = DATALoader(
            train_ds,
            batch_size=self.training_args.train_bs,
            sampler=sampler_train,
            shuffle=False if sampler_train else True,
            collate_fn=None,
        )
        self.valid_dl = DATALoader(
            test_ds,
            batch_size=self.training_args.eval_bs,
            shuffle=False,
            collate_fn=None,
        )
        self.render_dl = DATALoader(
            self.render_ds, batch_size=1, shuffle=False, collate_fn=None
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
        print(msg)

    @property
    def device(self):
        return torch.device("cuda")

    def save(self, path, loss=None):
        pkg = dict(
            model=self.bert_model.state_dict(),
            optim=self.optim.state_dict(),
            steps=self.steps,
            total_loss=self.best_loss if loss is None else loss,
        )

        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()

        pkg = torch.load(str(path), map_location="cuda")
        self.bert_model.load_state_dict(pkg["model"])

        self.optim.load_state_dict(pkg["optim"])
        self.steps = pkg["steps"]
        self.best_loss = pkg["total_loss"]

    def train_step(self):
        steps = int(self.steps.item())
        self.bert_model = self.bert_model.train()

        # logs

        logs = {}

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = input_ids.clone()
            input_ids, labels = get_mlm_mask(
                tokenizer=self.bert_model.tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mlm_probability=self.bert_args.mlm_probability,
                mask_span=self.bert_args.mask_span,
                targets=labels,
            )

            bert_output = self.bert_model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                labels=labels.to(self.device),
            )

            loss = bert_output.loss / self.grad_accum_every

            correct = (
                bert_output.logits.cpu().argmax(-1)[batch["input_ids"] != -100]
                == batch["input_ids"][batch["input_ids"] != -100]
            )
            correct_percent = correct.sum() / len(correct)
            loss.backward()

            accum_log(
                logs,
                dict(
                    loss=loss.detach().cpu(),
                    correct_percent=correct_percent / self.grad_accum_every,
                ),
            )

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()

        # build pretty printed losses

        losses_str = f"{steps}: bert model total loss: {logs['loss'].float():.3} correct_percent: {logs['correct_percent']}"

        # log
        if steps % self.wandb_every == 0:
            for key, value in logs.items():
                wandb.log({f"train_loss/{key}": value})

            self.print(losses_str)

        if steps % self.evaluate_every == 0:
            self.validation_step()
            # self.sample_render_hmlvec(os.path.join(self.output_dir, "samples"))

        # save model

        # if self.is_main and not (steps % self.save_model_every) and steps > 0:
        if not (steps % self.save_model_every):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            model_path = os.path.join(
                self.output_dir, "checkpoints", f"bert_motion.{steps}.pt"
            )
            self.save(model_path)
            print(float(logs["loss"]), self.best_loss)

            if float(logs["loss"]) <= self.best_loss:
                model_path = os.path.join(self.output_dir, f"bert_motion.pt")
                self.save(model_path)
                self.best_loss = logs["loss"]

            self.print(
                f'{steps}: saving model to {str(os.path.join(self.output_dir , "checkpoints") )}'
            )

        self.steps += 1
        return logs

    def validation_step(self):
        self.bert_model.eval()
        val_loss_ae = {}
        all_loss = 0.0

        self.print(f"validation start")

        with torch.no_grad():
            for batch in tqdm(
                (self.valid_dl),
                position=0,
                leave=True,
            ):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = input_ids.clone()
                input_ids, labels = get_mlm_mask(
                    tokenizer=self.bert_model.tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mlm_probability=self.bert_args.mlm_probability,
                    mask_span=self.bert_args.mask_span,
                    targets=labels,
                )

                bert_output = self.bert_model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    labels=labels.to(self.device),
                )

                loss = bert_output.loss / self.grad_accum_every

                correct = (
                    bert_output.logits.cpu().argmax(-1)[batch["input_ids"] != -100]
                    == batch["input_ids"][batch["input_ids"] != -100]
                )
                correct_percent = correct.sum() / len(correct)

                loss_dict = {
                    "total_loss": loss.detach().cpu(),
                    "correct_percent": correct_percent,
                }

                val_loss_ae.update(loss_dict)

                sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
                means_ae = {
                    k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict))
                    for k in sums_ae
                }
                val_loss_ae.update(means_ae)

        for key, value in val_loss_ae.items():
            wandb.log({f"val_loss_bert/{key}": value})

        print(
            "val/correct_percent",
            val_loss_ae["correct_percent"],
        )
        print(
            f"val/total_loss ",
            val_loss_ae["total_loss"],
        )

        self.bert_model.train()

    # def sample_render_hmlvec(self, save_path):
    #     save_file = os.path.join(save_path, f"{int(self.steps.item())}")
    #     os.makedirs(save_file, exist_ok=True)

    #     dataset_lens = self.render_ds.cumulative_sizes
    #     self.bert_model.eval()
    #     print(f"render start")
    #     with torch.no_grad():
    #         for idx, batch in tqdm(
    #             enumerate(self.render_dl),
    #         ):
    #             name = str(batch["names"][0])
    #             curr_dataset_idx = np.searchsorted(dataset_lens, idx + 1)

    #             input_ids = batch["input_ids"].to(self.device)
    #             attention_mask = batch["attention_mask"].to(self.device)
    #             labels = input_ids.clone()
    #             input_ids, labels = get_mlm_mask(
    #                 tokenizer=self.bert_model.tokenizer,
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 mlm_probability=self.bert_args.mlm_probability,
    #                 mask_span=self.bert_args.mask_span,
    #                 targets=labels,
    #             )

    #             bert_output = self.bert_model(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 labels=labels,
    #             )

    #             loss = bert_output.loss / self.grad_accum_every

    #             correct = (
    #                 bert_output.logits.cpu().argmax(-1)[batch["input_ids"] != -100]
    #                 == batch["input_ids"][batch["input_ids"] != -100]
    #             )

    #             gt_motion = (
    #                 self.render_ds.datasets[curr_dataset_idx]
    #                 .inv_transform(gt_motion.cpu())
    #                 .squeeze()
    #                 .float()
    #             )
    #             pred_motion = (
    #                 self.render_ds.datasets[curr_dataset_idx]
    #                 .inv_transform(pred_motion.cpu())
    #                 .squeeze()
    #                 .float()
    #             )

    #             gt_motion_xyz = recover_from_ric(gt_motion, self.bert_args.nb_joints)
    #             pred_motion_xyz = recover_from_ric(
    #                 pred_motion, self.bert_args.nb_joints
    #             )

    #             plot_3d.render(
    #                 gt_motion_xyz.numpy().squeeze(),
    #                 os.path.join(
    #                     save_file, os.path.basename(name).split(".")[0] + "_gt.gif"
    #                 ),
    #             )
    #             plot_3d.render(
    #                 pred_motion_xyz.numpy().squeeze(),
    #                 os.path.join(
    #                     save_file, os.path.basename(name).split(".")[0] + "_pred.gif"
    #                 ),
    #             )

    #     self.bert_model.train()

    def train(self, resume=False, log_fn=noop):
        self.best_loss = float("inf")
        print(self.output_dir)

        if resume:
            save_dir = self.args.output_dir
            save_path = os.path.join(save_dir, "bert_motion.pt")
            print("resuming from ", save_path)
            self.load(save_path)

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")
