from core.models.utils import instantiate_from_config, get_obj_from_str
from core.datasets.tmr_dataset import TMRDataset, load_dataset, simple_collate
from core import MotionRep, TextRep, AudioRep
from core.datasets.conditioner import ConditionProvider, ConditionFuser
from configs.config_tmr import get_cfg_defaults as get_cfg_defaults_tmr
from core.models.TMR.tmr import TMR
from core.models.TMR.tmr import get_score_matrix
import torch
import numpy as np
import os
import torch.nn as nn
from tqdm import tqdm
import json
from functools import partial
from torch import einsum, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import pack, rearrange, reduce, repeat, unpack
from functools import partial
from configs.config import get_cfg_defaults as vqvae_get_cfg_defaults
from core.eval.eval_text.eval_text import evaluation_vqvae

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)


def load_tmr(cfg_file):
    tmr_cfg = get_cfg_defaults_tmr()
    tmr_cfg.merge_from_file(cfg_file)
    tmr_cfg.freeze()
    tmr_parms = tmr_cfg.tmr

    _ = tmr_parms.pop("target")
    motion_encoder = instantiate_from_config(tmr_cfg.motion_encoder).to(device).eval()
    text_encoder = instantiate_from_config(tmr_cfg.text_encoder).to(device).eval()
    motion_decoder = instantiate_from_config(tmr_cfg.motion_decoder).to(device).eval()
    tmr = (
        TMR(
            motion_encoder,
            text_encoder,
            motion_decoder,
            lr=tmr_cfg.train.learning_rate,
            **tmr_parms,
        )
        .to(device)
        .eval()
    )
    pkg = torch.load(os.path.join(tmr_cfg.output_dir, "tmr.pt"))
    tmr.load_state_dict(pkg["model"])

    return tmr, tmr_cfg


def load_vqvae(cfg_file):
    vqvae_config = vqvae_get_cfg_defaults()
    vqvae_config.freeze()
    vqvae_config.merge_from_file(cfg_file)
    vqvae_model = instantiate_from_config(vqvae_config.vqvae).to(device).eval()
    vqvae_model.load(os.path.join(vqvae_config.output_dir, "vqvae_motion.pt"))
    # vqvae_model.load(
    #     "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_body_gpvc_512/checkpoints/vqvae_motion.70000_old copy.pt"
    # )

    return vqvae_model, vqvae_config


if __name__ == "__main__":
    tmr, tmr_cfg = load_tmr(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/tmr/tmr.yaml"
    )
    vqvae_model, vqvae_config = load_vqvae(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_body_gpvc_512/vqvae_body_gpvc_512.yaml"
    )

    dataset_args = tmr_cfg.dataset

    condition_provider = ConditionProvider(
        text_conditioner_name=dataset_args.text_conditioner_name,
        motion_rep=MotionRep(dataset_args.motion_rep),
        audio_rep=AudioRep(dataset_args.audio_rep),
        text_rep=TextRep(dataset_args.text_rep),
        motion_padding=dataset_args.motion_padding,
        motion_max_length_s=dataset_args.motion_max_length_s,
        fps=30,
    )
    condition_provider.audio_dim = 128

    ds, _, _ = load_dataset(dataset_args=dataset_args, split="test")
    data_loader = torch.utils.data.DataLoader(
        ds,
        256,
        collate_fn=partial(simple_collate, conditioner=condition_provider),
        drop_last=True,
    )

    real_metrics, pred_metrics = evaluation_vqvae(
        val_loader=data_loader, motion_vqvae=vqvae_model, tmr=tmr, normalize=True
    )

    msg = f"-->  FID. {pred_metrics[0]:.4f}, Diversity Real. {real_metrics[1]:.4f}, Diversity. {pred_metrics[1]:.4f}, R_precision_real. {real_metrics[2]}, R_precision. {pred_metrics[2]}, matching_score_real. {real_metrics[3]}, matching_score_pred. {pred_metrics[3]}"
