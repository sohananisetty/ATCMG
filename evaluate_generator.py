from core.models.utils import instantiate_from_config
from core.datasets.tmr_dataset import load_dataset, simple_collate
from core import MotionRep, TextRep, AudioRep
from core.datasets.conditioner import ConditionProvider
from configs.config_tmr import get_cfg_defaults as get_cfg_defaults_tmr
from core.models.TMR.tmr import TMR
import torch
import numpy as np
import os
import torch.nn as nn
from functools import partial
from core import MotionTokenizerParams
from core.models.utils import instantiate_from_config
from core import MotionRep, AudioRep, TextRep
from core.datasets.conditioner import ConditionProvider
from core.models.generation.muse2 import MotionMuse
from core.eval.eval_text.eval_text import evaluation_transformer

import einops
from configs.config_t2m import get_cfg_defaults as muse_get_cfg_defaults

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


from configs.config import cfg, get_cfg_defaults


def load_generator(cfg_file):
    gen_cfg = muse_get_cfg_defaults()
    gen_cfg.merge_from_file(cfg_file)
    gen_cfg.freeze()
    tranformer_config = gen_cfg.motion_generator
    fuse_config = gen_cfg.fuser

    target = tranformer_config.pop("target")
    motion_gen = MotionMuse(tranformer_config, fuse_config).to(device).eval()

    pkg = torch.load(
        os.path.join(gen_cfg.output_dir, "motion_muse.pt"),
        map_location="cuda",
    )
    motion_gen.load_state_dict(pkg["model"])
    motion_gen = torch.compile(motion_gen)

    return motion_gen, gen_cfg


def load_vqvae(gen_cfg):

    body_cfg = get_cfg_defaults()
    body_cfg.merge_from_file(gen_cfg.vqvae.body_config)
    body_model = instantiate_from_config(body_cfg.vqvae).to(device).eval()
    body_model.load(os.path.join(body_cfg.output_dir, "vqvae_motion.pt"))

    if (
        gen_cfg.vqvae.left_hand_config is None
        and gen_cfg.vqvae.right_hand_config is None
    ):
        return body_model, body_cfg

    if gen_cfg.vqvae.left_hand_config is not None:
        left_cfg = get_cfg_defaults()
        left_cfg.merge_from_file(gen_cfg.vqvae.left_hand_config)
        left_hand_model = instantiate_from_config(left_cfg.vqvae).to(device).eval()
        left_hand_model.load(os.path.join(left_cfg.output_dir, "vqvae_motion.pt"))
    else:
        left_hand_model = None

    if gen_cfg.vqvae.right_hand_config is not None:
        right_cfg = get_cfg_defaults()
        right_cfg.merge_from_file(gen_cfg.vqvae.right_hand_config)
        right_hand_model = instantiate_from_config(right_cfg.vqvae).to(device).eval()
        right_hand_model.load(os.path.join(right_cfg.output_dir, "vqvae_motion.pt"))
    else:
        right_hand_model = None

    return body_model, left_hand_model, right_hand_model, body_cfg, left_cfg, right_cfg


def bkn_to_motion(codes, dset, body_model, body_cfg, remove_translation=True):
    # codes b k n

    k = codes.shape[1]
    mrep = dset.motion_rep

    body_inds = codes[:, 0]
    body_motion = body_model.decode(body_inds[0:1]).detach().cpu()

    if remove_translation:
        z = torch.zeros(
            body_motion.shape[:-1] + (2,),
            dtype=body_motion.dtype,
            device=body_motion.device,
        )
        body_motion = torch.cat([body_motion[..., 0:1], z, body_motion[..., 1:]], -1)

    body_M = dset.toMotion(
        body_motion[0],
        motion_rep=MotionRep("body"),
        hml_rep=body_cfg.dataset.hml_rep,
    )

    return body_M


if __name__ == "__main__":
    tmr, tmr_cfg = load_tmr(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/tmr/tmr.yaml"
    )

    motion_gen, gen_cfg = load_generator(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/motion_muse_body_hands/motion_muse_body_hands.yaml"
    )

    body_model, left_hand_model, right_hand_model, body_cfg, left_cfg, right_cfg = (
        load_vqvae(gen_cfg)
    )

    dataset_arg_gen = gen_cfg.dataset

    condition_provider_gen = ConditionProvider(
        text_conditioner_name=dataset_arg_gen.text_conditioner_name,
        motion_rep=MotionRep(dataset_arg_gen.motion_rep),
        audio_rep=AudioRep(dataset_arg_gen.audio_rep),
        text_rep=TextRep(dataset_arg_gen.text_rep),
        motion_padding=dataset_arg_gen.motion_padding,
        audio_padding=dataset_arg_gen.audio_padding,
        motion_max_length_s=dataset_arg_gen.motion_max_length_s,
        audio_max_length_s=dataset_arg_gen.audio_max_length_s,
        pad_id=MotionTokenizerParams(gen_cfg.motion_generator.num_tokens).pad_token_id,
        fps=30,
    )

    ds, _, _ = load_dataset(
        dataset_names=["humanml"], dataset_args=tmr_cfg.dataset, split="test"
    )
    data_loader = torch.utils.data.DataLoader(
        ds,
        256,
        collate_fn=partial(simple_collate, conditioner=condition_provider_gen),
        drop_last=True,
    )

    real_metrics, pred_metrics = evaluation_transformer(
        val_loader=data_loader,
        condition_provider=condition_provider_gen,
        bkn_to_motion=partial(bkn_to_motion, body_model=body_model, body_cfg=body_cfg),
        motion_generator=motion_gen,
        tmr=tmr,
        normalize=True,
    )

    msg = f"-->  FID. {pred_metrics[0]:.4f}, Diversity Real. {real_metrics[1]:.4f}, Diversity. {pred_metrics[1]:.4f}, R_precision_real. {real_metrics[2]}, R_precision. {pred_metrics[2]}, matching_score_real. {real_metrics[3]}, matching_score_pred. {pred_metrics[3]}"
