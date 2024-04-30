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
from yacs.config import CfgNode as CN
import einops
from configs.config_t2m import get_cfg_defaults as muse_get_cfg_defaults
from core.models.translation_model import Predictor2
from configs.config_t2o import get_cfg_defaults as get_cfg_defaults_trans
from configs.config import cfg, get_cfg_defaults

from core.datasets.base_dataset import BaseMotionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dset = BaseMotionDataset(motion_rep=MotionRep.BODY, hml_rep="gpvc")


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


def load_translation_model(cfg_file):
    trans_cfg = get_cfg_defaults_trans()
    trans_cfg.merge_from_file(cfg_file)
    trans_model = Predictor2().to(device).eval()
    trans_model.load(os.path.join(trans_cfg.output_dir, "tcn_model.pt"))

    return trans_model, trans_cfg


def load_refine_model(cfg_file):
    # pkg = torch.load(
    #     "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_full_gpvc/vqvae_motion_v2.pt"
    # )
    # refiner_cfg = CN(pkg["config"])
    # refiner = instantiate_from_config(refiner_cfg.vqvae).to(device).eval()
    # refiner.load(
    #     "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_full_gpvc/vqvae_motion_v2.pt"
    # )
    body_refiner_cfg = get_cfg_defaults()
    body_refiner_cfg.merge_from_file(cfg_file)

    body_refiner_model = (
        instantiate_from_config(body_refiner_cfg.vqvae).to(device).eval()
    )
    body_refiner_model.load(
        os.path.join(body_refiner_cfg.output_dir, "vqvae_motion.pt")
    )
    body_refiner_model.freeze()

    return body_refiner_model, body_refiner_cfg


def load_vqvae(gen_cfg):

    body_cfg = get_cfg_defaults()
    body_cfg.merge_from_file(gen_cfg.vqvae.body_config)
    body_model = instantiate_from_config(body_cfg.vqvae).to(device).eval()
    body_model.load(os.path.join(body_cfg.output_dir, "vqvae_motion.pt"))
    return body_model, body_cfg

    # if (
    #     gen_cfg.vqvae.left_hand_config is None
    #     and gen_cfg.vqvae.right_hand_config is None
    # ):
    #     return body_model, body_cfg

    # if gen_cfg.vqvae.left_hand_config is not None:
    #     left_cfg = get_cfg_defaults()
    #     left_cfg.merge_from_file(gen_cfg.vqvae.left_hand_config)
    #     left_hand_model = instantiate_from_config(left_cfg.vqvae).to(device).eval()
    #     left_hand_model.load(os.path.join(left_cfg.output_dir, "vqvae_motion.pt"))
    # else:
    #     left_hand_model = None

    # if gen_cfg.vqvae.right_hand_config is not None:
    #     right_cfg = get_cfg_defaults()
    #     right_cfg.merge_from_file(gen_cfg.vqvae.right_hand_config)
    #     right_hand_model = instantiate_from_config(right_cfg.vqvae).to(device).eval()
    #     right_hand_model.load(os.path.join(right_cfg.output_dir, "vqvae_motion.pt"))
    # else:
    #     right_hand_model = None

    # return body_model, left_hand_model, right_hand_model, body_cfg, left_cfg, right_cfg


@torch.no_grad()
def bkn_to_motion_add_translation(
    body_ids,
    body_model,
    body_cfg,
    body_refiner_model,
    trans_model,
    remove_translation=True,
):
    gen_motion = bkn_to_motion(
        body_ids,
        dset,
        body_model,
        body_cfg,
        remove_translation,
    )

    print(gen_motion().shape)
    params = torch.split(gen_motion(), [4, 63, 66, 4], -1)
    x = torch.cat([params[1], params[2][..., 3:]], -1).reshape(1, -1, 21, 6)
    out = trans_model(x.to(device))

    out[..., -4:] = torch.round(out[..., -4:])
    out = out.squeeze()
    pred_motion = gen_motion().clone()
    pred_motion[..., 1:3] = out[..., 1:3]
    refined = body_refiner_model(pred_motion[None].to(device)).decoded_motion

    refined_M = dset.toMotion(
        refined[0],
        motion_rep=MotionRep("body"),
        hml_rep=body_cfg.dataset.hml_rep,
    )

    return refined_M


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

    trans_model, trans_cfg = load_translation_model(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/simple_motion_translation/simple_motion_translation.yaml"
    )

    refiner, refiner_cfg = load_refine_model(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_body_gpvc_5121/vqvae_body_gpvc_5121.yaml"
    )

    body_model, body_cfg = load_vqvae(gen_cfg)

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
        dataset_names=["moyo"], dataset_args=tmr_cfg.dataset, split="train"
    )
    data_loader = torch.utils.data.DataLoader(
        ds,
        4,
        collate_fn=partial(simple_collate, conditioner=condition_provider_gen),
        drop_last=True,
    )

    real_metrics, pred_metrics = evaluation_transformer(
        val_loader=data_loader,
        condition_provider=condition_provider_gen,
        bkn_to_motion=partial(
            bkn_to_motion_add_translation,
            body_model=body_model,
            body_cfg=body_cfg,
            body_refiner_model=refiner,
            trans_model=trans_model,
        ),
        motion_generator=motion_gen,
        tmr=tmr,
        normalize=True,
    )

    msg = f"-->  FID. {pred_metrics[0]:.4f}, Diversity Real. {real_metrics[1]:.4f}, Diversity. {pred_metrics[1]:.4f}, R_precision_real. {real_metrics[2]}, R_precision. {pred_metrics[2]}, matching_score_real. {real_metrics[3]}, matching_score_pred. {pred_metrics[3]}"
