"""
Default config
"""

# import argparse
# import yaml
import os
from glob import glob

from yacs.config import CfgNode as CN

cfg = CN()


cfg.abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg.device = "cuda"

cfg.vqvae_model_name = "vqvae"

cfg.motion_trans_model_name = "trans"
cfg.extractors_model_name = "aist_extractor_GRU"
cfg.bert_model_name = "bert"
cfg.pretrained_modelpath = os.path.join(
    cfg.abs_dir, f"checkpoints/{cfg.vqvae_model_name}/vqvae_motion.pt"
)
cfg.output_dir = os.path.join(cfg.abs_dir, "checkpoints/")

cfg.eval_output_dir = os.path.join(cfg.abs_dir, "eval/")

cfg.eval_model_path = os.path.join(
    cfg.abs_dir, f"checkpoints/{cfg.vqvae_model_name}/vqvae_motion.pt"
)


cfg.dataset = CN()
cfg.dataset.dataset_name = "t2m"  # "t2m or kit or aist or mix"
cfg.dataset.var_len = False
cfg.dataset.use_rotation = True
cfg.dataset.dataset_root = "/srv/hays-lab/scratch/sanisetty3/music_motion"
cfg.dataset.music_folder = "music"
cfg.dataset.fps = 30
cfg.dataset.enable_masking = False
cfg.dataset.train_mode = "full"

cfg.train = CN()
cfg.train.resume = True
cfg.train.seed = 42
cfg.train.fp16 = True
# cfg.train.train_mode = "full"  # body hand full

# cfg.train.output_dir = os.path.join(cfg.abs_dir , "checkpoints/")
cfg.train.num_stages = 6
cfg.train.num_train_iters = 500000  #'Number of training steps
cfg.train.save_steps = 5000
cfg.train.logging_steps = 10
cfg.train.wandb_every = 100
cfg.train.evaluate_every = 5000
cfg.train.eval_bs = 20
cfg.train.train_bs = 24
cfg.train.gradient_accumulation_steps = 4
cfg.train.bos_index = 1024
cfg.train.pad_index = 1025
cfg.train.eos_index = 1026
cfg.train.write_summary = True
cfg.train.log_dir = os.path.join(cfg.abs_dir, f"logs/{cfg.vqvae_model_name}")
cfg.train.max_grad_norm = 0.5

## optimization

cfg.train.learning_rate = 2e-4
cfg.train.weight_decay = 0.0
cfg.train.warmup_steps = 4000
cfg.train.gamma = 0.05
cfg.train.lr_scheduler_type = "cosine"
cfg.train.use_mixture = False

cfg.vqvae = CN()
cfg.vqvae.target = "core.models"

cfg.vqvae.nb_joints = 52
cfg.vqvae.nb_joints_hands = 30
cfg.vqvae.nb_joints_body = 22


cfg.vqvae.motion_dim_body = 263  #'Input motion dimension dimension'
cfg.vqvae.motion_dim_hand = 360  #'Input motion dimension dimension'
cfg.vqvae.motion_dim = 623  #'Input motion dimension dimension'

cfg.vqvae.dim_hands = 512
cfg.vqvae.dim_body = 512
cfg.vqvae.dim = 512

cfg.vqvae.depth_hand = 3
cfg.vqvae.depth_body = 3
cfg.vqvae.depth = 3
cfg.vqvae.dropout = 0.1
cfg.vqvae.down_sampling_ratio = 4
cfg.vqvae.conv_kernel_size = 5
cfg.vqvae.rearrange_output = False

cfg.vqvae.heads = 8
cfg.vqvae.codebook_dim = 768
cfg.vqvae.codebook_size = 1024
cfg.vqvae.codebook_dim_hands = 768
cfg.vqvae.codebook_size_hands = 1024
cfg.vqvae.codebook_dim_body = 768
cfg.vqvae.codebook_size_body = 1024

cfg.vqvae.num_quantizers = 2
cfg.vqvae.quantize_dropout_prob = 0.2
cfg.vqvae.shared_codebook = False
cfg.vqvae.sample_codebook_temp = 0.4
cfg.vqvae.freeze_model = False

## Loss
cfg.vqvae.commit = 1.0  # "hyper-parameter for the commitment loss"
cfg.vqvae.loss_vel = 1.0
cfg.vqvae.loss_motion = 1.0
cfg.vqvae.recons_loss = "l1_smooth"  # l1_smooth , l1 , l2
cfg.vqvae.window_size = 64
cfg.vqvae.max_length_seconds = 30
cfg.vqvae.min_length_seconds = 3

cfg.bert = CN()
cfg.bert.target = "core.models.motionBERT.motion_bert.BERTFORMER"
cfg.bert.mlm_probability = 0.15
cfg.bert.mask_span = 8
cfg.bert.max_motion_seconds = 8
cfg.bert.vqvae_downsample = 4
cfg.bert.dim = 512
cfg.bert.heads = 8
cfg.bert.depth = 6
cfg.bert.dropout = 0.1
cfg.bert.hidden_size = 768
cfg.bert.conv_kernel_size = 5

cfg.bert.vocab_size = 1027
cfg.bert.pad_token_id = 1024
cfg.bert.mask_token_id = 1026
cfg.bert.cls_token_id = 1025


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
