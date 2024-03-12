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

cfg.model_name = "ttmodel"

cfg.output_dir = os.path.join(cfg.abs_dir, "checkpoints/")


cfg.dataset = CN()
cfg.dataset.dataset_name = "mix"
cfg.dataset.dataset_root = "/srv/hays-lab/scratch/sanisetty3/music_motion"
cfg.dataset.music_folder = "music"
cfg.dataset.fps = 30
cfg.dataset.down_sampling_ratio = 4

cfg.dataset.text_rep = "pooled_text_embed"
cfg.dataset.motion_rep = "full"
cfg.dataset.hml_rep = "gprvc"  ## global pos rot6d vel contact
cfg.dataset.audio_rep = "encodec"
cfg.dataset.motion_min_length_s = 2
cfg.dataset.motion_max_length_s = 10
cfg.dataset.audio_max_length_s = 10
cfg.dataset.window_size = None
cfg.dataset.sampling_rate = 16000
cfg.dataset.text_conditioner_name = "t5-base"
cfg.dataset.audio_padding = "longest"
cfg.dataset.motion_padding = "longest"


cfg.train = CN()
cfg.train.resume = True
cfg.train.seed = 42

cfg.train.num_train_iters = 500000  #'Number of training steps
cfg.train.save_steps = 5000
cfg.train.logging_steps = 10
cfg.train.wandb_every = 100
cfg.train.evaluate_every = 5000
cfg.train.eval_bs = 20
cfg.train.train_bs = 24
cfg.train.gradient_accumulation_steps = 4
cfg.train.log_dir = os.path.join(cfg.abs_dir, f"logs/{cfg.model_name}")
cfg.train.max_grad_norm = 0.5

## optimization

cfg.train.learning_rate = 2e-4
cfg.train.weight_decay = 0.0
cfg.train.warmup_steps = 4000
cfg.train.gamma = 0.05
cfg.train.lr_scheduler_type = "cosine"

##optimizations
cfg.train.autocast = True
cfg.train.autocast_dtype = "float16"
## Loss
cfg.train.hand_loss = 0.6
cfg.train.body_loss = 1.2


cfg.codebooks_pattern = CN()
cfg.codebooks_pattern.modeling = "delay"  ## delay parralel unroll coarse_first
cfg.codebooks_pattern.delays = [0, 1, 2]
cfg.codebooks_pattern.flatten_first = 0
cfg.codebooks_pattern.empty_initial = 0
cfg.codebooks_pattern.flattening = [0, 1, 2]

cfg.fuser = CN()
cfg.fuser.fuse_method = [{"cross": ["audio"], "prepend": ["text"]}]
cfg.fuser.cross_attention_pos_emb = False
cfg.fuser.cross_attention_pos_emb_scale = 1.0

cfg.vqvae = CN()
cfg.vqvae.body_config = "/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/vqvae/vqvae_rv/vqvae_rv.yaml"
cfg.vqvae.left_hand_config = "/srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/checkpoints/smplx_resnet_left/smplx_resnet_left.yaml"
cfg.vqvae.right_hand_config = "/srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/checkpoints/smplx_resnet_right/smplx_resnet_right.yaml"

cfg.transformer_lm = CN()
cfg.transformer_lm.target = "core.models.generation.lm.MotionGen"
cfg.transformer_lm.dim = 512

cfg.transformer_lm.proj_input = False
cfg.transformer_lm.audio_input_dim = 128
cfg.transformer_lm.text_input_dim = 768
cfg.transformer_lm.num_heads = 8
cfg.transformer_lm.num_layers = 8
cfg.transformer_lm.hidden_scale = 4


cfg.transformer_lm.n_q = 3  # number of streams to model
cfg.transformer_lm.card = 1024
cfg.transformer_lm.dropout = 0.0  ## Dropout probability on attn_output_weights
cfg.transformer_lm.emb_lr = None
cfg.transformer_lm.activation = "gelu"
cfg.transformer_lm.norm_first = True  # use pre-norm instead of post-norm
cfg.transformer_lm.bias_ff = True  # use bias for the feedforward
cfg.transformer_lm.bias_attn = True  # use bias for the attention
cfg.transformer_lm.bias_proj = True  # use bias for the output projections
cfg.transformer_lm.past_context = None
cfg.transformer_lm.causal = True
cfg.transformer_lm.add_null_kv = False


## CFG
cfg.transformer_lm.cfg_dropout = 0.0
cfg.transformer_lm.cfg_coef = 0.3
## Optimizations
cfg.transformer_lm.custom = False  # use custom MHA implementation
cfg.transformer_lm.memory_efficient = False  # use flash attention
cfg.transformer_lm.attention_as_float32 = False  # use float32 for the attention part,
# recommended at the moment when memory_efficient is True.
cfg.transformer_lm.layer_scale = None
# positional embedding strategy (sin, rope, or sin_rope).
cfg.transformer_lm.positional_embedding = "sin"
cfg.transformer_lm.xpos = False  # apply xpos decay (rope only).
# layer checkpointing method, can be none, torch, xformers_default.
cfg.transformer_lm.checkpointing = "none"
# torch is the slowest but uses the least memory,
# xformers_default is somewhere in between.


##Initilizations
# weight initialization (none, gaussian or uniform)
cfg.transformer_lm.weight_init = "none"
# perform depthwise initialization (none, current, global)
cfg.transformer_lm.depthwise_init = "none"
# initialize bias to zero if bias in linears and
cfg.transformer_lm.zero_bias_init = False
# if a weight_init method is used.


## Normalizations
cfg.transformer_lm.norm = "layer_norm"  # normalization method to use in transformer.
cfg.transformer_lm.cross_attention = True
cfg.transformer_lm.qk_layer_norm = False
cfg.transformer_lm.qk_layer_norm_cross = False
cfg.transformer_lm.attention_dropout = None
cfg.transformer_lm.kv_repeat = 1
cfg.transformer_lm.two_step_cfg = False  # whether to do True 2 steps CFG, potentially resolving some padding issues or not...


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
