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

cfg.codebooks_pattern = CN()
cfg.codebooks_pattern.modeling = "delay"
cfg.codebooks_pattern.delay = [
    {
        "delays": [0, 1, 2, 3],
        "flatten_first": 0,
        "empty_initial": 0,
    }
]
cfg.codebooks_pattern.unroll = [
    {
        "flattening": [0, 1, 2, 3],
        "delays": [0, 0, 0, 0],
    }
]
cfg.codebooks_pattern.coarse_first = [{"delays": [0, 0, 0]}]


cfg.transformer_lm = CN()
cfg.transformer_lm.dim = 512
cfg.transformer_lm.num_heads = 8
cfg.transformer_lm.num_layers = 8
cfg.transformer_lm.hidden_scale = 4
cfg.transformer_lm.n_q = 3  # number of streams to model
cfg.transformer_lm.card = 1024
cfg.transformer_lm.dropout = 0.0
cfg.transformer_lm.emb_lr = None
cfg.transformer_lm.activation = "gelu"
cfg.transformer_lm.norm_first = False  # use pre-norm instead of post-norm
cfg.transformer_lm.bias_ff = True  # use bias for the feedforward
cfg.transformer_lm.bias_attn = True  # use bias for the attention
cfg.transformer_lm.bias_proj = True  # use bias for the output projections
cfg.transformer_lm.past_context = None
cfg.transformer_lm.causal = True
cfg.transformer_lm.custom = False  # use custom MHA implementation
cfg.transformer_lm.memory_efficient = False  # use flash attention
cfg.transformer_lm.attention_as_float32 = False  # use float32 for the attention part,
# recommended at the moment when memory_efficient is True.
cfg.transformer_lm.layer_scale = None
cfg.transformer_lm.positional_embedding = (
    "sin"  # positional embedding strategy (sin, rope, or sin_rope).
)
cfg.transformer_lm.xpos = False  # apply xpos decay (rope only).
cfg.transformer_lm.checkpointing = (
    None  # layer checkpointing method, can be None, torch, xformers_default.
)
# torch is the slowest but uses the least memory,
# xformers_default is somewhere in between.
cfg.transformer_lm.weight_init = (
    None  # weight initialization (None, gaussian or uniform)
)
cfg.transformer_lm.depthwise_init = (
    None  # perform depthwise initialization (None, current, global)
)
cfg.transformer_lm.zero_bias_init = (
    False  # initialize bias to zero if bias in linears and
)
# if a weight_init method is used.
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
