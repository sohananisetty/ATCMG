import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from configs.config_t2m import get_cfg_defaults
from ctl.trainer_muse import MotionMuseTrainer

if __name__ == "__main__":
    nme = "motion_muse_full"
    path = f"/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/checkpoints/{nme}/{nme}.yaml"
    cfg = get_cfg_defaults()
    print("loading config from:", path)
    cfg.merge_from_file(path)
    cfg.freeze()
    print("output_dir: ", cfg.output_dir)

    trainer = MotionMuseTrainer(
        args=cfg,
    ).cuda()

    trainer.train(cfg.train.resume)


# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py
# accelerate launch --mixed_precision=fp16 --num_processes=1 train_vqvae.py


# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml

# conformer_512_1024_affine
# convq_256_1024_affine
# salloc -p overcap -G 2080_ti:1 --qos debug
