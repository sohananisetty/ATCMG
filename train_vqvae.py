from configs.config import cfg, get_cfg_defaults
from ctl.trainer_vq_simple import VQVAEMotionTrainer as VQVAEMotionTrainerSimple


def main(cfg):
    trainer = VQVAEMotionTrainerSimple(
        args=cfg,
    ).cuda()

    trainer.train(cfg.train.resume)


if __name__ == "__main__":
    nme = "smplx_resnet"
    path = f"/srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/checkpoints/{nme}/{nme}.yaml"
    cfg = get_cfg_defaults()
    print("loading config from:", path)
    cfg.merge_from_file(path)
    cfg.freeze()
    print("output_dir: ", cfg.output_dir)

    main(cfg)


# accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 train.py
# accelerate launch --mixed_precision=fp16 --num_processes=1 train_vqvae.py


# accelerate configuration saved at /nethome/sanisetty3/.cache/huggingface/accelerate/default_config.yaml

# conformer_512_1024_affine
# convq_256_1024_affine
# salloc -p overcap -G 2080_ti:1 --qos debug
