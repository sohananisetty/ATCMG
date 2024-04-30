#!/bin/bash
#SBATCH --job-name=vqvae_full_gpvc
#SBATCH --output=/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/sbach_outputs/vqvae_full_gpvc.out
#SBATCH --error=/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/sbach_outputs/vqvae_full_gpvc.err
#SBATCH --ntasks=1
#SBATCH -G a40:1
#SBATCH -p overcap
#SBATCH --qos short
#SBATCH --cpus-per-task=15

cd /srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/
srun --exclusive --ntasks 1 -G 1 -c 15 /coc/flash5/sanisetty3/miniconda3/envs/tgm3d/bin/python -O /srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/train_vqvae.py
