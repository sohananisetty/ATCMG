#!/bin/bash
#SBATCH --job-name=smplx_resnet
#SBATCH --output=/srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/sbach_outputs2/smplx_resnet.out
#SBATCH --error=/srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/sbach_outputs2/smplx_resnet.err
#SBATCH --ntasks=1
#SBATCH -G a40:1
#SBATCH -p overcap
#SBATCH --qos short
#SBATCH --cpus-per-task=6

cd /srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/
srun --exclusive --ntasks 1 -G 1 -c 6 /coc/flash5/sanisetty3/miniconda3/envs/tgm3d/bin/python -O /srv/hays-lab/scratch/sanisetty3/music_motion/ACMG/train_vqvae.py
