#!/bin/bash
#SBATCH --job-name=simple_motion_translation
#SBATCH --output=/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/sbach_outputs/simple_motion_translation.out
#SBATCH --error=/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/sbach_outputs/simple_motion_translation.err
#SBATCH --ntasks=1
#SBATCH -G a40:1
#SBATCH -p overcap
#SBATCH --qos debug
#SBATCH --cpus-per-task=15

cd /srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/
srun --exclusive --ntasks 1 -G 1 -c 15 /coc/flash5/sanisetty3/miniconda3/envs/tgm3d/bin/python -O /srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/train_translation.py
