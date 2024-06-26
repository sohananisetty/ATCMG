#!/bin/bash
#SBATCH --job-name=tar
#SBATCH --output=/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/sbach_outputs/tar.out
#SBATCH --error=/srv/hays-lab/scratch/sanisetty3/music_motion/ATCMG/sbach_outputs/tar.err
#SBATCH --ntasks=1
#SBATCH -p overcap
#SBATCH --qos debug
#SBATCH --cpus-per-task=7

cd /srv/hays-lab/scratch/sanisetty3/motionx/motion_data/audio/wav
tar -czf /srv/hays-lab/flash5/sanisetty/wav.tar.gz *
# tar -cf /srv/hays-lab/flash5/sanisetty/texts.tar /srv/hays-lab/scratch/sanisetty3/motionx/texts/
# tar -cf /srv/hays-lab/flash5/sanisetty/wav.tar /srv/hays-lab/scratch/sanisetty3/motionx/audio/wav/