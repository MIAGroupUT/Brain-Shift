#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --output /home/imreb/brain-morphing/scripts/main_output.log
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --partition=mia,am
#SBATCH --error=/home/imreb/brain-morphing/scripts/err.err

cd /home/imreb/brain-morphing

module load nvidia/nvhpc/23.3

export HTTPS_PROXY=http://proxy.utwente.nl:3128
python -m src.main
