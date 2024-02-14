#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --output /home/imreb/brain-morphing/scripts/main_output.log
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --partition=mia,am
#SBATCH --error=/home/imreb/brain-morphing/scripts/err_infer.err

cd /home/imreb/brain-morphing

export HTTPS_PROXY=http://proxy.utwente.nl:3128
python -m src.infer
