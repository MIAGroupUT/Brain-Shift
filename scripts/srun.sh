srun -c 16  --time=24:00:00 --partition=mia,am --gres=gpu:1 --pty bash -i
conda activate brain-shift