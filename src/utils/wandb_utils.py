import torch
import wandb
from matplotlib import pyplot as plt
from src.constants import *


def experiment_init(r_name=None):

    if r_name is None:
        r_name = RUN_NAME

    wandb.init(
        project="brain-shift",
        entity="barisimre",
        name=r_name
    )
    wandb.log({"experiment variables": open(f"{GENERAL_PATH}/src/experiment_variables.py", "r").read()})


def detailed_plot_from3d_to_wandb(img, slice=None, line=True, iteration="unnamed", loss=0.0):
    if slice is None:
        slice = int(min(torch.tensor(img.shape[-3:]) / 2))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    if len(img.shape) == 5:
        img = img[0, 0]
    elif len(img.shape) == 4:
        img = img[0]

    img = img.cpu().detach()

    slices = [torch.rot90(img[slice, :, :]), img[:, :, slice]]
    titles = ['X-axis slice', 'Z-axis slice']

    for i in range(2):
        axs[i].imshow(slices[i], cmap="gray")
        if line:
            axs[i].axvline(x=slices[i].shape[1] // 2)

        axs[i].set_title(titles[i])

    fig.suptitle(f"Iteration: {iteration}, Loss: {loss:.3f}")

    wandb.log({"Morphed": fig})
    plt.close()
