import torch
from torch.utils.data import DataLoader
import wandb
from src.data_loading.datasets import AllBidsDataset, Dataset3D, AnnotatedBidsDataset, HDF5Dataset
from src.losses.losses import jeffreys_divergence_loss, ssim_loss, pixel_loss
from src.utils.movement import translate_and_rotate
from src.utils.general import *
from tqdm import tqdm
from src.utils.brain_visualization import detailed_plot_from3d
import os
import shutil


def optimize_centers(run_name, num_epochs, location, hdf5_target, batch_size=1):
    print(f"Started optimizing centers with the run name: {run_name}")

    save_location = f"{location}/outputs/centering/{run_name}"

    try:
        os.mkdir(path=save_location)
    except FileExistsError:
        shutil.rmtree(save_location, ignore_errors=True)
        os.mkdir(path=save_location)

    os.mkdir(path=f"{save_location}/visuals")
    os.mkdir(path=f"{save_location}/rotations")

    print("Loading data_loading")
    dataset = HDF5Dataset(f"{location}/data/hdf5/{hdf5_target}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for brain in tqdm(dataloader, position=0):

        # Parameters to optimize, initial values are guesses based on a few manual examples
        yaw = torch.tensor([0.0001], requires_grad=True, device="cuda")
        pitch = torch.tensor([0.0001], requires_grad=True, device="cuda")
        roll = torch.tensor([0.0001], requires_grad=True, device="cuda")
        translation = torch.tensor([0.01], requires_grad=True, device="cuda")

        optimizer = torch.optim.Adam([yaw, translation], lr=0.03)

        name = brain['name'][0]
        tqdm.write(f"Optimizing for: {name}")

        img = brain['ct'].to("cuda")[0].transpose(1, 2).unsqueeze(dim=0)

        for e in tqdm(range(num_epochs), position=1):
            optimizer.zero_grad()

            t = translate_and_rotate(img,
                                     100. * yaw,
                                     100. * pitch,
                                     100 * roll,
                                     100. * translation)

            s = ssim_loss(t, use_other=True)
            p = pixel_loss(t, binary=True)

            loss = s + p

            loss.backward()
            optimizer.step()

            wandb.log({name: loss.item()})

        # Do it one last time for logging
        t = translate_and_rotate(img,
                                 100. * yaw,
                                 100. * pitch,
                                 100. * roll,
                                 100. * translation)
        detailed_plot_from3d(t, save=True, save_location=f"{save_location}/visuals",
                             name=name, use_wandb=True)

        ps = [100. * yaw, 100. * pitch, 100 * roll, 100. * translation]
        ps = [p.cpu().detach().numpy() for p in ps]
        np.save(f"{save_location}/rotations/{name}", np.array(ps))
