from torch.utils.data import DataLoader
import wandb
from src.data_loading.datasets import AllBidsDataset, Dataset3D
from src.losses.losses import jeffreys_divergence_loss, ssim_loss, pixel_loss
from src.utils.movement import translate_and_rotate
from src.utils.general import *
from tqdm import tqdm
from src.utils.brain_visualization import detailed_plot_from3d
import os
import shutil


def optimize_centers(run_name, num_epochs, location, batch_size=1):
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
    dataset = AllBidsDataset(f"{location}/data", slice_thickness='small', caching=False)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for brain in tqdm(dataloader, position=0):

        # Parameters to optimize, initial values are guesses based on a few manual examples
        yaw = torch.tensor([0.8], requires_grad=True, device="cuda")
        pitch = torch.tensor([0.05], requires_grad=True, device="cuda")
        translation = torch.tensor([0.08], requires_grad=True, device="cuda")

        optimizer = torch.optim.Adam([yaw, pitch, translation], lr=0.02)

        name = brain['name'][0]
        tqdm.write(f"Optimizing for: {name}")

        img = brain['ct'].to("cuda").unsqueeze(dim=0)

        for e in tqdm(range(num_epochs), position=1):
            optimizer.zero_grad()

            t = translate_and_rotate(img, 100. * yaw, 100. * pitch, 100. * translation)

            s = ssim_loss(t, use_other=True)
            j = 10.0 * jeffreys_divergence_loss(t)
            p = pixel_loss(t, binary=False)

            loss = s + j + p

            loss.backward()
            optimizer.step()

            wandb.log({name: loss.item()})

            if e % 10 == 0:
                detailed_plot_from3d(t, name=f"{name}", use_wandb=True, loss=loss.item())

        # Do it one last time for logging
        t = translate_and_rotate(img, 100. * yaw, 100. * pitch, 100. * translation)
        detailed_plot_from3d(t, save=True, save_location=f"{save_location}/visuals",
                             name=name, use_wandb=True)

        ps = [yaw, pitch, translation]
        ps = [p.cpu().detach().numpy() for p in ps]
        np.save(f"{save_location}/rotations/{name}", np.array(ps))
