import torch
import monai
from src.data_loading.datasets import AnnotatedBidsDataset, SliceDataset, Dataset3D, HDF5Dataset
from monai.data import DataLoader
import wandb
from src.utils.brain_visualization import vis_to_wandb_segmentation, vis_to_wandb_segmentation_3d
from tqdm import tqdm
import os
import shutil


def train_segmentation(run_name, location, hdf5_name,  batch_size, num_epochs=1000, lr=3e-4, device="cuda",
                       dims=2, loader_num_workers=8):
    out_dir = f"{location}/outputs/{run_name}"
    try:
        os.mkdir(path=out_dir)
    except FileExistsError:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.mkdir(path=out_dir)
    os.mkdir(path=f"{out_dir}/weights")
    os.mkdir(path=f"{out_dir}/visuals")
    os.mkdir(path=f"{out_dir}/segmentations")

    print("Loading data_loading")

    # dataset = AnnotatedBidsDataset(f"{location}/data", slice_thickness=slice_thickness)
    filename = f"{location}/data/hdf5/{hdf5_name}"
    dataset = HDF5Dataset(hdf5_filename=filename, with_annotations=True)
    dataloader = None
    model = None

    if dims == 2:
        dataset_2d = SliceDataset(dataset)
        dataloader = DataLoader(dataset_2d, batch_size=batch_size, shuffle=True)

        model = monai.networks.nets.SwinUNETR(
            img_size=(512, 512),
            spatial_dims=2,
            in_channels=1,
            out_channels=4,
            drop_rate=0.1
        ).to(device)

    if dims == 3:
        dataset_3d = Dataset3D(dataset, random=True)
        dataloader = DataLoader(dataset_3d, batch_size=batch_size, shuffle=True, num_workers=loader_num_workers)

        model = monai.networks.nets.SwinUNETR(
            img_size=(256, 256, 32),
            spatial_dims=3,
            in_channels=1,
            out_channels=4,
            drop_rate=0.1
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9993)
    loss_fn = monai.losses.DiceLoss()

    model.train()

    for epoch in tqdm(range(num_epochs)):

        for d in dataloader:
            names = d['name']
            optimizer.zero_grad()
            img = d['ct'].to(device)
            mask = d['annotation'].to(device)

            out = model(img)
            out = torch.nn.Softmax(dim=1)(out)

            loss = loss_fn(out, mask)

            loss.backward()
            optimizer.step()
            wandb.log({"training_loss": loss.item()})

        if epoch % 50 == 0:
            vis_to_wandb_segmentation(img, out, mask, names, loss.item(), epoch=epoch, save=True,
                                      save_path=f"{out_dir}/visuals") if dims == 2 else vis_to_wandb_segmentation_3d(
                img, out, mask, names, loss.item(), epoch=epoch, save=True,
                save_path=f"{out_dir}/visuals")

        if epoch % 200 == 0:
            torch.save(model.state_dict(), f"{out_dir}/weights/{epoch}.pt")

        scheduler.step()
