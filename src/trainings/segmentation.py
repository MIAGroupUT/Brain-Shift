import torch
import monai
from src.data.BidsDataset import CTBidsDataset, SliceDataset
from monai.data import DataLoader
import wandb
from src.utils.brain_visualization import vis_to_wandb_segmentation
from tqdm import tqdm
import os


def train_segmentation(run_name, location, batch_size, num_epochs=5000, use_only_full_images=True, lr=3e-4, device="cuda"):

    out_dir = f"{location}/{run_name}"
    os.mkdir(path=out_dir)
    os.mkdir(path=f"{out_dir}/weights")
    os.mkdir(path=f"{out_dir}/visuals")
    os.mkdir(path=f"{out_dir}/segmentations")

    print("Loading data")
    dataset = CTBidsDataset(f"{location}/data/bids", slice_thickness=("small" if use_only_full_images else None))
    dataset_2d = SliceDataset(dataset)
    dataloader = DataLoader(dataset_2d, batch_size=batch_size, shuffle=True)

    print("Training")
    model = monai.networks.nets.SwinUNETR(
        img_size=(512, 512),
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        drop_rate=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9993)
    loss_fn = monai.losses.DiceLoss()

    for epoch in tqdm(range(num_epochs)):
        model.train()

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

        if epoch % 5 == 0:
            vis_to_wandb_segmentation(img, out, mask, names, loss.item(), epoch=epoch)

        if epoch % 200 == 0:

            torch.save(model.state_dict(), f"{out_dir}/weights/{epoch}.pt")
            vis_to_wandb_segmentation(img, out, mask, names, loss.item(), epoch=epoch, save=True, save_path=f"{out_dir}/visuals")

        scheduler.step()
