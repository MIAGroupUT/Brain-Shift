from torch.utils.data import DataLoader
import wandb
from src.data_loading.datasets import AllBidsDataset, Dataset3D
from src.losses.losses import *
from src.utils.general import *
from src.nets.Morph import Morph
from tqdm import tqdm
from src.utils.brain_visualization import detailed_plot_from3d
import os
import shutil


def train_morph(run_name, num_epochs, location, batch_size=1, lr=3e-4):
    print(f"Started morphing things out: {run_name}")

    save_location = f"{location}/outputs/morph/{run_name}"

    try:
        os.mkdir(path=save_location)
    except FileExistsError:
        shutil.rmtree(save_location, ignore_errors=True)
        os.mkdir(path=save_location)

    os.mkdir(path=f"{save_location}/visuals")
    os.mkdir(path=f"{save_location}/v_fields")
    os.mkdir(path=f"{save_location}/results")

    # TODO load data that has segmentations and is centered
    dataloader = None

    fixed_shape = (512, 512, 400)  # TODO
    model = Morph(fixed_shape)

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

        if epoch % 20 == 0:
            vis_to_wandb_segmentation(img, out, mask, names, loss.item(), epoch=epoch, save=True,
                                      save_path=f"{out_dir}/visuals") if dims == 2 else vis_to_wandb_segmentation_3d(
                img, out, mask, names, loss.item(), epoch=epoch, save=True,
                save_path=f"{out_dir}/visuals")

        if epoch % 200 == 0:
            torch.save(model.state_dict(), f"{out_dir}/weights/{epoch}.pt")

        scheduler.step()
