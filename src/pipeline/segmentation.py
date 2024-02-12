import torch
import monai
from src.data_loading.datasets import AllBidsDataset, SliceDataset, Dataset3D
from monai.data import DataLoader
from src.utils.brain_visualization import vis_to_wandb_segmentation, vis_to_wandb_segmentation_3d
from tqdm import tqdm
import os
import shutil
import nibabel
import numpy as np


def infer_segmentation(location, relative_model_path, run_name, slice_thickness="large", device="cuda"):

    out_dir = f"{location}/outputs/inferred/segmentation/{run_name}"
    try:
        os.mkdir(path=out_dir)
    except FileExistsError:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.mkdir(path=out_dir)
    os.mkdir(path=f"{out_dir}/visuals")
    os.mkdir(path=f"{out_dir}/out")

    # Load the data
    dataset = AllBidsDataset(f"{location}/data", slice_thickness=slice_thickness, exclude_registered=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    roi = (256, 256, 32)

    # Load the model
    model = monai.networks.nets.SwinUNETR(
        img_size=roi,
        spatial_dims=3,
        in_channels=1,
        out_channels=4,
    ).to(device)

    model.eval()
    model.load_state_dict(torch.load(f"{location}/outputs/{relative_model_path}"))

    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=roi,
        sw_batch_size=2,
        overlap=0.5,
        sw_device=device,
        device="cpu"
    )

    for item in tqdm(dataloader, position=0):

        brain = item['ct'].unsqueeze(dim=0)
        name = item['name'][0]
        affine = item['affine'][0]
        with torch.no_grad():
            output = inferer(inputs=brain, network=model)
            # nibabel.loadsave.save(brain.detach().cpu().numpy(), f"{out_dir}/out/{name}")
            # nibabel.loadsave.save(output.detach().cpu().numpy(), f"{out_dir}/out/{name}_mask")

            # print(output.shape)

            b = nibabel.Nifti1Image(brain.detach().cpu().numpy()[0, 0], affine)
            o = nibabel.Nifti1Image(np.argmax(output.detach().cpu().numpy()[0], axis=0).astype(float), affine)

            nibabel.save(b, f"{out_dir}/out/{name}")
            nibabel.save(o, f"{out_dir}/out/{name}_mask")

