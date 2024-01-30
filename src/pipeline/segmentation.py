import torch
import monai
from src.data_loading.datasets import AllBidsDataset, SliceDataset, Dataset3D
from monai.data import DataLoader
from src.utils.brain_visualization import vis_to_wandb_segmentation, vis_to_wandb_segmentation_3d
from tqdm import tqdm
import os
import shutil
import nibabel


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
    dataset = AllBidsDataset(f"{location}/data", slice_thickness=slice_thickness)
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
    model.load_state_dict(torch.load(f"{location}/{relative_model_path}"))

    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=roi,
        sw_batch_size=2,
        overlap=0.5,
        sw_device=device,
        device="cpu"
    )

    for item in tqdm(dataloader, position=0):

        brain = item['ct']
        name = item['name'][0]
        with torch.no_grad():
            output = inferer(inputs=brain, network=model, progress=True)
            nibabel.save(brain, f"{out_dir}/out/{name}")
            nibabel.save(output, f"{out_dir}/out/{name}_mask")