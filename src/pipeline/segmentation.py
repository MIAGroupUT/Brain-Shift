import torch
import monai
from src.data_loading.datasets import AllBidsDataset, SliceDataset, Dataset3D, NiftiDataset, HDF5Dataset
from monai.data import DataLoader
from src.utils.brain_visualization import vis_to_wandb_segmentation, vis_to_wandb_segmentation_3d
from tqdm import tqdm
import os
import shutil
import h5py
from src.utils.skull_stripping import skull_mask
from src.utils.general import add_result_to_hdf5


def infer_segmentation(location, relative_model_path, run_name, hdf5_location, device="cuda", make_hdf5=True,
                       do_skull_strip=False):
    out_dir = f"{location}/outputs/inferred/segmentation/{run_name}"
    try:
        os.mkdir(path=out_dir)
    except FileExistsError:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.mkdir(path=out_dir)
    os.mkdir(path=f"{out_dir}/visuals")
    os.mkdir(path=f"{out_dir}/out")
    os.mkdir(path=f"{out_dir}/tensors")

    hdf5_file = f'{out_dir}/{run_name}.hdf5'

    # Load the data
    dataset = HDF5Dataset(hdf5_filename=hdf5_location, with_skulls=False, with_annotations=False)

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
        overlap=0.7,
        sw_device=device,
        device="cpu",
        progress=False
    )

    for item in tqdm(dataloader, position=0):

        brain = item['ct'].float()
        name = item['name'][0]
        affine = item['affine']
        with torch.no_grad():

            output = inferer(inputs=brain, network=model)
            output = torch.argmax(output, dim=1).float()

            d = {
                'ct': brain[0],
                'annotation': output[0],
                'affine': affine[0],
                'name': name
            }

            if do_skull_strip:
                skull = skull_mask(brain)[0]
                d['skull'] = skull.int()

            if make_hdf5:
                print(d.keys())
                add_result_to_hdf5(d, hdf5_file)
