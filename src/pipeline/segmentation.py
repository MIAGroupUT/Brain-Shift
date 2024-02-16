import torch
import monai
from src.data_loading.datasets import AllBidsDataset, SliceDataset, Dataset3D, NiftiDataset
from monai.data import DataLoader
from src.utils.brain_visualization import vis_to_wandb_segmentation, vis_to_wandb_segmentation_3d
from tqdm import tqdm
import os
import shutil
import nibabel
import numpy as np
import h5py
from src.utils.skull_stripping import skull_mask


def add_result_to_hdf5(d, h5_file):
    with h5py.File(h5_file, 'a') as hf:
        subj_group = hf.create_group(d['name'])
        for k, v in d.items():

            if k == 'name':
                continue
            subj_group.create_dataset(k, data=v.numpy())


def infer_segmentation(location, relative_model_path, run_name, slice_thickness="large", device="cuda", make_hdf5=False, use_nifti=False, nifti_location="", do_skull_strip=False):

    out_dir = f"{location}/outputs/inferred/segmentation/{run_name}"
    try:
        os.mkdir(path=out_dir)
    except FileExistsError:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.mkdir(path=out_dir)
    os.mkdir(path=f"{out_dir}/visuals")
    os.mkdir(path=f"{out_dir}/out")
    os.mkdir(path=f"{out_dir}/tensors")

    hdf5_file = None
    if make_hdf5:
        hdf5_file = f'{out_dir}/{run_name}.hdf5'
        open_file = h5py.File(hdf5_file, 'w')


    # Load the data
    dataset = AllBidsDataset(f"{location}/data", slice_thickness=slice_thickness, exclude_registered=False)
    if use_nifti:
        dataset_nifti = NiftiDataset(location=f"{location}/{nifti_location}", caching=False)
        dataset = Dataset3D(dataset_nifti, random=False)

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

        brain = item['ct'].unsqueeze(dim=0)
        name = item['name'][0]
        affine = item['affine'][0]
        with torch.no_grad():

            output = inferer(inputs=brain, network=model)

            b = nibabel.Nifti1Image(brain.detach().cpu().numpy()[0, 0], affine)
            o = nibabel.Nifti1Image(np.argmax(output.detach().cpu().numpy()[0], axis=0).astype(float), affine)

            nibabel.save(b, f"{out_dir}/out/{name}")
            nibabel.save(o, f"{out_dir}/out/{name}_mask")

            d = {
                'img': brain,
                'annotation': output,
                'affine': affine,
                'name': name
            }

            if do_skull_strip:
                skull = skull_mask(brain)[0]
                d['skull'] = skull

            if make_hdf5:
                add_result_to_hdf5(d, hdf5_file)

