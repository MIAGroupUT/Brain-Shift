from torch.utils.data import DataLoader
from src.data_loading.datasets import AllBidsDataset, AnnotatedBidsDataset
from src.utils.movement import translate_and_rotate
from src.utils.general import *
from tqdm import tqdm
from src.utils.brain_visualization import detailed_plot_from3d
import os
import nibabel
import shutil
import torch
from src.utils.general import add_result_to_hdf5


def infer_centered(run_name, location, read_location, slice_thickness='small', do_annotations=False):
    print(f"Started optimizing centers with the run name: {run_name}")

    save_location = f"{location}/outputs/inferred/centering/{run_name}"

    try:
        os.mkdir(path=save_location)
    except FileExistsError:
        shutil.rmtree(save_location, ignore_errors=True)
        os.mkdir(path=save_location)

    hdf5_file = f'{save_location}/{run_name}.hdf5'
    open_file = h5py.File(hdf5_file, 'w')

    print("Loading data_loading")
    if do_annotations:
        dataset = AnnotatedBidsDataset(f"{location}/data/bids", slice_thickness=slice_thickness, caching=False)
    else:
        dataset = AllBidsDataset(f"{location}/data/bids", slice_thickness=slice_thickness, caching=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for brain in tqdm(dataloader, position=0):

        name = brain['name'][0]
        affine = brain['affine'][0]
        img = brain['ct'][0].to("cuda").transpose(1, 2).unsqueeze(dim=0)

        items = np.load(f"{location}/outputs/centering/{read_location}/rotations/{name}.npy")

        t = translate_and_rotate(img, torch.tensor([items[0, 0]]).to("cuda"), torch.tensor([items[1, 0]]).to("cuda"),
                                 torch.tensor([items[2, 0]]).to("cuda"), torch.tensor([items[3, 0]]).to("cuda"))
        detailed_plot_from3d(t, save=False, save_location=f"{save_location}/visuals",
                             name=name, use_wandb=True)

        d = {
            'ct': t[0],
            'affine': affine,
            'name': name
        }

        if do_annotations:
            mask = brain['annotation'][0].to("cuda").transpose(1, 2).unsqueeze(dim=0)
            print(mask.shape)
            m_t = translate_and_rotate(mask, torch.tensor([items[0, 0]]).to("cuda"),
                                       torch.tensor([items[1, 0]]).to("cuda"), torch.tensor([items[2, 0]]).to("cuda"),
                                       torch.tensor([items[3, 0]]).to("cuda"))

            d['annotation'] = m_t[0]

        add_result_to_hdf5(d, hdf5_file)



        # b = nibabel.Nifti1Image(t.detach().cpu().numpy()[0, 0], affine)
        # nibabel.save(b, f"{save_location}/{name}")
