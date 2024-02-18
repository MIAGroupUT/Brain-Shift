import torch
import monai
import numpy as np
from src.constants import *
import h5py


def save_brain(img, name):
    saver = monai.transforms.SaveImage(output_dir=f"/home/baris/Documents/Brain-SHIFT/brain-shift/data/fixed",
                                       separate_folder=False, writer="ITKWriter")
    saver(img.cpu().detach().numpy(), img.cpu().detach().meta)


def add_result_to_hdf5(d, h5_file):
    with h5py.File(h5_file, 'a') as hf:
        subj_group = hf.create_group(d['name'])
        for k, v in d.items():

            if k == 'name':
                continue
            subj_group.create_dataset(k, data=v.detach().cpu().numpy())
