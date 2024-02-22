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
        try:
            subj_group = hf.create_group(d['name'])
        except TypeError:
            subj_group = hf.create_group(d['name'][0])
        for k, v in d.items():

            if k == 'name':
                continue
            subj_group.create_dataset(k, data=v.detach().cpu().numpy())


def check_grad_fn(func, *args, **kwargs):
    """
    Checks if the output of the function `func` has a grad_fn, indicating that gradients can be computed.

    Parameters:
    - func: The function to be checked.
    - *args: Positional arguments to be passed to the function.
    - **kwargs: Keyword arguments to be passed to the function.

    Returns:
    - True if the function's output has a grad_fn and thus supports gradient computation, False otherwise.
    """
    # Call the function with the provided arguments
    output = func(*args, **kwargs)

    # Check if the output has a grad_fn and if it is not None
    if hasattr(output, 'grad_fn') and output.grad_fn is not None:
        return True
    else:
        return False
