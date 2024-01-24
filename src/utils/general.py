import torch
import monai
import numpy as np
from src.constants import *


def save_brain(img, name):

    saver = monai.transforms.SaveImage(output_dir=f"/home/baris/Documents/Brain-SHIFT/brain-shift/data/fixed", separate_folder=False, writer="ITKWriter")
    saver(img.cpu().detach().numpy(), img.cpu().detach().meta)
