from src.data_loading.datasets import HDF5Dataset
from monai.data import Dataloader
import matplotlib.pyplot as plt
from src.utils.general import add_result_to_hdf5
import torch


def prepare_dataset(item, final_shape=(512, 512, 128)):
        
    new_item = {
        'ct': torch.nn.functional.interpolate(item['ct'], final_shape),
        'skull': torch.nn.functional.interpolate(item['skull'].int().float(), final_shape),
        'annotation': torch.nn.functional.interpolate(item['annotation'].int().float(), final_shape),
        'name': item['name'],
        'affine': item['affine']
    }

    return new_item