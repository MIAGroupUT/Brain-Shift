from src.data_loading.datasets import HDF5Dataset
from monai.data import DataLoader
from src.utils.general import add_result_to_hdf5
import torch
from tqdm import tqdm


def prepare_item(item, final_shape=(512, 512, 128)):

    original_shape = item['ct'].shape[-1]
    ratio = original_shape / final_shape[-1]

    item['affine'][0, 2, 2] = item['affine'][0, 2, 2] * ratio
        
    new_item = {
        'ct': torch.nn.functional.interpolate(item['ct'], final_shape, mode='trilinear')[0],
        'skull': torch.nn.functional.interpolate(item['skull'].int().float(), final_shape, mode='nearest-exact')[0],
        'annotation': torch.nn.functional.interpolate(item['annotation'].int().float(), final_shape, mode='nearest-exact')[0],
        'name': item['name'][0],
        'affine': item['affine'][0]
    }

    return new_item


def prepare_dataset(hdf5_input, hdf5_target):

    dataset = HDF5Dataset(hdf5_input, with_skulls=True, with_annotations=True)
    dataloader = DataLoader(dataset, num_workers=1, batch_size=1)

    for item in tqdm(dataloader):

        new_item = prepare_item(item)
        add_result_to_hdf5(new_item, hdf5_target)


if __name__ == '__main__':
    location = "/home/imreb/brain-morphing"
    input_file = f"{location}/data/hdf5/all_segmented.hdf5"
    output_file = f"{location}/data/hdf5/ready_to_morph.hdf5"

    prepare_dataset(input_file, output_file)
