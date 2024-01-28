from torch.utils.data import Dataset
from typing import Literal
import nibabel as nib
from glob import glob
from os import path
import numpy as np
import torch
import monai
from monai.transforms import *

WINDOW_MIN = -5
WINDOW_MAX = 85

# TODO: add some correct rotations here
cached_transform = monai.transforms.Compose([
    # Apply CT windowing
    ScaleIntensityRanged(
        keys=["ct"],
        a_min=WINDOW_MIN,
        a_max=WINDOW_MAX,
        b_min=0,
        b_max=1,
        clip=True,
    ),
])

random_transform_2d = monai.transforms.Compose([
    monai.transforms.RandCropByPosNegLabeld(keys=['ct', 'annotation'], spatial_size=(512, 512, 1),
                                            label_key='annotation', num_samples=1, pos=1),
    monai.transforms.SqueezeDimd(keys=['ct', 'annotation'], dim=-1, update_meta=True),
    monai.transforms.RandRotated(keys=['ct', 'annotation'], range_x=3, prob=1, padding_mode="zeros",
                                 align_corners=True),
    monai.transforms.RandZoomd(keys=['ct', 'annotation'], min_zoom=0.8, max_zoom=1.2, prob=1, padding_mode="constant")
])

random_transform_3d = monai.transforms.Compose([
    monai.transforms.RandCropByPosNegLabeld(keys=['ct', 'annotation'], spatial_size=(256, 256, 30),
                                            label_key='annotation', num_samples=1, pos=1),
    monai.transforms.RandAffined(keys=['ct', 'annotation'], prob=0.5, rotate_range=3, scale_range=0.15, shear_range=0.15)
])


def add_background_channel(annotations):
    """
    Add a background channel to the annotation volumes.

    Parameters:
    annotations (Tensor): A tensor of shape [batch_size, channels, depth, height, width]

    Returns:
    Tensor: An updated tensor of shape [batch_size, channels + 1, depth, height, width]
    """
    # Create a background channel where all annotations are zero
    background = torch.prod(annotations == 0, dim=0, keepdim=True)

    # Concatenate the background channel
    annotations_with_background = torch.cat((background, annotations), dim=0)

    return annotations_with_background


class CTBidsDataset(Dataset):
    def __init__(self, bids_path: str, slice_thickness: Literal['small', 'large'] = None, exclude_registered=True,
                 caching=True):
        self.bids_path = bids_path
        self.caching = caching
        annotation_path_list = glob(path.join(bids_path, "sub-*", "ses-*", "annotation", "*annotation.nii.gz"))

        # Keep only participants with the wanted slice thickness if given
        if slice_thickness is not None:
            annotation_path_list = [annotation_path for annotation_path in annotation_path_list if
                                    f"slicethickness-{slice_thickness}" in annotation_path]

        # Keep only annotator 0 for the 2 participants who were annotated twice
        double_id_list = [(1175, "CT1"), (1170, "CT1")]
        for patient_id, session_id in double_id_list:
            removed_annotation_list = glob(
                path.join(bids_path, f"sub-{patient_id}", f"ses-{session_id}", "annotation",
                          "*annotator-1*annotation.nii.gz")
            )
            annotation_path_list = [annotation_path for annotation_path in annotation_path_list if
                                    annotation_path not in removed_annotation_list]

        if exclude_registered:
            annotation_path_list = [annotation_path for annotation_path in annotation_path_list if
                                    "registered-true" not in annotation_path]

        self.annotation_path_list = annotation_path_list
        self.cache = {}

    def __len__(self):
        return len(self.annotation_path_list)

    # @lru_cache()
    def __getitem__(self, index):

        if self.caching and index in self.cache:
            return self.cache[index]

        annotation_path = self.annotation_path_list[index]
        participant_path = path.dirname(path.dirname(annotation_path))
        participant_id, session_id, _, slice_thickness, registration, _ = path.basename(annotation_path).split("_")
        ct_path = path.join(participant_path, "ct",
                            f"{participant_id}_{session_id}_{slice_thickness}_{registration}_ct.nii.gz")

        # print(annotation_path)
        annotation_np = nib.load(annotation_path).get_fdata()
        nifti_data = nib.load(ct_path)
        ct_np = nifti_data.get_fdata()

        # Add channels
        ct_np = np.expand_dims(ct_np, axis=0)
        channels_annotation_np = np.zeros((3, *annotation_np.shape))
        for channel in range(3):
            channels_annotation_np[channel] = annotation_np == (channel + 1)

        if self.caching:

            self.cache[index] = {
                'name': f"{participant_id}_{session_id}_{slice_thickness}_{registration}",
                "ct": torch.tensor(ct_np, dtype=torch.float),
                "annotation": add_background_channel(torch.tensor(channels_annotation_np, dtype=torch.float))
            }

            self.cache[index] = cached_transform(self.cache[index])
            return self.cache[index]

        else:

            return cached_transform(
                {
                    'name': f"{participant_id}_{session_id}_{slice_thickness}_{registration}",
                    "ct": torch.tensor(ct_np, dtype=torch.float),
                    "annotation": add_background_channel(torch.tensor(channels_annotation_np, dtype=torch.float))
                }
            )


class SliceDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Load the full 3D volume using the base dataset
        data = self.base_dataset[idx]

        data = random_transform_2d(data)

        return data[0]


class Dataset3D(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Load the full 3D volume using the base dataset
        data = self.base_dataset[idx]
        data = random_transform_3d(data)

        return data
