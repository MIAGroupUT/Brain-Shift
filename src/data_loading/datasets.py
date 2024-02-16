import os

from torch.utils.data import Dataset
import nibabel as nib
from glob import glob
from os import path
import numpy as np
import torch
from src.data_loading.transforms import *


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


class AnnotatedBidsDataset(Dataset):
    def __init__(self, bids_path: str, slice_thickness: str = None,
                 exclude_registered: object = True,
                 caching: object = True, transform=cached_transform):
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
        self.transform = transform

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
                "annotation": add_background_channel(torch.tensor(channels_annotation_np, dtype=torch.float)),
                'affine': nifti_data.affine
            }

            self.cache[index] = self.transform(self.cache[index])
            return self.cache[index]

        else:

            return self.transform(
                {
                    'name': f"{participant_id}_{session_id}_{slice_thickness}_{registration}",
                    "ct": torch.tensor(ct_np, dtype=torch.float),
                    "annotation": add_background_channel(torch.tensor(channels_annotation_np, dtype=torch.float)),
                    'affine': nifti_data.affine
                }
            )


class AllBidsDataset(Dataset):
    def __init__(self, bids_path: str, slice_thickness: str = None,
                 exclude_registered: object = True,
                 caching: object = True, transform=cached_transform):
        self.bids_path = bids_path
        self.caching = caching
        path_list = glob(path.join(self.bids_path, "sub-*", "ses-*", "*ct", "*_ct*"))

        # Keep only participants with the wanted slice thickness if given
        if slice_thickness is not None:
            path_list = [img_path for img_path in path_list if
                         f"slicethickness-{slice_thickness}" in img_path]

        if exclude_registered:
            path_list = [img_path for img_path in path_list if
                         "registered-true" not in img_path]

        self.path_list = path_list
        self.cache = {}
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        if self.caching and index in self.cache:
            return self.cache[index]

        scan_path = self.path_list[index]
        participant_path = path.dirname(path.dirname(scan_path))
        participant_id, session_id, slice_thickness, registration, _ = path.basename(scan_path).split("_")
        ct_path = path.join(participant_path, "ct",
                            f"{participant_id}_{session_id}_{slice_thickness}_{registration}_ct.nii.gz")

        nifti_data = nib.load(ct_path)
        ct_np = nifti_data.get_fdata()

        if self.caching:

            self.cache[index] = {
                'name': f"{participant_id}_{session_id}_{slice_thickness}_{registration}",
                "ct": torch.tensor(ct_np, dtype=torch.float),
                'affine': nifti_data.affine
            }

            self.cache[index] = self.transform(self.cache[index])
            return self.cache[index]

        else:

            return self.transform(
                {
                    'name': f"{participant_id}_{session_id}_{slice_thickness}_{registration}",
                    "ct": torch.tensor(ct_np, dtype=torch.float),
                    'affine': nifti_data.affine
                }
            )


class NiftiDataset(Dataset):

    def __init__(self, location, caching=False):
        super().__init__()
        self.files = os.listdir(location)
        self.location = location
        self.caching = caching
        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if self.caching and idx in self.cache:
            return self.cache[idx]

        img = nib.load(os.path.join(self.location, self.files[idx]))
        affine = img.affine
        img = img.get_fdata()

        d = {
            'ct': img,
            'affine': affine,
            'name': self.files[idx]
        }

        if self.caching:
            self.cache[idx] = d

        return d


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
    def __init__(self, base_dataset, random=True):
        self.base_dataset = base_dataset
        self.random = random

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Load the full 3D volume using the base dataset
        data = self.base_dataset[idx]
        if self.random:
            data = random_transform_3d(data)

        return data
