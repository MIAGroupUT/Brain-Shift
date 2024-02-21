import nibabel
import pydicom
import pydicom_seg
import numpy as np
import monai
import nibabel as nib
from os import path
import pandas as pd
from copy import deepcopy
import torch
from typing import Tuple


class SegmentationLabelError(Exception):
    def __init__(self, label_name):
        message = f"Label {label_name} is not allowed in a segmentation file.\n"
        super().__init__(message)


class SegmentationImage:
    label_list = ["hematoma", "right_ventricle", "left_ventricle"]

    def __init__(self, seg_dcm_path: str, image_dcm_path: str):
        # Order of the labels
        self.seg_dcm_path = seg_dcm_path
        self.interpolation = False

        seg_dcm = pydicom.dcmread(seg_dcm_path)
        reader = pydicom_seg.SegmentReader()
        result = reader.read(seg_dcm)

        # Extract meta-data
        self.n_slices = len(seg_dcm[0x0008, 0x1115].value[0][0x0008, 0x114a].value)
        self.n_cols = seg_dcm[0x0028, 0x0011].value
        self.n_rows = seg_dcm[0x0028, 0x0010].value
        self.annotation_name = seg_dcm[0x0008, 0x103e].value

        # Find min_slice_idx (index of the first annotated slice)
        min_slice_idx = self.n_slices
        slices_header_list = seg_dcm[0x5200, 0x9230]
        for slice_header in slices_header_list:
            slice_idx = slice_header[0x0020, 0x9111].value[0][0x0020, 0x9157].value[1] - 1
            min_slice_idx = slice_idx if slice_idx < min_slice_idx else min_slice_idx
        self.min_slice_idx = min_slice_idx

        channels_image = np.zeros((3, self.n_slices, self.n_rows, self.n_cols))
        for label_header in seg_dcm[0x0062, 0x0002]:
            # Name of the label in DICOM seg (corresponds to on entry of label_list)
            label_name = label_header[0x0062, 0x0005].value
            # Number of the segment in the DICOM file
            segment_number = label_header[0x0062, 0x0004].value
            # Find channel index
            try:
                label_idx = self.label_list.index(label_name)
            except ValueError:
                raise SegmentationLabelError(label_name)

            array = result.segment_data(segment_number)
            channels_image[label_idx, -min_slice_idx - len(array):-min_slice_idx] = array

        channels_image = channels_image.transpose((0, 3, 2, 1))
        self.channels_image = channels_image

        loader = monai.transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)
        image_dict = loader({"image": image_dcm_path})
        self.image_dict = image_dict

    @property
    def shape(self) -> Tuple:
        return self.channels_image.shape[1::]

    @property
    def intensity_image(self) -> np.ndarray:
        """Returns a numpy array of size (L, W, H) in which each label corresponds to a different intensity."""
        image_np = np.zeros(self.shape)
        for label_idx, label in enumerate(self.label_list):
            image_np += self.channels_image[label_idx] * (label_idx + 1)

        return image_np

    def save(self, nii_path: str) -> None:
        """Saves the intensity_image as a compressed NifTi file."""
        if nii_path.endswith(".nii"):
            nii_path += ".gz"  # Force the result to be compressed
        elif not nii_path.endswith(".nii.gz"):
            nii_path += "nii.gz"

        image_nii = nibabel.Nifti1Image(self.intensity_image, affine=self.image_dict["image_meta_dict"]["affine"].numpy())
        nib.save(image_nii, nii_path)

    def compute_scans_tsv(self, nii_filename: str) -> pd.DataFrame:
        slice_thickness = self.image_dict["image_meta_dict"]["spacing"][-1]
        seg_filename = path.basename(self.seg_dcm_path)
        series_nb = int(path.splitext(seg_filename)[0].split("_")[-1][1::])
        row_df = pd.DataFrame(
            [[path.join("annotation", nii_filename), series_nb, slice_thickness, seg_filename, self.interpolation]],
            columns=["filename", "original_series_number", "slice_thickness", "original_seg_filename", "interpolation"])

        return row_df

    def resize_to_target(self, target_dcm_path: str) -> None:
        """Resize to the space of the given image"""

        loader = monai.transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)
        target_dict = loader({"image": target_dcm_path})

        resized_transform = monai.transforms.Spacingd(keys=["image"], pixdim=target_dict["image_meta_dict"]["spacing"])
        annotation_dict = deepcopy(self.image_dict)
        annotation_dict["image"].set_array(torch.from_numpy(self.channels_image))
        resized_annotation_dict = resized_transform(annotation_dict)
        # Binarize again post-interpolation
        resized_np = (resized_annotation_dict["image"].numpy() > 0.5).astype(int)

        # Estimate the offset on both sides (top and bottom)
        target_z = target_dict["image"].shape[-1]
        z_top_offset = int(target_dict['image_meta_dict']['lastImagePositionPatient'][-1] -
                           self.image_dict['image_meta_dict']['lastImagePositionPatient'][-1])
        z_bottom_offset = target_z - resized_annotation_dict["image"].shape[-1] - z_top_offset
        assert z_top_offset >= 0
        assert z_bottom_offset >= 0

        # Add slices which may have been removed when estimating the large thickness image
        registered_np = np.zeros((3, *target_dict["image_meta_dict"]["spatial_shape"]))
        registered_np[:, :, :, z_bottom_offset:target_z - z_top_offset] = resized_np

        self.channels_image = registered_np
        self.image_dict = target_dict
        self.interpolation = True


def dice_score(x1, x2):
    return (2 * (x1 * x2).sum() / (x1.sum() + x2.sum())).item()
