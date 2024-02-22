import monai
import torch
from monai.transforms import MapTransform
import numpy as np

WINDOW_MIN = -20
WINDOW_MAX = 100

# TODO: add some correct rotations here
cached_transform = monai.transforms.Compose([
    # Apply CT windowing
    monai.transforms.ScaleIntensityRanged(
        keys=["ct"],
        a_min=WINDOW_MIN,
        a_max=WINDOW_MAX,
        b_min=0,
        b_max=1,
        clip=True,
        allow_missing_keys=True,
    ),
])




class FlipAndSwapLabels(MapTransform):
    """
    Custom MONAI transform to flip the image and segmentation masks horizontally
    and swap the left and right ventricle labels with a given probability.

    Args:
        keys (Sequence[str]): Keys of the corresponding items to be transformed.
        prob (float): Probability of applying the transform.
        left_vent_idx (int): Index of the left ventricle in the segmentation mask channels.
        right_vent_idx (int): Index of the right ventricle in the segmentation mask channels.
    """

    def __init__(self, keys, prob=0.5, left_vent_idx=2, right_vent_idx=3, seg_key='annotation', img_key='ct'):
        super().__init__(keys)
        self.prob = prob
        self.left_vent_idx = left_vent_idx
        self.right_vent_idx = right_vent_idx
        self.seg_key = seg_key
        self.img_key = img_key

    def __call__(self, data):
        # Randomly decide whether to apply the transform
        apply_transform = np.random.rand() < self.prob

        if not apply_transform:
            return data


        # Flip the img itself
        data[self.img_key] = torch.flip(data[self.img_key], dims=[-2])

        # Flip the ventricles
        mask = data[self.seg_key]
        flipped_mask = torch.flip(mask, dims=[-2])
        swapped_mask = flipped_mask.clone()

        swapped_mask[self.left_vent_idx] = flipped_mask[self.right_vent_idx]
        swapped_mask[self.right_vent_idx] = flipped_mask[self.left_vent_idx]
        data[self.seg_key] = swapped_mask

        return data


random_transform_2d = monai.transforms.Compose([
    monai.transforms.RandCropByPosNegLabeld(keys=['ct', 'annotation'], spatial_size=(512, 512, 1),
                                            label_key='annotation', num_samples=1, pos=1),
    monai.transforms.SqueezeDimd(keys=['ct', 'annotation'], dim=-1, update_meta=True),
    monai.transforms.RandRotated(keys=['ct', 'annotation'], range_x=3, prob=1, padding_mode="zeros",
                                 align_corners=True),
    monai.transforms.RandZoomd(keys=['ct', 'annotation'], min_zoom=0.8, max_zoom=1.2, prob=1, padding_mode="constant")
])

random_transform_3d = monai.transforms.Compose([
    FlipAndSwapLabels(keys=['ct', 'annotation']),

    monai.transforms.RandRotated(keys=['ct', 'annotation'], range_z=2, prob=0.5, padding_mode="zeros",
                                 align_corners=True),
    monai.transforms.RandSpatialCropd(keys=['ct', 'annotation'], roi_size=(256, 256, 32), random_size=False),

])
