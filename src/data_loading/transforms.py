import monai

WINDOW_MIN = -5
WINDOW_MAX = 85

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


random_transform_2d = monai.transforms.Compose([
    monai.transforms.RandCropByPosNegLabeld(keys=['ct', 'annotation'], spatial_size=(512, 512, 1),
                                            label_key='annotation', num_samples=1, pos=1),
    monai.transforms.SqueezeDimd(keys=['ct', 'annotation'], dim=-1, update_meta=True),
    monai.transforms.RandRotated(keys=['ct', 'annotation'], range_x=3, prob=1, padding_mode="zeros",
                                 align_corners=True),
    monai.transforms.RandZoomd(keys=['ct', 'annotation'], min_zoom=0.8, max_zoom=1.2, prob=1, padding_mode="constant")
])


random_transform_3d = monai.transforms.Compose([
    monai.transforms.RandRotated(keys=['ct', 'annotation'], range_z=3, prob=1, padding_mode="zeros",
                                 align_corners=True),
    monai.transforms.RandCropByPosNegLabeld(keys=['ct', 'annotation'], spatial_size=(256, 256, 32),
                                            label_key='annotation', num_samples=1, pos=1),

])
