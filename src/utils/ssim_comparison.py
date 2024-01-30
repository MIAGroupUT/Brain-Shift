import torch
from src.losses.losses import *
from src.data_loading.datasets import *
from torch.utils.data import DataLoader
import time
from contextlib import contextmanager
import kornia
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


@contextmanager
def measure_loss_efficiency(loss_function_name):
    try:
        start_time = time.time()
        yield
    finally:
        end_time = time.time()
        print(f"{loss_function_name} took {end_time - start_time:.6f} seconds")


def local_ssim_loss(img, kernel_size=23):
    width = img.shape[-2]
    half_width = width // 2

    half1 = img[:, :, :half_width, :]
    half2 = img[:, :, half_width:, :]

    # Check the sizes, make sure they work together
    if half1.shape[-2] != half2.shape[-2]:
        half2 = half2[:, :, :-1, :]

    # return kornia.losses.ssim3d_loss(half1.unsqueeze(dim=0), torch.flip(half2.unsqueeze(dim=0), [-2]), kernel_size)
    return ssim(half1.unsqueeze(dim=0), half2.unsqueeze(dim=0), data_range=1, size_average=True, win_size=15)


if __name__ == '__main__':
    location = "/home/baris/Documents/work/brain-morphing"

    dataset = AllBidsDataset(f"{location}/data", slice_thickness='small', caching=False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    brain = next(iter(dataloader))['ct'].to("cuda")

    with measure_loss_efficiency('Custom Loss Function'):
        ssim = local_ssim_loss(brain)
        print(ssim)
