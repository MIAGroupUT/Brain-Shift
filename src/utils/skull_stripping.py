from src.constants import *
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
import torch

def skull_mask(img, threshold=0.95):
    """
    Poor men's skull mask. Dilate the intensity threshold image until the skull is watertight
    then use a whole filling algorithm to fill the brain whole. Finally, dilate more drastically then the fill
    to get a fitting mask. For the purposes it can be a bit less than the skull itself as the goal is to
    remove the skin and other things from the CT.
    """

    basic = img.cpu().detach()
    basic = (basic > threshold).int().numpy()[0, 0]

    # Binary dilate, fill the wholes and then erode with bigger kernel
    dilated = ndimage.binary_dilation(basic, structure=np.ones((5, 5, 5)))
    filled = ndimage.binary_fill_holes(dilated).astype(int)
    eroded = ndimage.binary_erosion(filled, structure=np.ones((5, 5, 5)))

    return torch.tensor(eroded).unsqueeze(dim=0).unsqueeze(dim=0).to(img.device)


def head_mask(img, threshold=0.05):

    basic = img.cpu().detach()
    basic = (basic > threshold).int().numpy()[0, 0]

    # Binary dilate, fill the wholes and then erode with bigger kernel
    dilated = ndimage.binary_dilation(basic, structure=np.ones((25, 25, 25)))
    filled = ndimage.binary_fill_holes(dilated).astype(int)
    eroded = ndimage.binary_erosion(filled, structure=np.ones((25, 25, 25)))

    return torch.tensor(eroded).unsqueeze(dim=0).unsqueeze(dim=0).to(img.device)
