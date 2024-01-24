import torch
import kornia
from kornia.geometry.transform import rotate3d


# By default it is okay to not use roll.
def rotate(img, yaw, pitch, roll=torch.tensor([0.], requires_grad=True)):
    return rotate3d(img, yaw, pitch, roll.to(img.device))


# Translate the image in amount of pixels
def translate(img, x):
    affine = torch.eye(4)
    affine[1, -1] = x
    affine = affine[:-1].to(img.device)

    return kornia.geometry.transform.affine3d(img, affine)


def translate_and_rotate(img, yaw, pitch, x):
    img = rotate(img, yaw, pitch)
    img = translate(img, x)
    return img



