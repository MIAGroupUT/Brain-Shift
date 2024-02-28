import torch
import torch.nn.functional as F
from src.losses.differentiable_histogram import differentiable_histogram
import kornia
import torchvision
import monai
import wandb
import einops
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

dice_loss_no_background = monai.losses.DiceLoss(include_background=False)
dice_loss_yes_background = monai.losses.DiceLoss()


# def pixel_loss(img, binary=False):
#     width = img.shape[3]
#     half_width = width // 2
#
#     half1 = img[:, :, :, :half_width, :]
#     half2 = img[:, :, :, half_width:, :]
#
#     # Check the sizes, make sure they work together
#     if half1.shape[3] != half2.shape[3]:
#         half2 = half2[:, :, :, :-1, :]
#
#     if binary:
#         return (torch.abs(torch.sum(half1 != 0) - torch.sum(half2 != 0)) + 1e-8) / torch.sum(img != 0)
#     else:
#         return (torch.abs(torch.sum(half1) - torch.sum(half2)) + 1e-8) / torch.sum(img)

def pixel_loss(img, binary=False):
    width = img.shape[3]
    half_width = width // 2

    half1 = img[:, :, :, :half_width]
    half2 = img[:, :, :, half_width:]

    # Ensure equal dimensions for odd widths
    if half1.shape[3] != half2.shape[3]:
        half2 = half2[:, :, :, :-1]

    if binary:
        # Apply a sigmoid to softly approximate the binary condition
        # The beta parameter controls the steepness of the transition
        beta = 100  # This is a hyperparameter you can adjust
        soft_binary_half1 = torch.sigmoid(beta * (half1 - 0.001))
        soft_binary_half2 = torch.sigmoid(beta * (half2 - 0.001))
        loss = (torch.abs(torch.sum(soft_binary_half1) - torch.sum(soft_binary_half2)) + 1e-8) / (
                torch.sum(torch.sigmoid(beta * (img - 0.001))) + 1e-8)
    else:
        loss = (torch.abs(torch.sum(half1) - torch.sum(half2)) + 1e-8) / torch.sum(img)

    return loss


def jeffreys_divergence_loss(img, bins=90):
    hist_x = differentiable_histogram(img[:, :, :, :img.shape[3] // 2, :], n_bins=bins)[0]
    hist_y = differentiable_histogram(img[:, :, :, img.shape[3] // 2:, :], n_bins=bins)[0]

    eps = 1e-10

    pi = hist_x / torch.sum(hist_x)
    qi = hist_y / torch.sum(hist_y)

    return torch.sum(pi * torch.log(((pi + eps) / (qi + eps)))) + torch.sum(qi * torch.log(((qi + eps) / (pi + eps))))


def ssim_loss(img, kernel_size=23, use_other=True):
    width = img.shape[3]
    half_width = width // 2

    half1 = img[:, :, :, :half_width, :]
    half2 = img[:, :, :, half_width:, :]

    # Check the sizes, make sure they work together
    if half1.shape[3] != half2.shape[3]:
        half2 = half2[:, :, :, :-1, :]

    if use_other:
        return 1.0 - ssim(half1, torch.flip(half2, [3]), data_range=1, size_average=True, win_size=kernel_size)
    return kornia.losses.ssim3d_loss(half1, torch.flip(half2, [3]), kernel_size)


def dice_binary_loss(img):
    width = img.shape[3]
    half_width = width // 2

    half1 = img[:, :, :, :half_width, :]
    half2 = img[:, :, :, half_width:, :]

    # Check the sizes, make sure they work together
    if half1.shape[3] != half2.shape[3]:
        half2 = half2[:, :, :, :-1, :]

    return dice_loss_yes_background(half1 != 0, torch.flip(half2 != 0, [3]))

def ventricle_volume(before, after):
    return torch.relu((torch.sum(before) - torch.sum(after)) / torch.sum(before))


def half_half_mse(img, blur_kernel=13):
    width = img.shape[2]
    half_width = width // 2

    half1 = img[:, :, :half_width, :]
    half2 = img[:, :, half_width:, :]

    half1 = torchvision.transforms.functional.gaussian_blur(half1, kernel_size=blur_kernel)
    half2 = torchvision.transforms.functional.gaussian_blur(half2, kernel_size=blur_kernel)

    # Check the sizes, make sure they work together
    if half1.shape[2] != half2.shape[2]:
        half2 = half2[:, :, :-1, :]

    return torch.nn.functional.mse_loss(half1, half2)


def get_skull(scan, skull_threshold=0.99):
    skull = scan > skull_threshold
    return skull.float()


def skull_loss(original_skull, morphed_skull):
    return dice_loss_no_background(original_skull, morphed_skull)


def volume_loss(original_mask, new_mask):
    original_hematoma_volume = torch.sum(original_mask)

    return 1 - (original_hematoma_volume - torch.sum(new_mask)) / original_hematoma_volume


def symmetry_loss(original_jeffrey, original_ssim, morphed_img, log_wandb=False):
    jeffrey = jeffreys_divergence_loss(morphed_img)
    ssim = ssim_loss(morphed_img)

    j_diff = 1 - (original_jeffrey - jeffrey) / original_jeffrey
    s_diff = 1 - (original_ssim - ssim) / original_ssim

    if log_wandb:
        wandb.log({'jeffrey': j_diff.item(), 'ssim': s_diff.item()})

    return (j_diff + 5.0 * s_diff) / 6.0


def foreground_loss(original_foreground, morphed_img):
    m = torch.sum(morphed_img > 0.0)

    return torch.abs((original_foreground - m) / original_foreground)


def spatial_gradient_l1(deformation_field):
    volume_shape = deformation_field.shape[2:]  # Assuming the input is [B, C, *SpatialDims]

    dfs_sum = 0

    for dim in range(len(volume_shape)):
        # Compute the gradient along each spatial dimension independently
        slice1 = [slice(None)] * len(deformation_field.shape)  # All items along all dimensions
        slice2 = [slice(None)] * len(deformation_field.shape)

        slice1[dim + 2] = slice(1, None)  # Shift start to exclude the first element along the current spatial dimension
        slice2[dim + 2] = slice(None, -1)  # Shift end to exclude the last element along the current spatial dimension

        # Calculate the gradient by subtracting shifted views
        df_dim = deformation_field[slice1] - deformation_field[slice2]

        # Calculate the L1 norm of the gradient and mean across all dimensions except the batch dimension
        df_abs_mean = torch.mean(torch.abs(df_dim), dim=list(range(1, df_dim.ndim)))

        # Accumulate the mean gradients
        dfs_sum += df_abs_mean

    # Average the accumulated gradients over the number of dimensions
    grad = dfs_sum / len(volume_shape)

    return grad.mean()


def gradient_loss(s, power=1, step=1):
    dy = torch.abs(s[:, :, step:, :, :] - s[:, :, :-step, :, :])
    dx = torch.abs(s[:, :, :, step:, :] - s[:, :, :, :-step, :])
    dz = torch.abs(s[:, :, :, :, step:] - s[:, :, :, :, :-step])

    dy **= power
    dx **= power
    dz **= power

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)

    return d / 3.0




def ventricle_volume(before, after):
    return torch.relu((torch.sum(before) - torch.sum(after)) / torch.sum(before))


def compute_jacobian_det(displacement, voxel_spacing=(1, 1, 1)):
    """
    Calculate the Jacobian determinant value at each point of the displacement map.

    Args:
    - displacement (array): Size of b*h*w*d*3 in the cubic volume of [-1, 1]^3.
    - voxel_spacing (tuple): Spacing of voxels in x, y, z directions. Default is (1, 1, 1).
er segmentation changes"
    Returns:
    - Jacobian determinant (array): The calculated Jacobian determinant at each voxel.
    """

    dx, dy, dz = voxel_spacing

    # Derivatives
    D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :]) / dy
    D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :]) / dx
    D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :]) / dz

    # Components of the Jacobian matrix determinant
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = D_x[..., 1] * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = D_x[..., 2] * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    return F.pad(D1 - D2 + D3, (1, 0, 1, 0, 1, 0), mode='constant')


# If there are voxels with negative jacobian outside the segmentation mask, this is not good
def jacobian_loss(v_field, voxel_size, seg_mask, stripped_brain):
    r = einops.rearrange(v_field, "b three h w d -> b h w d three")

    brain_mask = stripped_brain > 0.0

    det = compute_jacobian_det(r, voxel_size)

    # Negative mask -> wherever the hematoma is not
    # Rule the negative determinant, get all the negatives to add up
    return F.relu((-det * (1 - seg_mask)) * brain_mask).mean()


def ventricle_overlap(ventricle_left, ventricle_right, one_piece=False):

    if one_piece:
        point = ventricle_left.shape[-2] // 2

        points = torch.clamp((ventricle_left + ventricle_right), min=0.0, max=1.0)

        return dice_loss_no_background(points[:, :, :, :point, :],
                                       torch.flip(points[:, :, :, point:, :], [-2]))

    point = ventricle_left.shape[-2] // 2
    return dice_loss_no_background(ventricle_left[:, :, :, :point, :],
                                   torch.flip(ventricle_right[:, :, :, point:, :], [-2]))


def distance_based_loss_3D(volume, correct_side):
    """
    Compute the distance-based loss for 3D volumes in a batch with multiple channels.

    Parameters:
    - volume: Tensor of shape [batch, channel, depth, height, width] with 1s indicating the presence of the volume and 0s otherwise.
    - correct_side: Should be 'above' or 'below' indicating the correct side of the central plane.

    Returns:
    - Normalized distance-based loss with gradients, averaged over the batch and channel dimensions.
    """
    # Assume volume is a float tensor for the purpose of gradient computation
    volume = volume.float()

    # Calculate the central plane (middle of the y-dimension, which is index -2)
    center_y = volume.shape[-2] / 2.0

    # Generate a grid of y-coordinate values
    _, _, _, y_grid, _ = torch.meshgrid(
        torch.arange(volume.shape[0], device=volume.device),
        torch.arange(volume.shape[1], device=volume.device),
        torch.arange(volume.shape[2], device=volume.device),
        torch.arange(volume.shape[3], device=volume.device),
        torch.arange(volume.shape[4], device=volume.device),
        indexing="ij"
    )
    y_grid = y_grid.float()

    # Calculate the distance of all points from the central plane
    distances = torch.abs(y_grid - center_y)

    # Determine which side of the plane the volume should be on
    if correct_side == 'right':
        # Penalize volume that is below the center (y < center_y)
        penalties = (y_grid < center_y).float() * volume
    else:
        # Penalize volume that is above the center (y > center_y)
        penalties = (y_grid > center_y).float() * volume

    # Calculate the sum of distances for the wrongly placed voxels
    total_distance = torch.sum(distances * penalties, dim=(-3, -2, -1))  # sum over depth, height, and width

    # Calculate the maximum possible sum of distances for normalization
    max_distance = torch.sum(distances, dim=(-3, -2, -1)) / 2.0  # sum over depth, height, and width

    # Normalize the total distance by the maximum possible sum of distances
    normalized_loss = torch.sqrt(
        torch.mean((total_distance + 1e-10) / max_distance)) * 5.0  # average over the batch and channel dimensions

    return normalized_loss


def ventricle_wrong_side(deformed_left, deformed_right):
    l_loss = distance_based_loss_3D(deformed_left, 'left')
    r_loss = distance_based_loss_3D(deformed_right, 'right')

    return l_loss + r_loss
