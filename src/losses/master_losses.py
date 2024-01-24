from src.losses.losses import *
from src.models.voxelmorph_layers import apply_deformation_field


def master_loss(original_skull,
                hematoma_mask,
                left_v_mask,
                right_v_mask,
                original_foreground,
                skull_stripped,
                velocity_field,
                deformation_field,
                full_out,
                stripped_out,
                loss_weights,
                ssim_kernel,
                jeffrey_bins):

    morphed_skull = apply_deformation_field(original_skull, deformation_field)
    morphed_hematoma = apply_deformation_field(hematoma_mask, deformation_field)
    morphed_left_v = apply_deformation_field(left_v_mask, deformation_field)
    morphed_right_v = apply_deformation_field(right_v_mask, deformation_field)

    s = skull_loss(original_skull, morphed_skull)
    v = volume_loss(hematoma_mask, morphed_hematoma)
    ssim = ssim_loss(stripped_out, kernel_size=ssim_kernel)
    jeffrey = jeffreys_divergence_loss(stripped_out, bins=jeffrey_bins)
    f = foreground_loss(original_foreground, full_out)
    reg = spatial_gradient_l1(velocity_field)
    jac = jacobian_loss(velocity_field, voxel_size=(0.434, 0.434, 0.500), seg_mask=hematoma_mask, stripped_brain=skull_stripped)
    v_side = ventricle_wrong_side(deformed_left=morphed_left_v, deformed_right=morphed_right_v)
    v_overlap = ventricle_overlap(ventricle_left=morphed_left_v, ventricle_right=morphed_right_v)

    #
    wandb.log(
        {'skull preservation': s.item() * loss_weights['skull'],
         'hematoma volume': v.item() * loss_weights['volume'],
         'jeffrey': jeffrey.item() * loss_weights['jeffrey'],
         'ssim': ssim.item() * loss_weights['ssim'],
         'foreground': f.item() * loss_weights['foreground'],
         'regular': reg.item() * loss_weights['grad_reg'],
         'jacobian': jac.item() * loss_weights['jacobian'],
         'ventricle side': v_side.item() * loss_weights['ventricle_side'],
         'ventricle overlap': v_overlap.item() * loss_weights['ventricle_symm']
         }
    )

    return (
            loss_weights['skull'] * s +
            loss_weights['volume'] * v +
            loss_weights['ssim'] * ssim +
            loss_weights['jeffrey'] * jeffrey +
            loss_weights['foreground'] * f +
            loss_weights['grad_reg'] * reg +
            loss_weights['jacobian'] * jac +
            loss_weights['ventricle_side'] * v_side +
            loss_weights['ventricle_symm'] * v_overlap
    )
