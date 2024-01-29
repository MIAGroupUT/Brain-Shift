import torch
import torch.nn as nn
import torch.nn.functional as f

'''Code snippets taken from the voxelmorph git repository'''
'''https://github.com/voxelmorph/voxelmorph'''


def apply_velocity_field(target, velocity_field):
    integrate = VecInt(in_shape=target.shape[-3:], num_steps=7).to(target.device)
    spatial_transformer = SpatialTransformer(size=target.shape[-3:]).to(target.device)
    deformation_field = integrate(velocity_field)

    return spatial_transformer(target, deformation_field)


def apply_deformation_field(target, deformation_field):
    spatial_transformer = SpatialTransformer(size=target.shape[-3:]).to(target.device)

    return spatial_transformer(target, deformation_field)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return f.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    # TODO: keep the middle steps as well so we can make a gif :)
    def __init__(self, in_shape, num_steps):
        super().__init__()

        assert num_steps >= 0, 'num_steps should be >= 0, found: %d' % num_steps
        self.num_steps = num_steps
        self.scale = 1.0 / (2 ** self.num_steps)
        self.transformer = SpatialTransformer(in_shape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.num_steps):
            vec = vec + self.transformer(vec, vec)
        return vec

class VecIntAll(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    # TODO: keep the middle steps as well so we can make a gif :)
    def __init__(self, in_shape, num_steps):
        super().__init__()

        assert num_steps >= 0, 'num_steps should be >= 0, found: %d' % num_steps
        self.num_steps = num_steps
        self.scale = 1.0 / (2 ** self.num_steps)
        self.transformer = SpatialTransformer(in_shape)

    def forward(self, vec):
        vs = []
        vec = vec * self.scale
        vs.append(vec)
        for _ in range(self.num_steps):
            vec = vec + self.transformer(vec, vec)
            vs.append(vec)
        return vs


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, num_dims):
        super().__init__()

        assert num_dims in [1, 2, 3], 'num_dims should be either 1, 2, or 3'

        self.factor = 1.0 / vel_resize

        # Subtle attack to people who run lower versions of python
        match num_dims:
            case 1:
                self.mode = 'linear'
            case 2:
                self.mode = 'bilinear'
            case 3:
                self.mode = 'trilinear'

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = f.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = f.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
