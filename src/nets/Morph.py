import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import monai
from src.nets.voxelmorph_layers import *


# Inspired by Voxelmorph, should morph an image in a diffeomorphic way.
class Morph(nn.Module):

    def __init__(self, in_shape, mode='general', *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.in_shape = in_shape
        self.n_dim = len(in_shape)

        if mode == 'general' or mode == 'instance':

            self.unet = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.n_dim,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=2,
            )

        elif mode == 'aided' or mode == 'instance_aided':

            self.unet = monai.networks.nets.UNet(
                spatial_dims=3,
                in_channels=1+4,
                out_channels=self.n_dim,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=2,
            )
        else:
            raise NotImplementedError

        if self.n_dim == 2:
            self.flow_conv = nn.Conv2d(self.n_dim, self.n_dim, kernel_size=3, padding='same')
        else:
            self.flow_conv = nn.Conv3d(self.n_dim, self.n_dim, kernel_size=3, padding='same')

        # In the voxelmorph repo they start this layer with very small weights.
        self.flow_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_conv.weight.shape))
        self.flow_conv.bias = nn.Parameter(torch.zeros(self.flow_conv.bias.shape))

        self.integrate = VecInt(in_shape=in_shape, num_steps=7)

        self.spatial_transformer = SpatialTransformer(size=in_shape)

    def forward(self, img):

        x = self.unet(img)

        velocity_field = self.flow_conv(x)
        deformation_field = self.integrate(velocity_field)

        morphed_image_full = self.spatial_transformer(img, deformation_field)

        return morphed_image_full, velocity_field, deformation_field
