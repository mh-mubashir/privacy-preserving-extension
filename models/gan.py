import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
# """
# This work is based on the Theano/Lasagne implementation of
# Progressive Growing of GANs paper from tkarras:
# https://github.com/tkarras/progressive_growing_of_gans

# PyTorch Model definition
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from collections import OrderedDict


# class PixelNormLayer(nn.Module):
#     def __init__(self):
#         super(PixelNormLayer, self).__init__()

#     def forward(self, x):
#         return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


# class WScaleLayer(nn.Module):
#     def __init__(self, size):
#         super(WScaleLayer, self).__init__()
#         self.scale = nn.Parameter(torch.randn([1]))
#         self.b = nn.Parameter(torch.randn(size))
#         self.size = size

#     def forward(self, x):
#         x_size = x.size()
#         x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
#             x_size[0], self.size, x_size[2], x_size[3])

#         return x


# class NormConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding):
#         super(NormConvBlock, self).__init__()
#         self.norm = PixelNormLayer()
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, kernel_size, 1, padding, bias=False)
#         self.wscale = WScaleLayer(out_channels)

#     def forward(self, x):
#         x = self.norm(x)
#         x = self.conv(x)
#         x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
#         return x


# class NormUpscaleConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding):
#         super(NormUpscaleConvBlock, self).__init__()
#         self.norm = PixelNormLayer()
#         self.up = nn.Upsample(scale_factor=2, mode='nearest')
#         self.conv = nn.Conv2d(
#             in_channels, out_channels, kernel_size, 1, padding, bias=False)
#         self.wscale = WScaleLayer(out_channels)

#     def forward(self, x):
#         x = self.norm(x)
#         x = self.up(x)
#         x = self.conv(x)
#         x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
#         return x


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         self.features = nn.Sequential(
#             NormConvBlock(512, 512, kernel_size=4, padding=3),
#             NormConvBlock(512, 512, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
#             NormConvBlock(512, 512, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
#             NormConvBlock(512, 512, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
#             NormConvBlock(512, 512, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
#             NormConvBlock(256, 256, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
#             NormConvBlock(128, 128, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
#             NormConvBlock(64, 64, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1),
#             NormConvBlock(32, 32, kernel_size=3, padding=1),
#             NormUpscaleConvBlock(32, 16, kernel_size=3, padding=1),
#             NormConvBlock(16, 16, kernel_size=3, padding=1))

#         self.output = nn.Sequential(OrderedDict([
#                         ('norm', PixelNormLayer()),
#                         ('conv', nn.Conv2d(16,
#                                            3,
#                                            kernel_size=1,
#                                            padding=0,
#                                            bias=False)),
#                         ('wscale', WScaleLayer(3))
#                     ]))

#     def forward(self, x):
#         x = self.features(x)
#         x = self.output(x)
#         return x

# UNet generator
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Generator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
        
# class Discriminator(nn.Module):
#     def __init__(self, nc=3, ndf=64):
#         super(Discriminator, self).__init__()
#         self.nc = nc
#         self.ndf = ndf
#         self.main = nn.Sequential(
#             # 输入: (batch_size, 6, 224, 224)
#             nn.Conv2d(self.nc * 2, self.ndf, 4, 2, 1, bias=False),  # -> (batch_size, 64, 112, 112)
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),  # -> (batch_size, 128, 56, 56)
#             nn.BatchNorm2d(self.ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),  # -> (batch_size, 256, 28, 28)
#             nn.BatchNorm2d(self.ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),  # -> (batch_size, 512, 14, 14)
#             nn.BatchNorm2d(self.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),  # -> (batch_size, 512, 7, 7)
#             nn.BatchNorm2d(self.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),  # -> (batch_size, 512, 4, 4)
#             nn.BatchNorm2d(self.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.AdaptiveAvgPool2d((1, 1)),  # -> (batch_size, 512, 1, 1)
#             # 1×1卷积将通道数从512变为1
#             nn.Conv2d(self.ndf * 8, 1, 1, 1, 0, bias=False),  # -> (batch_size, 1, 1, 1)
#             nn.Sigmoid()
#         )
#         self.apply(weights_init)

#     def forward(self, condition, target):
#         input = torch.cat([condition, target], 1)
#         return self.main(input)

class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, condition, target=None):
        """Standard forward."""
        if target is not None:
            input = torch.cat([condition, target], 1)
        else:
            input = condition
        return self.model(input)


if __name__ == "__main__":
    data_dir = "/home/xing.ruo/datasets"
    workers = 2
    batch_size = 128
    img_size = 224
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 5
    lr = 0.0002
    beta1 = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    # G = Generator(input_nc=3, output_nc=3, num_downs=5, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False).to(device)
    D = Discriminator(input_nc=6, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)

    f_img = torch.randn(64, 3, 224, 224).to(device)
    img = torch.randn(64, 3, 224, 224).to(device)

    outG = G(f_img)
    outD = D(f_img, img)
    print(outG.shape)
    print(outD.shape)

    Loss = nn.BCEWithLogitsLoss()
    rand_label = torch.full((64, 1, 26, 26), 1.0).to(device)
    print(Loss(outD, rand_label))