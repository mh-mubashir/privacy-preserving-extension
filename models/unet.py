""" Full assembly of the parts to form the complete network """

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchprofile


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, size='standard'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        assert size in ['standard', 'small', 'mini', 'tiny', 'super_tiny', 'super_tiny_2', 'super_tiny_3'], 'Invalid UNet size'
        
        if size == 'standard':
            scaling = 1
        elif size == 'small':
            scaling = 2
        elif size == 'mini':
            scaling = 4
        elif size == 'tiny':
            scaling = 8
        elif size == 'super_tiny':
            scaling = 16
        elif size == 'super_tiny_2':
            scaling = 32
        elif size == 'super_tiny_3':
            scaling = 64
        else:
            raise ValueError('Invalid size')

        self.inc = (DoubleConv(n_channels, 64 // scaling))
        self.down1 = (Down(64 // scaling, 128 // scaling))
        self.down2 = (Down(128 // scaling, 256 // scaling))
        self.down3 = (Down(256 // scaling, 512 // scaling))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512 // scaling, 1024 // scaling // factor))
        self.up1 = (Up(1024 // scaling, 512 // scaling // factor, bilinear))
        self.up2 = (Up(512 // scaling, 256 // scaling // factor, bilinear))
        self.up3 = (Up(256 // scaling, 128 // scaling // factor, bilinear))
        self.up4 = (Up(128 // scaling, 64 // scaling, bilinear))
        self.outc = (OutConv(64 // scaling, n_classes))

    def forward(self, x, no_clamp=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if no_clamp:
            return logits
        return torch.clamp(logits, 0, 1)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
        
if __name__ == '__main__':
    model = UNet(1, 1, size='super_tiny_3')
    # print(model)
    x = torch.randn(1, 1, 224, 224)
    macs = torchprofile.profile_macs(model, (x,))
    print(f"MACs: {macs / 1e8:.2f} x10^8")