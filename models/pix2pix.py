import torch
import torch.nn as nn
import torch.nn.functional as F


### 1. EncoderDecoderGenerator - Basic Encoder-Decoder

class EncoderDecoderGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(EncoderDecoderGenerator, self).__init__()
        
        # Encoder layers
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )
        
        # Decoder layers
        self.d1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        self.d2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        self.d3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        self.d4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.d5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        self.d6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        self.d7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        
        self.d8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder
        d1 = self.d1(e8)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)
        
        return d8



class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator, self).__init__() 
        
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        
        self.e2 = nn.Sequential( 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e8 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )
        
        # Decoder layers with skip connections
        self.d1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        # After concatenation with e7: ngf*8*2 channels
        self.d2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        # After concatenation with e6: ngf*8*2 channels
        self.d3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        # After concatenation with e5: ngf*8*2 channels
        self.d4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # After concatenation with e4: ngf*8*2 channels
        self.d5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        # After concatenation with e3: ngf*4*2 channels
        self.d6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        # After concatenation with e2: ngf*2*2 channels
        self.d7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        
        # After concatenation with e1: ngf*2 channels
        self.d8 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        
        # Decoder with skip connections
        d1 = self.d1(e8)
        d1_cat = torch.cat([d1, e7], dim=1)
        
        d2 = self.d2(d1_cat)
        d2_cat = torch.cat([d2, e6], dim=1)
        
        d3 = self.d3(d2_cat)
        d3_cat = torch.cat([d3, e5], dim=1)
        
        d4 = self.d4(d3_cat)
        d4_cat = torch.cat([d4, e4], dim=1)
        
        d5 = self.d5(d4_cat)
        d5_cat = torch.cat([d5, e3], dim=1)
        
        d6 = self.d6(d5_cat)
        d6_cat = torch.cat([d6, e2], dim=1)
        
        d7 = self.d7(d6_cat)
        d7_cat = torch.cat([d7, e1], dim=1)
        
        d8 = self.d8(d7_cat)
        
        return d8


class UnetGenerator128(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator128, self).__init__()
        
        # Encoder layers
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        
        self.e2 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e6 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e7 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1)
        )
        
        # Decoder layers with skip connections
        self.d1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        self.d2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        self.d3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        self.d4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        self.d5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        self.d6 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        
        self.d7 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        
        # Decoder with skip connections
        d1 = self.d1(e7)
        d1_cat = torch.cat([d1, e6], dim=1)
        
        d2 = self.d2(d1_cat)
        d2_cat = torch.cat([d2, e5], dim=1)
        
        d3 = self.d3(d2_cat)
        d3_cat = torch.cat([d3, e4], dim=1)
        
        d4 = self.d4(d3_cat)
        d4_cat = torch.cat([d4, e3], dim=1)
        
        d5 = self.d5(d4_cat)
        d5_cat = torch.cat([d5, e2], dim=1)
        
        d6 = self.d6(d5_cat)
        d6_cat = torch.cat([d6, e1], dim=1)
        
        d7 = self.d7(d6_cat)
        
        return d7


class BasicDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ndf=64):
        super(BasicDiscriminator, self).__init__()
        # Delegates to NLayerDiscriminator with n_layers=3
        self.model = NLayerDiscriminator(input_nc, output_nc, ndf, n_layers=3)
        
    def forward(self, input, target):
        return self.model(input, target)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ndf=64):
        super(PixelDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # input is (input_nc + output_nc) x 256 x 256
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            
            # state size: (ndf) x 256 x 256
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            
            # state size: (ndf*2) x 256 x 256
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0),
            
            # state size: 1 x 256 x 256
            nn.Sigmoid()
        )
        
    def forward(self, input, target):
        # Concatenate input and target along channel dimension
        x = torch.cat([input, target], dim=1)
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        
        if n_layers == 0:
            # Use PixelDiscriminator instead
            self.model = PixelDiscriminator(input_nc, output_nc, ndf).model
        else:
            sequence = []
            
            # First layer
            sequence.append(nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1))
            sequence.append(nn.LeakyReLU(0.2, True))
            
            nf_mult = 1
            nf_mult_prev = 1
            
            # Intermediate layers
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence.append(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                             kernel_size=4, stride=2, padding=1)
                )
                sequence.append(nn.BatchNorm2d(ndf * nf_mult))
                sequence.append(nn.LeakyReLU(0.2, True))
            
            # Penultimate layer
            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence.append(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=1, padding=1)
            )
            sequence.append(nn.BatchNorm2d(ndf * nf_mult))
            sequence.append(nn.LeakyReLU(0.2, True))
            
            # Final layer
            sequence.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
            sequence.append(nn.Sigmoid())
            
            self.model = nn.Sequential(*sequence)
        
    def forward(self, input):
        return self.model(input)


class UnetGenerator224(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator224, self).__init__() 
        
        # Encoder layers - 5 layers for 224x224
        self.e1 = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        
        self.e2 = nn.Sequential( 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        self.e3 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        self.e4 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        self.e5 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8)
        )
        
        # Decoder layers with skip connections - 5 layers
        self.d1 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.Dropout2d(0.5)
        )
        
        # After concatenation with e4: ngf*8*2 channels
        self.d2 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4)
        )
        
        # After concatenation with e3: ngf*4*2 channels
        self.d3 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2)
        )
        
        # After concatenation with e2: ngf*2*2 channels
        self.d4 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf)
        )
        
        # After concatenation with e1: ngf*2 channels
        self.d5 = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)  # 112x112
        e2 = self.e2(e1)  # 56x56
        e3 = self.e3(e2)  # 28x28
        e4 = self.e4(e3)  # 14x14
        e5 = self.e5(e4)  # 7x7
        
        # Decoder with skip connections
        d1 = self.d1(e5)  # 14x14
        d1_cat = torch.cat([d1, e4], dim=1)
        
        d2 = self.d2(d1_cat)  # 28x28
        d2_cat = torch.cat([d2, e3], dim=1)
        
        d3 = self.d3(d2_cat)  # 56x56
        d3_cat = torch.cat([d3, e2], dim=1)
        
        d4 = self.d4(d3_cat)  # 112x112
        d4_cat = torch.cat([d4, e1], dim=1)
        
        d5 = self.d5(d4_cat)  # 224x224
        
        return d5


if __name__ == "__main__":
    # Initialize models
    generator = UnetGenerator224(input_nc=1, output_nc=1, ngf=64)
    discriminator = NLayerDiscriminator(input_nc=1, output_nc=1, ndf=64, n_layers=3)

    # Example forward pass
    batch_size = 4
    img_size = 224
    input_image = torch.randn(batch_size, 1, img_size, img_size)
    target_image = torch.randn(batch_size, 1, img_size, img_size)



    # Generator forward pass
    generated_image = generator(input_image)

    # Discriminator forward pass
    input = torch.cat([input_image, target_image], dim=1)
    disc_output = discriminator(input)

    print(f"Generated image shape: {generated_image.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")