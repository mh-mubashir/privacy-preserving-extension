import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------
# 1. ResNetLikeStem: [B, 3, 256, 256] -> [B, 64, 64, 64]
# ------------------------------------------------
class ResNetLikeStem(nn.Module):
    def __init__(self):
        super().__init__()
        # First conv: 256 -> 128 (using stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # One additional downsampling layer (stride=2):
        # 128 -> 64
        self.down1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
    def forward(self, x):
        x = self.conv1(x)   # [B, 64, 128, 128]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.down1(x)   # [B, 64, 64, 64]
        x = self.bn2(x)
        x = self.relu(x)
        return x

# ------------------------------------------------
# 2. Conv Block (Downsampling): Conv2d -> BatchNorm -> ReLU
# ------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        """
        This block applies a convolution with the given stride to downsample,
        followed by BatchNorm and ReLU.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

# ------------------------------------------------
# 3. Deconv Block (Upsampling): ConvTranspose2d -> BatchNorm -> ReLU
# ------------------------------------------------
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        """
        This block upsamples the feature map using a transposed convolution,
        followed by BatchNorm and ReLU.
        """
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding)
        self.bn     = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return self.relu(x)

# ------------------------------------------------
# 4. Full Model: Chain the modules as desired
# ------------------------------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet-like stem: [B, 3, 256, 256] -> [B, 64, 64, 64]
        self.stem = ResNetLikeStem()
        
        # Conv blocks (downsampling)
        # [B, 64, 64, 64] -> [B, 32, 32, 32]
        self.conv1 = ConvBlock(64, 32, kernel_size=3, stride=2, padding=1)
        # [B, 32, 32, 32] -> [B, 16, 16, 16]
        self.conv2 = ConvBlock(32, 16, kernel_size=3, stride=2, padding=1)
        
        # Deconv blocks (upsampling)
        # [B, 16, 16, 16] -> [B, 8, 32, 32]
        self.deconv1 = DeconvBlock(16, 8, kernel_size=2, stride=2)
        # [B, 8, 32, 32] -> [B, 3, 64, 64]
        self.deconv2 = DeconvBlock(8, 3, kernel_size=2, stride=2)
        
    def forward(self, x):
        # Stem
        x = self.stem(x)      # shape: [B, 64, 64, 64]
        # Downsampling
        x = self.conv1(x)     # shape: [B, 32, 32, 32]
        x = self.conv2(x)     # shape: [B, 16, 16, 16]
        # Upsampling
        x = self.deconv1(x)   # shape: [B, 8, 32, 32]
        x = self.deconv2(x)   # shape: [B, 3, 64, 64]
        return x

# ------------------------------------------------
# 5. Quick Test
# ------------------------------------------------
if __name__ == "__main__":
    # Create a dummy tensor with shape [batch, 3, 512, 512]
    dummy_input = torch.randn(1, 3, 256, 256)
    model = Encoder()
    output = model(dummy_input)
    print("Output shape:", output.shape)
    # Expected output shape: [1, 4, 64, 64]
