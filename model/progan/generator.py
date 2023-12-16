import torch
import torch.nn as nn
import torch.nn.functional as F

from .network import Network
from ..common.blocks import WSConv2d, PixelNorm, ConvBlock

factors = [1, 1, 1, 1,  1/2, 1/4, 1/8, 1/16, 1/32]


class Generator(Network):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, kernel_size=4,
                               stride=1, padding=0),  # 1x1 to 4x4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels,
                     kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.z_dim = z_dim
        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([
            self.initial_rgb])

        for i in range(len(factors) - 1):
            # factors[i] -> factors[i+1]
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            self.prog_blocks.append(
                ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(
                WSConv2d(conv_out_channels, img_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upscaled_img, generated_img):
        # tanh -1, 1
        return torch.tanh(alpha * generated_img + (1-alpha) * upscaled_img)

    def forward(self, x, alpha, steps):
        # if steps = 0 (4x4) -> 1 (8x8) -> 2 (16x16) -> 3 (32x32) -> 4 (64x64) ...
        out = self.initial(x)  # 4x4
        if (steps == 0):
            return self.initial_rgb(out)
        # How many progressive blocks should we run it through
        for step in range(steps):
            upscaled_image = F.interpolate(out, scale_factor=2, mode="nearest")
            out = self.prog_blocks[step](upscaled_image)

        final_upscaled_image = self.rgb_layers[steps-1](upscaled_image)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled_image, final_out)
