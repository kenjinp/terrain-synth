import torch.nn as nn
import torch

from .network import Network
from common.blocks import PixelNorm, WSConv2d


class Generator(Network):
    def __init__(self, z_dim, in_channels, image_channels, target_image_size):
        super(Generator, self).__init__()
        modules = nn.ModuleList()
        modules.append(self._block(z_dim, in_channels *
                       16, kernel_size=4, stride=1, padding=0))

        current_image_size = 4
        feature_size_multiplier = 16

        def calculate_output_size(current_image_size, stride, padding, kernel_size, output_padding):
            return (current_image_size - 1) * stride - 2 * padding + kernel_size + output_padding

        while current_image_size < target_image_size / 2:
            input_features_size = in_channels * feature_size_multiplier
            feature_size_multiplier = max(2, feature_size_multiplier // 2)
            kernel_size = 4
            stride = 2
            padding = 1
            output_padding = 0
            output_feature_size = in_channels * feature_size_multiplier
            current_image_size = calculate_output_size(
                current_image_size, stride, padding, kernel_size, output_padding)
            modules.append(self._block(
                in_channels=input_features_size,
                out_channels=output_feature_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))  # img: 8x8

        self.net = nn.Sequential(
            *modules,
            nn.ConvTranspose2d(
                in_channels * 2, image_channels, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            PixelNorm(),

        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    IMG_CHANNELS = 3
    gen = Generator(
        Z_DIM,
        IN_CHANNELS,
        IMG_CHANNELS,
        target_image_size=1024
    )
    example_output = gen(torch.randn(1, Z_DIM, 1, 1))
    print(gen)
    print(example_output.shape)
