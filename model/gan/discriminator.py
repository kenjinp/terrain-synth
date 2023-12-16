import torch
import torch.nn as nn
from .network import Network
from .generator import Generator


class Discriminator(Network):
    def __init__(self, image_channels, in_channels, input_image_size):
        super(Discriminator, self).__init__()
        modules = nn.ModuleList(
            [nn.Conv2d(image_channels, in_channels,
                       kernel_size=4, stride=2, padding=1),
             nn.LeakyReLU(0.2)]
        )

        current_image_size = input_image_size / 2
        target_image_size = 4
        feature_size_multiplier = 1

        def conv2d_output_size(input_size, kernel_size, stride, padding):
            """
            Calculate the output size of a Conv2d layer.

            Parameters:
            - input_size: Tuple representing the input size (H_in, W_in).
            - kernel_size: Tuple representing the kernel size (kernel_size_h, kernel_size_w).
            - stride: Tuple representing the stride (stride_h, stride_w).
            - padding: Tuple representing the padding (padding_h, padding_w).

            Returns:
            - Tuple representing the output size (H_out, W_out).
            """
            H_in, W_in = input_size
            kernel_size_h, kernel_size_w = kernel_size
            stride_h, stride_w = stride
            padding_h, padding_w = padding

            H_out = ((H_in + 2 * padding_h - kernel_size_h) // stride_h) + 1
            W_out = ((W_in + 2 * padding_w - kernel_size_w) // stride_w) + 1

            return H_out, W_out

        input_features_size = in_channels
        while current_image_size > target_image_size:
            feature_size_multiplier = min(16, feature_size_multiplier * 2)
            output_feature_size = in_channels * feature_size_multiplier

            kernel_size = 4
            stride = 2
            padding = 1

            h, w = conv2d_output_size(
                (current_image_size, current_image_size), (kernel_size, kernel_size), (stride, stride), (padding, padding))

            current_image_size = h

            modules.append(self._block(
                in_channels=input_features_size,
                out_channels=output_feature_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

            input_features_size = output_feature_size

        self.disc = nn.Sequential(
            *modules,
            nn.Conv2d(in_channels * feature_size_multiplier,
                      1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    IMG_CHANNELS = 3
    IMG_SIZE = 1024
    img_size = IMG_SIZE
    gen = Generator(
        Z_DIM,
        IN_CHANNELS,
        IMG_CHANNELS,
        target_image_size=IMG_SIZE
    )
    critic = Discriminator(
        IMG_CHANNELS,
        IN_CHANNELS,
        input_image_size=IMG_SIZE
    )
    print(critic)
    print(f"Testing img size {img_size}x{img_size}")
    x = torch.randn((1, Z_DIM, 1, 1))
    z = gen(x)
    assert z.shape == (1, IMG_CHANNELS, img_size, img_size)
    out = critic(z)
    print(out.shape)
    assert out.shape == (1, 1, 1, 1)
    print(f"Success! At img_size {img_size}")
