import torch
import torch.nn as nn
import datetime

from torchsummary import summary
from .network import Network


class Generator(Network):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            self._block(features_g * 2, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
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
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# class Generator(nn.Module):
#     def __init__(self, num_features, num_gpu=1, num_channels=1, latent_dim=100):
#         super(Generator, self).__init__()
#         self.num_gpu = num_gpu
#         self.latent_dim = latent_dim
#         self.num_features = num_features
#         self.num_channels = num_channels
#         timestamp = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
#         self.id = f"terrain-gen-{timestamp}"
#         self.ngpu = num_gpu

#         def generator_sequential(output_size, latent_dim, num_channels, num_features):
#             layers = []
#             # Calculate the number of transposed convolutions needed to go from input_size to output_size
#             current_size = 1
#             stride = 2
#             multiplier = 4
#             while current_size < output_size:
#                 layers.append(nn.ConvTranspose2d(
#                     latent_dim, num_features * multiplier, 4, stride, 1, bias=False))
#                 layers.append(nn.BatchNorm2d(num_features * multiplier))
#                 layers.append(nn.ReLU(True))
#                 latent_dim = num_features * multiplier
#                 num_features //= 2
#                 current_size *= stride
#                 # print(f"current_size: {current_size}")
#                 # print(f"stride: {stride}")
#                 # print(f"latent_dim: {latent_dim}")
#                 # print(f"num_features: {num_features}")
#             # Append the final convolution layer to ensure the output size matches the specified output_size
#             # # Append the final convolution layer
#             layers.append(nn.ConvTranspose2d(
#                 latent_dim, num_channels, 4, stride, 1, bias=False))
#             layers.append(nn.Tanh())

#             return nn.Sequential(*layers)

#         # self.main = generator_sequential(
#         #     num_features, latent_dim, num_channels, num_features)

#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(latent_dim, num_features *
#                                8, 4, 1, , bias=False),
#             nn.BatchNorm2d(num_features * 8),
#             nn.ReLU(True),
#             # state size. ``(num_features*8) x 4 x 4``
#             nn.ConvTranspose2d(
#                 num_features * 8, num_features * 4, 4, 2, 1, bias=False),
#             # nn.BatchNorm2d(num_features * 4),
#             nn.ReLU(True),
#             # state size. ``(num_features*4) x 8 x 8``
#             nn.ConvTranspose2d(
#                 num_features * 4, num_features * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_features * 2),
#             nn.ReLU(True),
#             # state size. ``(num_features*2) x 16 x 16``
#             nn.ConvTranspose2d(
#                 num_features * 2, num_features, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(num_features),
#             nn.ReLU(True),
#             # state size. ``(num_features) x 32 x 32``
#             nn.ConvTranspose2d(num_features, num_channels,
#                                4, 4, 1, bias=False),
#             nn.Tanh()
#             # state size. ``(nc) x 64 x 64``
#         )
#         self.init_weights()
#         print(self)

#     def summary(self):
#         summary(self, input_size=(self.latent_dim,
#                                   self.num_features, self.num_features))

#     def forward(self, input):
#         return self.main(input)

#     def init_weights(self):
#         def weights_init(m):
#             classname = m.__class__.__name__
#             if classname.find('Conv') != -1:
#                 nn.init.normal_(m.weight.data, 0.0, 0.02)
#             elif classname.find('BatchNorm') != -1:
#                 nn.init.normal_(m.weight.data, 1.0, 0.02)
#                 nn.init.constant_(m.bias.data, 0)
#         self.apply(weights_init)

#     def save(self, path):
#         torch.save(self.state_dict(), f"{self.path}/{self.id}.pt")
