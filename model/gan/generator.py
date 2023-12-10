import torch
import torch.nn as nn
import datetime


class Generator(nn.Module):
    def __init__(self, num_gpu=1, num_channels=1, latent_dim=100, num_features=64):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.latent_dim = latent_dim
        self.num_features = num_features
        timestamp = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
        self.id = f"terrain-gen-{timestamp}"
        self.ngpu = num_gpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, num_features *
                               8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_features * 8),
            nn.ReLU(True),
            # state size. ``(num_features*8) x 4 x 4``
            nn.ConvTranspose2d(
                num_features * 8, num_features * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(num_features * 4),
            nn.ReLU(True),
            # state size. ``(num_features*4) x 8 x 8``
            nn.ConvTranspose2d(
                num_features * 4, num_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features * 2),
            nn.ReLU(True),
            # state size. ``(num_features*2) x 16 x 16``
            nn.ConvTranspose2d(
                num_features * 2, num_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(True),
            # state size. ``(num_features) x 32 x 32``
            nn.ConvTranspose2d(num_features, num_channels,
                               4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )
        self.init_weights()

    def forward(self, input):
        return self.main(input)

    def init_weights(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        self.apply(weights_init)

    def save(self, path):
        torch.save(self.state_dict(), f"{self.path}/{self.id}.pt")
