
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

from .generator import Generator
from .discriminator import Discriminator
from .export import export
from .utils import gradient_penalty


class GAN:
    def __init__(self,
                 z_dim,
                 channels_img,
                 features_d,
                 features_g,
                 learning_rate,
                 device="cpu"
                 ):
        self.generator = Generator(
            channels_noise=z_dim,
            channels_img=channels_img,
            features_g=features_g,
        ).to(device)
        self.discriminator = Discriminator(
            channels_img=channels_img,
            features_d=features_d
        ).to(device)
        self.z_dim = z_dim
        self.channels_img = channels_img
        self.features_d = features_d
        self.device = device

        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

        timestamp = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
        self.id = f"terrain-gan"

        def make_path(path):
            if not os.path.exists(path):
                os.makedirs(path)
            return path

        # Make a folder to store all the goodies
        self.path = make_path(Path(os.getcwd()).resolve() / f".{self.id}")
        self.image_path = make_path(self.path / "images")
        self.checkpoint_path = f"{self.path}/checkpoint-{self.id}.pt"

        # Load checkpoint if it exists
        if (os.path.exists(self.checkpoint_path)):
            self.reload_checkpoint(self.checkpoint_path)
        else:
            self.generator.initialize_weights()
            self.discriminator.initialize_weights()

        self.generator.train()
        self.discriminator.train()

    def fake_generator_input(self, batch_size):
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        return noise

    def generate_image(self, batch_size):
        noise = self.fake_generator_input(batch_size)
        with torch.no_grad():
            fake = self.generator(noise)
            img = fake[0]
            img = img.squeeze()
            img_npy = img.detach().cpu().numpy()
            print(fake.shape)
            print(img.shape)
            return img_npy

    def show_image(self, img_npy):
        # save_image(img, f"{self.image_path}/%s.png" %
        #            label, nrow=5, normalize=True)
        plt.imshow(img_npy, cmap="gray")
        plt.show()

    def save_image(self, label, batch_size):
        noise = self.fake_generator_input(batch_size)
        with torch.no_grad():
            fake = self.generator(noise)
            img = fake[0]
            img = img.squeeze()
            path = f"{self.image_path}/%s.png" % label
            print(f"Saving image to {path}")
            save_image(img, path, nrow=5, normalize=True)

    def reload_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        timestamp = checkpoint['timestamp']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(
            checkpoint['discriminator_state_dict'])
        # self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        # self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.generator.train()
        self.discriminator.train()
        print(f"=> Loaded checkpoint from {timestamp}")

    def save_checkpoint(self):
        print(f"=> Saving checkpoint to {self.checkpoint_path}")
        torch.save({
            'timestamp': datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S"),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_D_state_dict': self.optimizer_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_D.state_dict(),
        }, self.checkpoint_path)

    def export(self, batch_size):
        export(self.generator, self.fake_generator_input(batch_size),
               f"{self.path}/generator.onnx")

    def test():
        N, in_channels, H, W = 8, 3, 64, 64
        noise_dim = 100
        features = 12
        learning_rate = 2e-4
        gan = GAN(
            noise_dim,
            in_channels,
            features,
            features,
            learning_rate,
        )
        x = torch.randn((N, in_channels, H, W))
        disc = gan.discriminator
        assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
        gen = gan.generator
        z = torch.randn((N, noise_dim, 1, 1))
        assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"


if __name__ == "__main__":
    GAN.test()
