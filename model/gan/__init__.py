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
from .train import train


class GAN:
    def __init__(self,
                 device,
                 batch_size=64,
                 num_gpu=1,
                 num_channels=1,
                 latent_dim=100,
                 num_features=64,
                 optimizer_learning_rate=3e-4,
                 critic_iterations=10,
                 lambda_gp=10
                 ):
        self.generator = Generator(
            num_gpu, num_channels, latent_dim, num_features).to(device)
        self.discriminator = Discriminator(
            num_gpu, num_channels, num_features).to(device)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.critic_iterations = critic_iterations
        self.lambda_gp = lambda_gp
        self.device = device
        self.learning_rate = optimizer_learning_rate

        # Optimizers
        self.optimizer_G = optim.RMSprop(
            self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_D = optim.RMSprop(
            self.discriminator.parameters(), lr=self.learning_rate)
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

    def fake_generator_input(self):
        return torch.randn(self.batch_size, self.latent_dim, 1, 1, device=self.device)

    def fake_discriminator_input(self):
        noise = self.fake_generator_input()
        return self.disciminator(noise)

    def generate_image(self):
        noise = self.fake_generator_input()
        with torch.no_grad():
            fake = self.generator(noise)
            img = fake[0]
            img = img.squeeze()
            img_npy = img.detach().cpu().numpy()
            return img_npy

    def show_image(self, img_npy, label):
        # save_image(img, f"{self.image_path}/%s.png" %
        #            label, nrow=5, normalize=True)
        plt.imshow(img_npy, cmap="gray")
        plt.show()

    def save_image(self, label):
        noise = self.fake_generator_input()
        with torch.no_grad():
            fake = self.generator(noise)
            img = fake[0]
            img = img.squeeze()
            save_image(img, f"{self.image_path}/%s.png" %
                       label, nrow=5, normalize=True)

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
        print(f"Loaded checkpoint from {timestamp}")

    def save_checkpoint(self):
        print(f"Saving checkpoint to {self.checkpoint_path}")
        torch.save({
            'timestamp': datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S"),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_D_state_dict': self.optimizer_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_D.state_dict(),
        }, self.checkpoint_path)

    def export(self):
        export(self.generator, self.fake_generator_input(),
               f"{self.path}/generator.onnx")
