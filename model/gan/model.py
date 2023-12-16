
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .generator import Generator
from .discriminator import Discriminator
from .export import export
from .gp import gradient_penalty
from common.checkpoint import load_checkpoint, save_checkpoint


class GAN:
    def __init__(self,
                 z_dim,
                 image_channels,
                 in_channels,
                 learning_rate,
                 target_image_size,
                 device="cpu"
                 ):
        self.generator = Generator(
            z_dim=z_dim,
            image_channels=image_channels,
            in_channels=in_channels,
            target_image_size=target_image_size
        ).to(device)
        self.discriminator = Discriminator(
            image_channels=image_channels,
            in_channels=in_channels,
            input_image_size=target_image_size
        ).to(device)
        self.z_dim = z_dim
        self.channels_img = image_channels
        self.in_channels = in_channels
        self.device = device
        self.learning_rate = learning_rate
        self.target_image_size = target_image_size

        # Optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))

        # for float16 training?
        self.scalar_critic = torch.cuda.amp.GradScaler()
        self.scalar_generator = torch.cuda.amp.GradScaler()

        timestamp = datetime.datetime.now().strftime("%m-%d-%y_%H:%M:%S")
        self.timestamp = timestamp
        self.id = f"terrain-WGAN"

        def make_path(path):
            if not os.path.exists(path):
                os.makedirs(path)
            return path

        # Make a folder to store all the goodies
        self.path = make_path(Path(os.getcwd()).resolve() / f".{self.id}")
        self.image_path = make_path(self.path / "images")
        self.writer_path = self.path / f"logs/{timestamp}"
        self.checkpoint_path = self.path / "checkpoint"
        make_path(self.checkpoint_path)
        self.examples_path = make_path(self.path / "examples")

        self.writer = SummaryWriter(log_dir=self.writer_path)

        # Load checkpoint if it exists
        if (os.path.exists(self.checkpoint_path / "gen.pt")):
            self.reload_checkpoint()
        else:
            self.generator.initialize_weights()
            self.discriminator.initialize_weights()
            self.tensorboard_step = 0

        self.generator.train()
        self.discriminator.train()

    def fake_generator_input(self, batch_size):
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        return noise

    def generate_image(self, batch_size):
        noise = self.fake_generator_input(batch_size)
        with torch.no_grad():
            fake = self.generator(noise)
            return fake

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

    def reload_checkpoint(self):
        print(f"=> Loading checkpoint to {self.checkpoint_path}")

        load_checkpoint(self.checkpoint_path / "gen.pt",
                        self.generator, self.optimizer_G, self.learning_rate, self.device)
        load_checkpoint(self.checkpoint_path / "disc.pt",
                        self.discriminator, self.optimizer_D, self.learning_rate, self.device)

        if os.path.exists(self.checkpoint_path / "misc.pt"):
            misc = torch.load(self.checkpoint_path / "misc.pt")
            self.tensorboard_step = misc["tensorboard_step"]
        else:
            self.tensorboard_step = 0

    def save_checkpoint(self):
        print(f"=> Saving checkpoint to {self.checkpoint_path}")
        save_checkpoint(self.generator, self.optimizer_G,
                        self.checkpoint_path / "gen.pt")
        save_checkpoint(self.discriminator, self.optimizer_D,
                        self.checkpoint_path / "disc.pt")
        torch.save({
            "tensorboard_step": self.tensorboard_step,
        }, self.checkpoint_path / "misc.pt")

    def export(self, batch_size=1):
        export(self.generator, self.fake_generator_input(batch_size),
               f"{self.path}/generator.onnx")

    # def generate_examples(gan, steps, z_dim, device, truncation=0.7, n=100):
    #     """
    #     Tried using truncation trick here but not sure it actually helped anything, you can
    #     remove it if you like and just sample from torch.randn
    #     """
    #     gen = gan.generator
    #     gen.eval()
    #     for i in range(n):
    #         with torch.no_grad():
    #             noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(
    #                 1, z_dim, 1, 1)), device=device, dtype=torch.float32)
    #             img = gen(noise, alpha, steps)
    #             print("examples/img_{i}.png")
    #             save_image(img*0.5+0.5, gan.path / f"examples/img_{i}.png")
    #     gen.train()

    def generate_example_plot(self, name, show=False):
        columns = 4
        rows = 5
        fig = plt.figure(figsize=(9, 13))
        images = self.generate_image(columns * rows)
        ax = []
        for i, img in enumerate(images):
            img = img.squeeze()
            ax.append(fig.add_subplot(rows, columns, i+1))
            plt.imshow(img.cpu(), cmap="gray")

        if (show):
            plt.show()
        else:
            plt.savefig(self.examples_path /
                        f"{name}.png", bbox_inches='tight')

        # plt.show()  # finally, render the plot
