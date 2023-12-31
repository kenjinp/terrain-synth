
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import datetime
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from math import log2

from .generator import Generator
from .discriminator import Discriminator
from .utils import load_checkpoint, save_checkpoint


class ProGAN:
    def __init__(self,
                 z_dim,
                 in_channels,
                 img_channels,
                 learning_rate,
                 device="cpu"
                 ):
        self.z_dim = z_dim
        self.in_channels = in_channels
        self.img_channels = img_channels
        self.learning_rate = learning_rate
        self.device = device

        self.generator = Generator(
            z_dim,
            in_channels,
            img_channels
        ).to(device)
        self.discriminator = Discriminator(
            in_channels,
            img_channels
        ).to(device)

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

        self.id = f"terrain-ProGAN"

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
        make_path(self.path / "examples")
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

    def save_image(self, fixed_fakes, label):
        img = fixed_fakes[0]
        img = img.squeeze()
        path = f"{self.image_path}/%s.png" % label
        print(f"Saving image to {path}")
        save_image(img, path, nrow=5, normalize=True)

    def fake_generator_input(self, batch_size):
        noise = torch.randn(batch_size, self.z_dim, 1, 1).to(self.device)
        return noise

    def generate_image(self, step, batch_size):
        noise = self.fake_generator_input(batch_size)
        with torch.no_grad():
            fake = self.generator(noise, 1, step)
            return fake

    def show_image(self, image_size):
        img_npy = self.generate_image(1, image_size)
        plt.imshow(img_npy, cmap="gray")
        plt.show()

    def generate_example_plot(self, step, name, show=False):
        columns = 4
        rows = 5
        fig = plt.figure(figsize=(9, 13))
        images = self.generate_image(step, columns * rows)
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

    def export(self, step):
        dummy_input_1 = torch.randn(8, self.z_dim, 1, 1).to(self.device)

        # This is how we would call the PyTorch model
        # self.generator(dummy_input_1, 1.0, step)

        # This is how to export it with multiple inputs
        alpha = 1
        args = dummy_input_1, alpha, step
        print(args)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            torch.onnx.export(self.generator,
                              args=args,
                              f=f"{self.path}/generator.onnx",
                              input_names=["latent", "alpha", "step"],
                              output_names=["output"])


# Let's see if things run!!!!
if __name__ == "__main__":
    Z_DIM = 50
    IN_CHANNELS = 256
    IMG_CHANNELS = 3
    gen = Generator(
        Z_DIM,
        IN_CHANNELS,
        img_channels=IMG_CHANNELS)
    critic = Discriminator(
        IN_CHANNELS,
        img_channels=IMG_CHANNELS)

    for img_size in [4, 8, 16, 32, 64, 256, 512, 1024]:
        num_steps = int(log2(img_size / 4))
        print(f"Testing img size {img_size}x{img_size} with {num_steps} steps")
        x = torch.randn((1, Z_DIM, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, IMG_CHANNELS, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)
        print(f"Success! At img_size {img_size}")
