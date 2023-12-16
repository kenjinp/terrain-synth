import torch
import torch.nn as nn
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import gradient_penalty, plot_to_tensorboard


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    lambda_gp,
    progressive_epochs,
    scaler_gen,
    scaler_critic,
    fixed_noise,
    device,
    image_size,
    critic_iterations
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(device)
        current_batch_size = real.shape[0]

        # Train Critic: E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        # decreasing the gap there
        for _ in range(critic_iterations):
            noise = torch.randn(current_batch_size, gen.z_dim, 1, 1).to(device)
            with torch.cuda.amp.autocast():
                fake = gen(noise, alpha, step)
                critic_real = critic(real, alpha, step)
                critic_fake = critic(fake.detach(), alpha, step)
                gp = gradient_penalty(
                    critic, real, fake.detach(), alpha, step, device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + lambda_gp * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )
            opt_critic.zero_grad()
            scaler_critic.scale(loss_critic).backward()
            scaler_critic.step(opt_critic)
            scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += current_batch_size / \
            (len(dataset) * progressive_epochs[step] * 0.5)
        alpha = min(alpha, 1)

        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise, alpha, step) * 0.5 + 0.5

            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
                image_size
            )
            tensorboard_step += 1

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
            loss_gen=loss_gen.item()
        )
    return tensorboard_step, alpha


def train(gan, get_loader, batch_sizes, start_train_at_img_size, progressive_epochs, save_model, lambda_gp, critic_iterations):
    device = gan.device
    gen = gan.generator
    critic = gan.discriminator
    opt_gen = gan.optimizer_G
    opt_critic = gan.optimizer_D
    scaler_gen = gan.scalar_generator
    scaler_critic = gan.scalar_critic
    writer = gan.writer

    step = int(math.log2(start_train_at_img_size / 4))

    progressive_epochs = [progressive_epochs] * len(batch_sizes)
    fixed_noise = torch.randn(8, gan.z_dim, 1, 1).to(gan.device)
    tensorboard_step = gan.tensorboard_step

    print(f"Progressive epochs: {progressive_epochs}")

    for num_epochs in progressive_epochs[step:]:
        # This should be set to 1 at some point?
        image_size = 4*2**step
        alpha = 1e-5
        loader, dataset = get_loader(image_size)
        print(f"Image size: {image_size}x{image_size}")
        print(f"Alpha: {alpha}")
        print(f"Step: {step}")

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                lambda_gp,
                progressive_epochs,
                scaler_gen,
                scaler_critic,
                fixed_noise,
                device,
                image_size,
                critic_iterations
            )

            if save_model:
                with torch.no_grad():
                    gan.save_checkpoint()
                    # I',m not sure about the step stuff...
                    gan.export(step)
                    gan.generate_example_plot(step, f"epoch-{epoch}")

        step += 1
