import torch
from tqdm import tqdm
from .gp import gradient_penalty
from common.telemetry import plot_to_tensorboard


def train(gan, dataloader, num_epochs, lambda_gp, save_model, critic_iterations):
    device = gan.device
    gen = gan.generator
    critic = gan.discriminator
    opt_gen = gan.optimizer_G
    opt_critic = gan.optimizer_D
    z_dim = gan.z_dim

    # for tensorboard plotting
    fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
    step = 0

    for epoch in range(num_epochs):
        # Target labels not needed! <3 unsupervised
        loop = tqdm(dataloader, leave=True)
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(critic_iterations):
                noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) -
                      torch.mean(critic_fake)) + lambda_gp * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:

                with torch.no_grad():
                    fixed_fakes = gen(fixed_noise) * 0.5 + 0.5

                plot_to_tensorboard(
                    gan.writer,
                    loss_critic.item(),
                    loss_gen.item(),
                    real.detach(),
                    fixed_fakes.detach(),
                    tensorboard_step=step,
                )
                loop.set_postfix(
                    gp=gp.item(),
                    loss_critic=loss_critic.item(),
                    loss_gen=loss_gen.item(),
                )
                step += 1

        if epoch % 1 == 0 & save_model:
            with torch.no_grad():
                gan.save_checkpoint()
                gan.export()
                if epoch % 4 == 0 & save_model:
                    gan.generate_example_plot(f"epoch-{epoch}")
