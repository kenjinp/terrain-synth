import torch
import torch.nn as nn
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import gradient_penalty


def train(gan, dataloader, num_epochs, z_dim, lambda_gp,  save_interval=10, critic_iterations=5):
    device = gan.device
    gen = gan.generator
    critic = gan.discriminator
    opt_gen = gan.optimizer_G
    opt_critic = gan.optimizer_D

    # for tensorboard plotting
    fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
    writer_real = SummaryWriter(f"{gan.path}/logs/real")
    writer_fake = SummaryWriter(f"{gan.path}/fake")
    step = 0

    for epoch in range(num_epochs):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(tqdm(dataloader)):
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
            if batch_idx % 100 == 0 and batch_idx > 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    # take out (up to) 32 examples
                    img_grid_real = torchvision.utils.make_grid(
                        real[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(
                        fake[:32], normalize=True)

                    writer_real.add_image(
                        "Real", img_grid_real, global_step=step)
                    writer_fake.add_image(
                        "Fake", img_grid_fake, global_step=step)

                step += 1
        if epoch % save_interval == 0:
            with torch.no_grad():
                batch_size = 1
                gan.save_image(epoch, batch_size)
                gan.save_checkpoint()
                gan.export(batch_size)

    # def noisify(images):
    #     return images + torch.randn_like(images).to(gan.device) * noise_std

    # # show example from generator
    # # gan.show_image(gan.generate_image())

    # # return

    # last_epoch_time = datetime.now()
    # for epoch in range(epochs):
    #     datalength = len(dataloader.dataset)
    #     num_batches = math.ceil(datalength / dataloader.batch_size)
    #     last_batch_time = datetime.now()
    #     batch_times = []
    #     for batch_index, (image_batches, _) in enumerate(dataloader):
    #         i = batch_index
    #         batch_size = image_batches.size(0)
    #         image_batches = image_batches.to(gan.device)

    #         # Configure input
    #         real_imgs = image_batches.to(gan.device)
    #         # ---------------------
    #         #  Train Discriminator
    #         # ---------------------

    #         gan.optimizer_D.zero_grad()

    #         # Generate a batch of images
    #         # Generate batch of latent vectors
    #         noise = gan.fake_generator_input()
    #         fake_imgs = gan.generator(noise)

    #         # Wgan critic training
    #         for _ in range(gan.critic_iterations):
    #             outputs_real = gan.discriminator(noisify(real_imgs))
    #             outputs_fake = gan.discriminator(noisify(fake_imgs.detach()))

    #             loss_D = outputs_fake.mean() - outputs_real.mean()

    #             # Gradient penalty
    #             # gradient_penalty = gan.gradient_penalty(real_imgs.data, fake_imgs.data)

    #             # Backward pass
    #             loss_D_total = loss_D

    #             # why is this necessary?
    #             loss_D_total.backward()
    #             gan.optimizer_D.step()

    #             # Clip discriminator weights
    #             for p in gan.discriminator.parameters():
    #                 p.data.clamp_(-clip_threshold, clip_threshold)

    #         # -----------------
    #         #  Train Generator
    #         # -----------------

    #         ##

    #         outputs = gan.discriminator(fake_imgs)

    #         # Compute the generator loss
    #         loss_G = -outputs.mean()
    #         gan.optimizer_G.zero_grad()

    #         # Backward pass
    #         loss_G.backward()

    #         now = datetime.now()
    #         delta = now - last_batch_time
    #         last_batch_time = now
    #         batch_times.append(delta.seconds)

    #         if batch_index % 10 == 0:
    #             average_batch_time = sum(batch_times) / len(batch_times)
    #             print(
    #                 "[Epoch %d/%d] [Batch %d/%d:%ds] [D loss: %f] [G loss: %f]"
    #                 % (epoch, epochs, batch_index, num_batches, average_batch_time, loss_D.item(), loss_G.item())
    #             )

    #     now = datetime.now()
    #     print(f"Epoch time: {now - last_epoch_time}")
    #     last_epoch_time = now
    #     # Save model at specific intervals
        # if epoch % save_interval == 0:
        #     gan.save_image(epoch)
        #     gan.save_checkpoint()
        #     gan.export()
