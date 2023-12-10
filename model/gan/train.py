import torch
import torch.nn as nn

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.


def train(gan, dataloader, epochs, save_interval=10, clip_threshold=0.01, noise_std=0.01):
    def noisify(images):
        return images + torch.randn_like(images).to(gan.device) * noise_std

    for epoch in range(epochs):

        for batch_index, (image_batches, _) in enumerate(dataloader):
            i = batch_index
            batch_size = image_batches.size(0)
            image_batches = image_batches.to(gan.device)

            # Configure input
            real_imgs = image_batches.to(gan.device)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            gan.optimizer_D.zero_grad()

            # Generate a batch of images
            # Generate batch of latent vectors
            noise = torch.randn(batch_size, gan.latent_dim,
                                1, 1, device=gan.device)
            fake_imgs = gan.generator(noise)

            # Wgan critic training
            for _ in range(gan.critic_iterations):
                outputs_real = gan.discriminator(noisify(real_imgs))
                outputs_fake = gan.discriminator(noisify(fake_imgs.detach()))

                loss_D = outputs_fake.mean() - outputs_real.mean()

                # Gradient penalty
                # gradient_penalty = gan.gradient_penalty(real_imgs.data, fake_imgs.data)

                # Backward pass
                loss_D_total = loss_D

                # why is this necessary?
                loss_D_total.backward()
                gan.optimizer_D.step()

                # Clip discriminator weights
                for p in gan.discriminator.parameters():
                    p.data.clamp_(-clip_threshold, clip_threshold)

            # -----------------
            #  Train Generator
            # -----------------

            ##

            outputs = gan.discriminator(fake_imgs)

            # Compute the generator loss
            loss_G = -outputs.mean()
            gan.optimizer_G.zero_grad()

            # Backward pass
            loss_G.backward()
            gan.optimizer_G.step()

            if batch_index % 10 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, batch_size, loss_D.item(), loss_G.item())
                )

        # Save model at specific intervals
        if epoch % save_interval == 0:
            gan.save_image(epoch)
            gan.save_checkpoint()
            gan.export()
