from gan import discriminator, generator, GAN, export, train
import torch
import data
import constants

conf = constants.config()
print(conf)

dataloader = data.dataset(
    conf["downsample_size"],
    conf["number_of_datasamples"],
    conf["elevations_dataset_path"],
    conf["batch_size"],
)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device {device}")

myGAN = GAN(device=device)

train(myGAN, dataloader, 10_000)

# fake = myGAN.fake_generator_input()
# out = myGAN.generator(fake)
# export(myGAN.generator, fake,
#        f"{myGAN.path}/{myGAN.generator.id}.onnx")
# print(out)
# print(out.shape)
