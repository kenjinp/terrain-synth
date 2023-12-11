from gan import model, export, train
import torch
import data
import constants

conf = constants.config()

dataloader = data.dataset(
    conf["image_size"],
    conf["number_of_datasamples"],
    conf["elevations_dataset_path"],
    conf["batch_size"],
)

# data.print_examples(dataloader)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

gan = model.GAN(device=device,
                z_dim=conf["z_dim"],
                channels_img=conf["channels_img"],
                features_d=conf["features_critic"],
                features_g=conf["features_gen"],
                learning_rate=conf["learning_rate"])

train(gan, dataloader,
      conf["num_epochs"],
      conf["z_dim"],
      conf["lambda_gp"],
      save_interval=conf["save_interval"],
      critic_iterations=conf["critic_iterations"])
