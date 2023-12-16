from progan import model, export, train, loader, utils
from common import constants
import torch
import warnings
import math


def main():

    conf = constants.config()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    path = conf["elevations_dataset_path"]
    image_size = conf["image_size"]
    channels_img = conf["channels_img"]
    number_of_datasamples = conf["number_of_datasamples"]
    num_workers = conf["num_workers"]
    batch_sizes = conf["batch_sizes"]
    start_train_at_img_size = conf["start_train_at_img_size"]
    progressive_epochs = conf["progressive_epochs_by_image_size"]
    save_model = conf["save_model"]
    lambda_gp = conf["lambda_gp"]
    critic_iterations = conf["critic_iterations"]
    z_dim = conf["z_dim"]
    num_epochs = conf["num_epochs"]
    batch_size = conf["batch_size"]
    learning_rate = conf["learning_rate"]
    in_channels = conf["in_channels"]

    warnings.filterwarnings("ignore")

    print(conf)
    print(f"Device: {device}")

    warnings.filterwarnings("ignore")

    gan = model.ProGAN(
        z_dim=conf["z_dim"],
        in_channels=conf["in_channels"],
        img_channels=conf["channels_img"],
        learning_rate=conf["learning_rate"],
        device=device
    )

    get_loader = loader.get_loader_maker(
        path, channels_img, batch_sizes, number_of_datasamples, num_workers
    )
    train.train(gan, get_loader, batch_sizes, start_train_at_img_size,
                progressive_epochs, save_model, lambda_gp, critic_iterations)


if __name__ == "__main__":
    main()
