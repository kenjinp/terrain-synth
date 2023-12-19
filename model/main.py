from common import data, constants
from gan import model, train
import torch
import warnings


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
    gan = model.GAN(
        z_dim=z_dim,
        in_channels=in_channels,
        image_channels=channels_img,
        learning_rate=learning_rate,
        target_image_size=image_size,
        device=device
    )

    gan.export()

    # dataloader, dataset = data.get_loader(
    #     image_size, channels_img, batch_size, number_of_datasamples, num_workers, path
    # )

    # train.train(gan,
    #             dataloader, num_epochs,
    #             lambda_gp,
    #             save_model, critic_iterations)


if __name__ == "__main__":
    main()
