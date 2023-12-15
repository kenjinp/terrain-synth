from progan import model, export, train, constants, loader
import torch
import warnings

conf = constants.config()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

warnings.filterwarnings("ignore")


def main():
    gan = model.ProGAN(
        z_dim=conf["z_dim"],
        in_channels=conf["in_channels"],
        img_channels=conf["channels_img"],
        learning_rate=conf["learning_rate"],
        device=device
    )
    get_loader = loader.get_loader_maker(
        path, image_size, channels_img, batch_sizes, number_of_datasamples, num_workers
    )

    train.train(gan, get_loader, batch_sizes, start_train_at_img_size,
                progressive_epochs, save_model, lambda_gp, critic_iterations)


if __name__ == "__main__":
    main()
