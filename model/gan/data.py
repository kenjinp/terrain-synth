from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch


def dataset(downsample_size, number_of_datasamples, elevations_dataset_path, batch_size):
    transform = transforms.v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1),
        v2.RandomResizedCrop(
            size=(downsample_size, downsample_size), antialias=True),
        # v2.Resize((DOWNSAMPLE_SIZE, DOWNSAMPLE_SIZE), antialias=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])

    # add transformation directly
    dataset = datasets.ImageFolder(
        elevations_dataset_path, transform=transform)

    subset = list(range(0, number_of_datasamples))
    training_subset = torch.utils.data.Subset(dataset, subset)

    dataloader = torch.utils.data.DataLoader(
        training_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def print_examples(dataloader):
    # Print some examples of our dataset
    for batch_index, real_imgs in enumerate(dataloader):
        if (batch_index > 0):
            break
        image_batch = real_imgs[0]
        for i in range(4):
            img = image_batch[i][0]
            print(image_batch[i].shape)
            print(img)
            print(f"mean={img.mean()} max={img.max()} min={img.min()}")
            plt.imshow(img, cmap="gray")
            plt.show()
            plt.hist(img.flatten())
            plt.axvline(img.mean())
            plt.show()
