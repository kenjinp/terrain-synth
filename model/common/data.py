from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch
import math


def get_loader(image_size, channels_img, batch_size, number_of_datasamples, num_workers, elevations_dataset_path):
    transform = transforms.v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(num_output_channels=1),
        # This is to prevent focusing too much on the edges?
        v2.RandomResizedCrop(
            size=(image_size, image_size), antialias=True),
        # prevent overfitting by flipping things around
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.Normalize([0.5 for _ in range(channels_img)],
                     [0.5 for _ in range(channels_img)]),
    ])

    # add transformation directly
    dataset = datasets.ImageFolder(
        elevations_dataset_path, transform=transform)

    subset = list(range(0, number_of_datasamples))
    training_subset = torch.utils.data.Subset(dataset, subset)

    loader = torch.utils.data.DataLoader(
        training_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader, dataset
