from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch
import math


def get_loader_maker(elevations_dataset_path, image_size, channels_img, batch_sizes, number_of_datasamples, num_workers):
    # do stuff
    def get_loader(image_size):
        transform = transforms.v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Grayscale(num_output_channels=1),
            v2.RandomResizedCrop(
                size=(image_size, image_size), antialias=True),
            v2.Normalize([0.5 for _ in range(channels_img)],
                         [0.5 for _ in range(channels_img)]),
        ])

        batch_size = batch_sizes[int(
            math.log2(image_size / 4))]

        # add transformation directly
        dataset = datasets.ImageFolder(
            elevations_dataset_path, transform=transform)

        subset = list(range(0, number_of_datasamples))
        training_subset = torch.utils.data.Subset(dataset, subset)

        loader = torch.utils.data.DataLoader(
            training_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        return loader, dataset
    return get_loader
