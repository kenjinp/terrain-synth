from ..common.data import get_loader
from math import log2


def get_loader_maker(elevations_dataset_path, channels_img, batch_sizes, number_of_datasamples, num_workers):
    def loader_factory(image_size):
        batch_size = batch_sizes[int(log2(image_size / 4))]
        return get_loader(image_size, channels_img, batch_sizes, number_of_datasamples, num_workers, elevations_dataset_path)
    return loader_factory
