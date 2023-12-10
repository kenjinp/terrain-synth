import os
from dotenv import load_dotenv

load_dotenv()

DOWNSAMPLE_SIZE = os.getenv('DOWNSAMPLE_SIZE')
NUMBER_OF_DATASAMPLES = os.getenv('NUMBER_OF_DATASAMPLES')
ELEVATIONS_DATASET_PATH = os.getenv('ELEVATIONS_DATASET_PATH')
BATCH_SIZE = os.getenv('BATCH_SIZE')


def config():
    return {
        'downsample_size': int(DOWNSAMPLE_SIZE),
        'number_of_datasamples': int(NUMBER_OF_DATASAMPLES),
        'elevations_dataset_path': ELEVATIONS_DATASET_PATH,
        'batch_size': int(BATCH_SIZE),
    }
