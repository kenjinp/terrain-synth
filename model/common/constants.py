import os
from dotenv import load_dotenv

load_dotenv()

IMAGE_SIZE = os.getenv('IMAGE_SIZE')
NUMBER_OF_DATASAMPLES = os.getenv('NUMBER_OF_DATASAMPLES')
ELEVATIONS_DATASET_PATH = os.getenv('ELEVATIONS_DATASET_PATH')
BATCH_SIZES = os.getenv('BATCH_SIZES')
LEARNING_RATE = os.getenv('LEARNING_RATE')
IMAGE_SIZE = os.getenv('IMAGE_SIZE')
CHANNELS_IMG = os.getenv('CHANNELS_IMG')
IN_CHANNELS = os.getenv('IN_CHANNELS')
Z_DIM = os.getenv('Z_DIM')
NUM_EPOCHS = os.getenv('NUM_EPOCHS')
CRITIC_ITERATIONS = os.getenv('CRITIC_ITERATIONS')
LAMBDA_GP = os.getenv('LAMBDA_GP')
SAVE_MODEL = os.getenv('SAVE_INTERVAL')
NUM_WORKERS = os.getenv('NUM_WORKERS')
SAVE_MODEL = os.getenv('SAVE_MODEL')
BATCH_SIZE = os.getenv('BATCH_SIZE')
NUM_EPOCHS = os.getenv('NUM_EPOCHS')
PROGRESSIVE_EPOCHS_BY_IMAGE_SIZE = os.getenv(
    'PROGRESSIVE_EPOCHS_BY_IMAGE_SIZE')
START_TRAIN_AT_IMG_SIZE = os.getenv('START_TRAIN_AT_IMG_SIZE')


def config():
    return {
        'image_size': int(IMAGE_SIZE),
        'number_of_datasamples': int(NUMBER_OF_DATASAMPLES),
        'elevations_dataset_path': ELEVATIONS_DATASET_PATH,
        'batch_sizes': list(map(lambda x: int(x), BATCH_SIZES.split(","))),
        'learning_rate': float(LEARNING_RATE),
        'channels_img': int(CHANNELS_IMG),
        'z_dim': int(Z_DIM),
        'num_epochs': int(NUM_EPOCHS),
        'critic_iterations': int(CRITIC_ITERATIONS),
        'lambda_gp': float(LAMBDA_GP),
        'save_model': bool(SAVE_MODEL),
        'num_workers': int(NUM_WORKERS),
        'progressive_epochs_by_image_size': int(PROGRESSIVE_EPOCHS_BY_IMAGE_SIZE),
        'in_channels': int(IN_CHANNELS),
        'start_train_at_img_size': int(START_TRAIN_AT_IMG_SIZE),
        'critic_iterations': int(CRITIC_ITERATIONS),
        'batch_size': int(BATCH_SIZE),
        'num_epocks': int(NUM_EPOCHS),
    }
