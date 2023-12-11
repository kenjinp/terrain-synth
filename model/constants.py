import os
from dotenv import load_dotenv

load_dotenv()

IMAGE_SIZE = os.getenv('IMAGE_SIZE')
NUMBER_OF_DATASAMPLES = os.getenv('NUMBER_OF_DATASAMPLES')
ELEVATIONS_DATASET_PATH = os.getenv('ELEVATIONS_DATASET_PATH')
BATCH_SIZE = os.getenv('BATCH_SIZE')
LEARNING_RATE = os.getenv('LEARNING_RATE')
CHANNELS_IMG = os.getenv('CHANNELS_IMG')
Z_DIM = os.getenv('Z_DIM')
NUM_EPOCHS = os.getenv('NUM_EPOCHS')
FEATURES_CRITIC = os.getenv('FEATURES_CRITIC')
FEATURES_GEN = os.getenv('FEATURES_GEN')
CRITIC_ITERATIONS = os.getenv('CRITIC_ITERATIONS')
LAMBDA_GP = os.getenv('LAMBDA_GP')
SAVE_INTERVAL = os.getenv('SAVE_INTERVAL')


def config():
    return {
        'image_size': int(IMAGE_SIZE),
        'number_of_datasamples': int(NUMBER_OF_DATASAMPLES),
        'elevations_dataset_path': ELEVATIONS_DATASET_PATH,
        'batch_size': int(BATCH_SIZE),
        'learning_rate': float(LEARNING_RATE),
        'channels_img': int(CHANNELS_IMG),
        'z_dim': int(Z_DIM),
        'num_epochs': int(NUM_EPOCHS),
        'features_critic': int(FEATURES_CRITIC),
        'features_gen': int(FEATURES_GEN),
        'critic_iterations': int(CRITIC_ITERATIONS),
        'lambda_gp': float(LAMBDA_GP),
        'save_interval': int(SAVE_INTERVAL),
    }
