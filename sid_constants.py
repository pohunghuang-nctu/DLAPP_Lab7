WAV_DIR = '/data_container/LibriSpeechTrain/train-clean-100'
DATASET_DIR = '/data_container/LibriSpeechTrain/train-clean-100-npy'

BATCH_SIZE = 32       
TRIPLET_PER_BATCH = 3

NUM_FRAMES = 160   
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
NUM_SPEAKERS = 251
EMBEDDING_SIZE = 512
