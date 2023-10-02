
import torch 

TRAIN_DATA_PATH = "./Raw_Data/train.csv"
VAL_DATA_PATH = "./Raw_Data/valid.csv"
TEST_DATA_PATH = "./Raw_Data/test.csv"
PRETRAINED_MODEL_PATH="camembert-base"
MAX_LENGTH = 130
BATCH_SIZE = 8

RESULT_PATH = "simple_test.txt"

FINAL_TRAIN_MODEL_PATH = "BorkounouBERT.h5"

def device_GPU_CPU():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Thre are {torch.cuda.device_count()} GPU(s) available." )
        print('Device name:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device
