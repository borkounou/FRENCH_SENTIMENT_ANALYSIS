# Import Librairies and packages

from numpy.core.numeric import False_
import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import random 
import time

from classifier import initialize_model
from tokenizer_local import tokenization
from config import MAX_LENGTH, BATCH_SIZE,TRAIN_DATA_PATH, VAL_DATA_PATH,device_GPU_CPU
from data_prepocess import data_preprocess

torch.cuda.empty_cache()

# Configuration of device
device = device_GPU_CPU()

loss_fn = nn.CrossEntropyLoss()

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def evaluate(model, val_dataloader,labels_test):
    model.eval()
    # Tracking the variables
    val_accuracy = []
    val_loss = []
    all_logits = []
    # For each batch in our validation set:
    for batch in val_dataloader:
        # Load bathc to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        
        all_logits.append(logits)
        # Compute loss 
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds =torch.argmax(logits, dim=1).flatten()

