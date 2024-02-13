from numpy.core.numeric import False_
import pandas as pd
from config import device_GPU_CPU, PRETRAINED_MODEL_PATH
import torch 
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import torch.nn.functional as F
import numpy as np

from classifier import MainClassifier
from tokenizer_local import tokenization,single_tokenizer
from config import MAX_LENGTH, BATCH_SIZE, FINAL_TRAIN_MODEL_PATH, RESULT_PATH,TEST_DATA_PATH


device = device_GPU_CPU()
print(device_GPU_CPU)

def predict_test(model, test_dataloader):
    model.eval()
    all_logits = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask= tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs


def test_data_preprocess(path:str):
     raw_data = pd.read_csv(path,encoding="utf-8")
     data = raw_data[["review","polarity"]]
     data = data.dropna()
     data = data[:2]
     X_test = data['review']
     X_test = np.array(X_test)
     input_ids_test, attention_masks_test = tokenization(X_test, MAX_LENGTH)
     dataset_test= TensorDataset(input_ids_test, attention_masks_test)
     dataloader_test =  DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=BATCH_SIZE)
     return data, dataloader_test


def single_process(element):
 
     input_ids_test, attention_masks_test = single_tokenizer(element, MAX_LENGTH)
     dataset_test= TensorDataset(input_ids_test, attention_masks_test)
     dataloader_test =  DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=BATCH_SIZE)
     return element, dataloader_test



def test_executable(model, path_test):
    dataframe, dataloader_test = test_data_preprocess(path_test)
    probs = predict_test(model, dataloader_test)
    preds = np.argmax(probs, axis=1)
    dataframe['predict'] = preds
    np.savetxt(RESULT_PATH, dataframe[["polarity", "predict"]], delimiter=" ", fmt='%s')
    print(f"The file of predicted result is saved in this path: {RESULT_PATH}. Check it out!")



def predictor():
    model = MainClassifier()
    model.to(device)
    model.load_state_dict(torch.load(FINAL_TRAIN_MODEL_PATH))
    test_executable(model, TEST_DATA_PATH)

def sing_predictor(element):
    model = MainClassifier()
    model.to(device)
    state_dict = torch.load(FINAL_TRAIN_MODEL_PATH)
  
    model.load_state_dict(state_dict)
    element, dataloader_test = single_process(element)
    probs = predict_test(model, dataloader_test)
    preds = np.argmax(probs, axis=1)
    return element, preds
    
