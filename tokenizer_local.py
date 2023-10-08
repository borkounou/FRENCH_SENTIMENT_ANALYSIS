from transformers import CamembertTokenizer
import torch 
from config import PRETRAINED_MODEL_PATH


# The below function tokenizes the data and it has two argupments

def tokenization(data, max_length):

    print("Starting tokenization...")

    tokenizer = CamembertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    input_ids = []
    attention_mask = []
    # Encoding every sentence
    for element in data:
        encoded_element = tokenizer.encode_plus(str(element), add_special_tokens=True, 
                                                truncation=True,max_length=max_length,
                                                padding='max_length', return_tensors='pt')
        input_ids.append(encoded_element["input_ids"])
        attention_mask.append(encoded_element["attention_mask"])
    print("Tokenization finished!")


    input_ids = torch.cat(input_ids,dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    return input_ids, attention_mask

def single_tokenizer(element, max_length):
    tokenizer = CamembertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
 
    encoded_element = tokenizer.encode_plus(str(element), add_special_tokens=True, 
                                                truncation=True,max_length=max_length,
                                                padding='max_length', return_tensors='pt')
    print("Tokenization finished!")
    # input_ids.append(encoded_element["input_ids"])
    # attention_mask.append(encoded_element["attention_mask"])
    return encoded_element["input_ids"], encoded_element["attention_mask"]
   

