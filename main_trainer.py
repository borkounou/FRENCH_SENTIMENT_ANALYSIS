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
from config import MAX_LENGTH, BATCH_SIZE,TRAIN_DATA_PATH, VAL_DATA_PATH,device_GPU_CPU,FINAL_TRAIN_MODEL_PATH
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

        # Calculate the accuracy rate
        accuracy = (preds==b_labels).cpu().numpy().mean() *100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    all_logits= torch.cat(all_logits, dim=0)
    
    # probabilities 
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    labels_test = labels_test.ravel()
    
    # AUC of multiclass

    # roc_auc = roc_auc_score(labels_test,probs,multi_class='ovr')
    return val_loss, val_accuracy#,roc_auc

def train(model, train_dataloader, optimizer, scheduler, labels_test, val_dataloader=None, epochs=3, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'AUC':^9} | {'Elapsed':^9}")
        print("-"*70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0.0, 0, 0

        # Put the model into the training mode
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            b_labels = b_labels.squeeze()
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
           
            logits = model(b_input_ids, b_attn_mask)
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits.squeeze(), b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 1000 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 10000 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
            # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)

        # =======================================
        #               Evaluation Section
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy= evaluate(model, val_dataloader, labels_test)
            min_loss = np.inf
            if val_loss < min_loss:
                min_loss = val_loss
                # Saving the model
                torch.save(model.state_dict(),FINAL_TRAIN_MODEL_PATH)
                print("model saved")
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch 
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    #torch.save(model.state_dict(),path_modele)
    print("Training complete!")

# Performance metrics

def all_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, preds_flat), confusion_matrix(labels_flat, preds_flat)

# Prediction de validation

def predict(model, test_dataloader):
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs
# Performance metrics

def all_metrics(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, preds_flat), confusion_matrix(labels_flat, preds_flat)

# Prediction de validation

def predict(model, test_dataloader):
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs



# =======================================
#           Section d'execution
# =======================================

def execute():
    
    X_train, y_train= data_preprocess(TRAIN_DATA_PATH)
    X_test, y_test = data_preprocess(VAL_DATA_PATH)
    # Tokenizing training data 
    input_ids_train, attention_masks_train = tokenization(X_train, MAX_LENGTH)
    labels_train = torch.tensor(y_train)
    # Tokenizing testing data 
    input_ids_test, attention_masks_test = tokenization(X_test, MAX_LENGTH)
    labels_test = torch.tensor(y_test)
    # Train dataset
    dataset_train = TensorDataset(input_ids_train,attention_masks_train,labels_train)
    # Test dataset
    dataset_test= TensorDataset(input_ids_test, attention_masks_test,labels_test)
    dataloader_train =  DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=BATCH_SIZE)
    dataloader_val =  DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=BATCH_SIZE)
    set_seed(42)    # Set seed for reproducibility
    classifier, optimizer, scheduler = initialize_model(dataloader_train, device, epochs=3)
    train(classifier, dataloader_train, optimizer, scheduler, labels_test, dataloader_val, epochs=3, evaluation=True)
    probs = predict(classifier, dataloader_val)
    classification, matriceP = all_metrics(probs, labels_test)
    print("______________classification report______________________")
    print(classification)
    print("______________ Confusion matrix_______________________")
    print(matriceP)



