import torch.nn as nn 
import torch 
from transformers import CamembertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from config import PRETRAINED_MODEL_PATH



class MainClassifier(nn.Module):
    def __init__(self):
        super(MainClassifier, self).__init__()
        # Pretrained Model
        self.camembert = CamembertModel.from_pretrained(PRETRAINED_MODEL_PATH)

        # Hidden layer
        self.fc1 = nn.Linear(self.camembert.config.hidden_size, 200)
        # Dropout Regularizer
        self.dropout = nn.Dropout(p=0.2)
        # Batch normalization
        self.batchnorm = nn.BatchNorm1d(self.camembert.config.hidden_size)
        # Output layers
        self.out = nn.Linear(200,2)


    def forward(self,input_ids, input_masks):
        camembert = self.camembert(input_ids, input_masks)
        camembert = camembert[0]
        x = camembert[:, 0, :]
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))
        output = self.out(x)
        return output
    

# Initialization of the model 
def initialize_model(dataloader_train, device="cpu", epochs=3):
    classifier = MainClassifier()
    classifier.to(device)
    optimizer = AdamW(classifier.parameters, lr=5e-5, eps=1e-8)
    total_steps  = len(dataloader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return classifier, optimizer, scheduler

