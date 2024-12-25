from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, AutoConfig, BertModel, BertForSequenceClassification
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.nn import TripletMarginLoss
from peft import LoraConfig, TaskType
from peft import get_peft_model
from torch.optim import Adam
import random
from transformers import BertTokenizer, BertModel
import gc
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import optuna
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
#from Data_preparation_bin import X_train,X_test, y_test, X_val, y_train, y_val
#from model import SubstrateClassifier1
from sklearn.utils.class_weight import compute_class_weight
import json
from torch.nn import functional as F
import time
#import timeout_decorator
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import BertTokenizer,BertForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from torchmetrics.classification import BinaryMatthewsCorrCoef
import datasets
from datasets import load_dataset
import re
from torch.utils.data import Dataset
import csv
gpu_index = 0
total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
    # Memory currently being used
current_memory_allocated = torch.cuda.memory_allocated(gpu_index)
    # Memory cached by the allocator
current_memory_cached = torch.cuda.memory_reserved(gpu_index)
print(f"Total GPU Memory: {total_memory / 1e9} GB")
print(f"Current Memory Allocated: {current_memory_allocated / 1e9} GB")
print(f"Current Memory Cached: {current_memory_cached / 1e9} GB")


class SimpleTripletSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_samples = self._make_class_samples()

    def _make_class_samples(self):
        # Organize samples by class for easy retrieval
        class_samples = {}
        for idx, sample in enumerate(self.dataset):
            label = sample['labels'].item()
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(idx)
        return class_samples

    def __iter__(self):
        for _ in range(len(self)):
            indices = []
            classes = list(self.class_samples.keys())
            for _ in range(self.batch_size // 3):  # Assuming batch_size is divisible by 3
                anchor_class = random.choice(classes)
                positive_class = anchor_class
                negative_class = random.choice([cls for cls in classes if cls != anchor_class])

                anchor_idx = random.choice(self.class_samples[anchor_class])
                positive_idx = random.choice([idx for idx in self.class_samples[positive_class] if idx != anchor_idx])  # Ensure positive is not the same as anchor
                negative_idx = random.choice(self.class_samples[negative_class])

                indices.extend([anchor_idx, positive_idx, negative_idx])
            yield indices

    def __len__(self):
        # Approximate length given dataset size and batch size
        return len(self.dataset) // self.batch_size


class ProteinSeqDataset(Dataset):
    def __init__(self, filepath, max_length=1024):
        self.dataframe = pd.read_csv(filepath)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        #idx = int(idx)
        assert isinstance(idx, int), f"Index must be an integer, got {type(idx)}"
        text = self.dataframe.iloc[idx, 0]  # Assuming text is the first column
        label = self.dataframe.iloc[idx, 1]  # Assuming label is the second column
        return {
            'seq':text,  # Remove batch dimension
            'labels': label
        }
device = torch.device("cuda")
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
best_mcc =-1

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5,40)  # Tuning the number of epochs
    train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f1.csv')
    val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
    test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f1.csv')

    batch_size = 3  # This should be a multiple of 3
    triplet_sampler = SimpleTripletSampler(train_dataset, batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=triplet_sampler)
    #train_loader = DataLoader(train_dataset, batch_size=3, sampler=RandomSampler(train_dataset))

    triplet_loss = TripletMarginLoss(margin=1.0, p=2)

    model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=True)
    # Other configuration parameters as needed
    lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-5)
   # Training Loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            print('batch')
            print(batch)
            tokenized_batch = tokenizer(batch['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
            print('tokens')
            print(tokenized_batch)

            input_ids = tokenized_batch['input_ids'].to(device)
            attention_mask = tokenized_batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]

            # Define anchor, positive, and negative samples from the embeddings
            anchor = embeddings[0].unsqueeze(0)  # First sequence as anchor
            positive = embeddings[1].unsqueeze(0)  # Second sequence as positive
            negative = embeddings[2].unsqueeze(0)  # Third sequence as negative

            # Initialize the TripletMarginLoss
            triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

            # Calculate the loss
            loss = triplet_loss_fn(anchor, positive, negative)
            print('loss: '+str(loss))
            loss.backward()
    #
            # Update the model's weights
            optimizer.step()
            print('end of batch')
            del tokenized_batch, input_ids, attention_mask, outputs, embeddings, anchor, positive, negative, loss

            # Manually invoke garbage collection in critical places
            if gc.isenabled():
                gc.collect()

            torch.cuda.empty_cache()
            current_memory_cached = torch.cuda.memory_reserved(gpu_index)
            print(f"Total GPU Memory: {total_memory / 1e9} GB")
            print(f"Current Memory Allocated: {current_memory_allocated / 1e9} GB")
            print(f"Current Memory Cached: {current_memory_cached / 1e9} GB")


    #testing with KNN
    # get the train embedingsi
    train_emb = []
    val_emb = []
    y_train =[]
    y_val =[]
    for item in train_dataset:
    #for i in range(5):
     #   item = train_dataset[i]
        with torch.no_grad():       
             tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
		    
             input_ids = tokenized_seq['input_ids'].to(device)
             attention_mask = tokenized_seq['attention_mask'].to(device)

             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
             embeddings = outputs.last_hidden_state[:, 0, :]

		# Define anchor, positive, and negative samples from the embeddings
             train_emb.append(embeddings.detach().cpu().numpy()[0])
             y_train.append(item['labels'])
    for item in test_dataset:
    #for i in range(5):
     #   item = val_dataset[i]
        with torch.no_grad():
             tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')

             input_ids = tokenized_seq['input_ids'].to(device)
             attention_mask = tokenized_seq['attention_mask'].to(device)

             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
             embeddings = outputs.last_hidden_state[:, 0, :]

             # Define anchor, positive, and negative samples from the embeddings
             val_emb.append(embeddings.detach().cpu().numpy()[0])
             y_val.append(item['labels'])
	     
    # test val set with knn
    knn = KNeighborsClassifier(n_neighbors=1)
    print(train_emb)
    print(len(train_emb[1]))
    # Train the classifier
    knn.fit(train_emb, y_train)
    predictions = list(knn.predict(val_emb))
    print(y_train)
    print(predictions)
    mcc = matthews_corrcoef(y_val, predictions)
    global best_mcc
    if mcc > best_mcc:
            best_mcc = mcc
            # Save the best model state dictionary
            torch.save(model.state_dict(), 'best_model_offline_triplet_lora_optuna')
            config = model.config

            # Save the config in a file
            config_dict = config.to_dict()
            with open("best_model_triplet_lora_optuna.json", "w") as config_file:
                json.dump(config_dict, config_file, indent=2)
    current_memory_cached = torch.cuda.memory_reserved(gpu_index)
    print(f"Test Memory Cached: {current_memory_cached / 1e9} GB")
    del model
    return mcc
def optimize_hyperparameters():
    best_mcc= -1
    study = optuna.study.create_study(direction='maximize')
    study.optimize(objective,n_trials=10,  gc_after_trial=True)
    best_params = study.best_params
    best_mcc = study.best_value

    return best_params, best_mcc

best_params, best_mcc = optimize_hyperparameters()
print("Best Parameters:", best_params)
print("Best MCC:", best_mcc)
