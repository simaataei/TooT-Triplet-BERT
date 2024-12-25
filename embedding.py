from transformers import BertTokenizer, BertModel
import gc
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
from torch.nn import functional as F
import time
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import BertTokenizer,BertForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertForSequenceClassification, AdamW, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
from datasets import load_dataset
import re
from torch.utils.data import Dataset
import csv
from sklearn.metrics import classification_report

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


train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f1.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f1.csv')
total_dataset =  ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3.csv')

#load model

model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=True)
print(model)
    # Other configuration parameters as needed
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])
#tokenizer=BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")

#model = SimpleNNWithBERT()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)

model_path = './models/euclidean/spec_offline_euclidean_e90'
model.load_state_dict(torch.load(model_path))

#testing with KNN
# get the train embedingsi
train_emb = []
test_emb = []
y_train =[]
y_test =[]
#for i in range(10):
 #   item = train_dataset[i]
for item in train_dataset+test_dataset:
    print(item)
    with torch.no_grad():
         tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
         input_ids = tokenized_seq['input_ids'].to(device)
         attention_mask = tokenized_seq['attention_mask'].to(device)
         print(tokenized_seq) 
         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
         embeddings = outputs.last_hidden_state[:, 0, :]
         print(embeddings.unsqueeze(0).cpu().numpy())
         print(embeddings.detach().cpu().numpy()[0])
         # Define anchor, positive, and negative samples from the embeddings
         train_emb.append(embeddings.detach().cpu().numpy()[0])
         y_train.append(item['labels'])

         del tokenized_seq, input_ids, attention_mask, outputs, embeddings

         # Manually invoke garbage collection in critical places
         if gc.isenabled():
            gc.collect()
         torch.cuda.empty_cache()
print(train_emb)
'''
for i in range(9):
	item = test_dataset[i]
#for item in test_dataset:
	with torch.no_grad():
		tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
		input_ids = tokenized_seq['input_ids'].to(device)
		attention_mask = tokenized_seq['attention_mask'].to(device)

		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		embeddings = outputs.last_hidden_state[:, 0, :]

	# Define anchor, positive, and negative samples from the embeddings
	test_emb.append(embeddings.detach().cpu().numpy()[0])
	y_test.append(item['labels'])

	del tokenized_seq, input_ids, attention_mask, outputs, embeddings

	# Manually invoke garbage collection in critical places
	if gc.isenabled():
	    gc.collect()
	torch.cuda.empty_cache()

print(test_emb)

'''

#with open('Dataset/spec_emb_train_f1_untrained.txt', 'w') as file:
 #   for item in train_emb:
  #      file.write(f"{item}\n")
#with open('Dataset/spec_emb_test_f1_untrained.txt', 'w') as file:
 #   for item in test_emb:
 #       file.write(f"{item}\n")

'''
# test val set with knn
knn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier
knn.fit(train_emb, y_train)
predictions = knn.predict(val_emb)
mcc = matthews_corrcoef(y_val, predictions)
print(mcc)
print(classification_report(y_val, predictions))'''
