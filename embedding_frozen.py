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
from model import SubstrateClassifier1

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
#tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
#best_mcc =-1


train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f1.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f1.csv')
total_dataset =  ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3.csv')
model_name = 'Rostlab/prot_bert_bfd'
tokenizer = BertTokenizer.from_pretrained(model_name)
protbert_model = BertModel.from_pretrained(model_name)

#model=SubstrateClassifier1(96).cuda()
#model.load_state_dict(torch.load("./finetuned-spec"))

def get_protein_embeddings(protein_sequences):
    inputs = tokenizer(protein_sequences, padding=True, truncation=True, max_length=1024, return_tensors='pt')
    with torch.no_grad():
        outputs = protbert_model(**inputs)
    protein_embeddings = outputs.last_hidden_state[:, 0, :].unsqueeze(0).detach().cpu().numpy()[0][0]
    #print(len(protein_embeddings))
    #print(protein_embeddings)
    return protein_embeddings

#train_sequences = [train_dataset[i]['seq'] for i in range(10)]
#train_sequences = [train_dataset[i]['seq'] for i in range(10)]
#train_labels = [int(train_dataset[i]['labels']) for i in range(10)]
train_sequences = [train_dataset[i]['seq'] for i in range(len(train_dataset))]
train_labels = [int(train_dataset[i]['labels']) for i in range(len(train_dataset))]
# Get embeddings
train_emb = []
#for item in train_sequences:
 #   print(item)
  #  print(len(item))
  #  train_emb.append(get_protein_embeddings(item))
    #pred, emb = model(item)
    #train_emb.append(emb.cpu().detach().numpy())
#test_sequences = [test_dataset[i]['seq'] for i in range(10)]
#test_labels = [int(test_dataset[i]['labels']) for i in range(10)]

test_sequences = [test_dataset[i]['seq'] for i in range(len(test_dataset))]
test_labels = [int(test_dataset[i]['labels']) for i in range(len(test_dataset))]

test_emb = []
for item in train_sequences+test_sequences:
    test_emb.append(get_protein_embeddings(item))
#    pred, emb = model(item)
 #   test_emb.append(emb.cpu().detach().numpy())

frozen_emb = test_emb
y_frozen = train_labels+test_labels

#protein_data = torch.cat(protein_data, dim=0)
#protein_data = get_protein_embeddings(sequences)
