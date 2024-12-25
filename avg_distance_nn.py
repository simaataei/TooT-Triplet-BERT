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
from Bio import SeqIO
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict

def cosine_distance(a, b):
    # Cosine similarity is between 0 and 1, but we need a distance metric
    # So, we return 1 - cosine similarity
    return 1 - cosine_similarity(a, b)

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

def load_label_names(filepath):
    label_dict = {}
    with open(filepath) as f:
        data = f.readlines()
        for d in data:
            parts = d.split(',')
            label_dict[int(parts[2].strip())] = parts[1].strip()
    return label_dict


def avg_nn_distance(model_name, label_filepath):
    device = torch.device("cuda")
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
    best_mcc =-1

    train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
    #val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
    test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f2.csv')
    file_name  ='spec'

    #load model

    model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=True)
    print(model)
    # Other configuration parameters as needed
    lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)
    ##model_name = 'triplet_model_97e_f2_cosine'
   #mdel_path = './triplet_model_97e_f2_cosine'
    #model_name = model_path.strip('./')
    dis_metric = model_name.split('_')[2]
    print(dis_metric)
    model_path = f'./models/{dis_metric}/{model_name}'

    model.load_state_dict(torch.load(model_path))

    #testing with KNN
    # get the train embedingsi
    train_emb = []
    val_emb = []
    y_train =[]
    y_val =[]
    for item in train_dataset:
 #   for i in range(5):
  #      item = train_dataset[i]  
        with torch.no_grad():
             tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
             input_ids = tokenized_seq['input_ids'].to(device)
             attention_mask = tokenized_seq['attention_mask'].to(device)
             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
             embeddings = outputs.last_hidden_state[:, 0, :]
             train_emb.append(embeddings.detach().cpu().numpy()[0])
             y_train.append(item['labels'])

             del tokenized_seq, input_ids, attention_mask, outputs, embeddings

             # Manually invoke garbage collection in critical places
             if gc.isenabled():
                gc.collect()
             torch.cuda.empty_cache()


    label_dict = load_label_names(label_filepath)
    # test val set with knn
    knn = KNeighborsClassifier(metric=dis_metric,algorithm='brute',n_neighbors=1)

    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    # Train the classifier
    knn.fit(train_emb, y_train)
    distances, indices = knn.kneighbors(train_emb)
    print(distances)
    print(indices)
    class_distances = defaultdict(list)
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        class_label = y_train[i]
        class_distances[class_label].append(dist[0])
        print(dist)
    # Step 4: Calculate the average and standard deviation of the distances for each class
    class_stats = []
    for class_label, dists in class_distances.items():
        avg_distance = np.mean(dists)
        std_distance = np.std(dists)
        class_name = label_dict[class_label]
        class_stats.append({'Class Name': class_name, 'Average Distance': avg_distance, 'Standard Deviation': std_distance})
    df = pd.DataFrame(class_stats)
    # Save the DataFrame as a long table LaTeX file
    with open(f'Results/class_avg_distances_{model_name}.tex', 'w') as f:
         f.write(df.to_latex(index=False, longtable=True))   

    return class_stats

model_name = 'spec_offline_manhattan_e90'
label_filepath = './Dataset/Label_name_list_transporter_uni_ident100_t3'
class_stats = avg_nn_distance(model_name, label_filepath)
