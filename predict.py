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
from joblib import dump
from sklearn.preprocessing import StandardScaler



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


train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f2.csv')
# Create a new dataset instance with the subset
#train_dataset = ProteinSeqDataset('./Dataset/small_trans.csv')

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

model_path = './triplet_model_97e_f2'
model.load_state_dict(torch.load(model_path))
gpu_index = 0
total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
    # Memory currently being used
current_memory_allocated = torch.cuda.memory_allocated(gpu_index)
    # Memory cached by the allocator
current_memory_cached = torch.cuda.memory_reserved(gpu_index)
print(f"Total GPU Memory: {total_memory / 1e9} GB")
print(f"Current Memory Allocated: {current_memory_allocated / 1e9} GB")
print(f"Current Memory Cached: {current_memory_cached / 1e9} GB")


fasta_file = 'Dataset/external_mutual.fasta'
file_name = 'external'
# List to store sequences
sequences = []
seq_name = []
# Read the FASTA file
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append(str((' ').join(record.seq)))
    seq_name.append(str(record.id))
#testing with KNN
# get the train embedingsi
train_emb = []
val_emb = []
y_train =[]
y_val =[]
for item in train_dataset+val_dataset:
#for i in range(10):
 #   item = train_dataset[i]  
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
for item in sequences:
#for i in range(10):
#	item = val_dataset[i]
	with torch.no_grad():
		tokenized_seq = tokenizer(item,max_length=1024, padding=True, truncation=True, return_tensors='pt')
		input_ids = tokenized_seq['input_ids'].to(device)
		attention_mask = tokenized_seq['attention_mask'].to(device)

		outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		embeddings = outputs.last_hidden_state[:, 0, :]

	# Define anchor, positive, and negative samples from the embeddings
	val_emb.append(embeddings.detach().cpu().numpy()[0])

knn = KNeighborsClassifier(n_neighbors=1)

scaler = StandardScaler()
train_emb = scaler.fit_transform(train_emb)
val_emb = scaler.transform(val_emb)


# Train the classifier
knn.fit(train_emb, y_train)
model_filename = '1nn_model_offline_e97.joblib'
dump(knn, model_filename)

predictions = knn.predict(val_emb)

label_name = {}
label_chebi ={}
with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
    data = f.readlines()
    for d in data:
        d = d.split(',')
        label_name[int(d[2].strip('\n'))] = d[1]
        label_chebi[int(d[2].strip('\n'))] = d[0]



with open(f'Results/{file_name}_resutls_offline_e97.txt', 'w') as file:
    file.write("Seq Name, Predicted Label , Predicted Label ChEBI ID, Predicted Label Name\n")
    # Loop through all predictions and corresponding sequence names
    for name, label in zip(seq_name, predictions):
        # Fetch the label name and ChEBI ID from the dictionaries
        label_name_list = label_name.get(label, "Unknown Label Name")
        chebi_id_list = label_chebi.get(label, "Unknown ChEBI ID")
        
        # Write to file
        file.write(f"{name}, {label}, {chebi_id_list}, {label_name_list}\n")

df = pd.DataFrame({
    'Seq Name': seq_name,
    'Predicted Label': predictions,
    'Predicted Label ChEBI ID': chebi_id_list,
    'Predicted Label Name': label_name_list
}) 
# Convert the DataFrame to LaTeX code
latex_code = df.to_latex(index=False, caption='Predicted labels and their corresponding ChEBI IDs and names', label='tab:prediction_table', longtable=True)

with open(f'Results/{file_name}_resutls_offline_e97_latex.tex', 'w') as file:
    file.write(latex_code)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_emb, y_train)
model_filename = '10nn_model_offline_e97.joblib'
dump(knn, model_filename)

# Retrieve the 10 closest neighbors and their distances for the validation set
distances, indices = knn.kneighbors(val_emb)
print(indices)
nei_name= [
    [label_name[y_train[number]] for number in row]
    for row in indices
]
print("Distances:\n", distances)

print("Names of Neighbors:\n",nei_name )
name_distance = [
    [(nei_name[i][j], distances[i][j]) for j in range(len(nei_name[i]))]
    for i in range(len(distances))
]
print("Name-Distances:\n", name_distance)
df = pd.DataFrame({'neighbours':name_distance})
with open(f'./Results/{file_name}_offline_sampling_up_100_e97_10nn.txt', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(df.to_latex(index=True))



all_seq_names = []
all_nei_names = []
all_distances = []
for index, names in enumerate(nei_name):
    for j in range(len(names)):
        all_seq_names.append(seq_name[index])
        all_nei_names.append(nei_name[index][j])
        all_distances.append(distances[index][j])

# Create DataFrame
df = pd.DataFrame({
    'Seq Name': all_seq_names,
    'Neighbor Name': all_nei_names,
    'Distance': all_distances
})

print(df)
latex_code = df.to_latex(index=False, caption='Neighbor Details', label='tab:neighbor_details', longtable=True)
with open(f'Results/10neighbor_details_{file_name}_offline_e97_test_train_wo_proton.tex', 'w') as file:
    file.write(latex_code)










knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_emb, y_train)
model_filename = '3nn_model_offline_e97.joblib'
dump(knn, model_filename)

# Retrieve the 10 closest neighbors and their distances for the validation set
distances, indices = knn.kneighbors(val_emb)
print(indices)
nei_name= [
    [label_name[y_train[number]] for number in row]
    for row in indices
]
print("Distances:\n", distances)

print("Names of Neighbors:\n",nei_name )
name_distance = [
    [(nei_name[i][j], distances[i][j]) for j in range(len(nei_name[i]))]
    for i in range(len(distances))
]
print("Name-Distances:\n", name_distance)
df = pd.DataFrame({'neighbours':name_distance})
with open(f'./Results/{file_name}_offline_sampling_up_100_e97_3nn_test_train_wo_proton.txt', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(df.to_latex(index=True))



all_seq_names = []
all_nei_names = []
all_distances = []
for index, names in enumerate(nei_name):
    for j in range(len(names)):
        all_seq_names.append(seq_name[index])
        all_nei_names.append(nei_name[index][j])
        all_distances.append(distances[index][j])

# Create DataFrame
df = pd.DataFrame({
    'Seq Name': all_seq_names,
    'Neighbor Name': all_nei_names,
    'Distance': all_distances
})

print(df)
latex_code = df.to_latex(index=False, caption='Neighbor Details', label='tab:neighbor_details', longtable=True)
with open(f'Results/3neighbor_details_{file_name}_offline_e97_test_train_wo_proton.tex', 'w') as file:
    file.write(latex_code)
