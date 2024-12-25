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
from scipy.spatial.distance import cityblock
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def manhattan_distance(x1, x2):
    return torch.cdist(x1, x2, p=1)

def euclidean_distance(x1, x2, eps=1e-6):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + eps)

def cosine_distance(x1, x2, p=2, eps=1e-6):
    return 1 - F.cosine_similarity(x1, x2, eps=eps)
def mse_distance(a, b):
    return np.mean((a - b) ** 2)

# Function to find the closest vector using MSE
def find_closest_vector(target_vector, vector_set):
    min_distance = float('inf')
    closest_vector = None
    
    for vec in vector_set:
#        distance = cityblock(target_vector, vec)
        distance = np.linalg.norm(target_vector-vec) 
        if distance < min_distance:
            min_distance = distance
            closest_vector = vec
    
    return closest_vector, min_distance

def load_label_names(filepath):
    label_dict = {}
    with open(filepath) as f:
        data = f.readlines()
        for d in data:
            parts = d.split(',')
            label_dict[int(parts[2].strip())] = parts[1].strip()
    return label_dict

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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def combine_histograms_log(set1_same_label, set1_diff_label, set2_same_label, set2_diff_label):
    # Combine all data to calculate the common axis limits
    all_data = np.concatenate([set1_same_label, set1_diff_label, set2_same_label, set2_diff_label])
    x_max = max(int(np.max(set1_diff_label)), int(np.max(set2_diff_label)))
    x_min = 0

    # Calculate bins based on the combined data
    _, bins = np.histogram(all_data, bins=50)

    # Plotting the histogram for distances with the same label
    plt.figure(figsize=(20, 10))
    for data, color, label in zip([set1_same_label, set2_same_label],
                                  ['lightcoral', 'lightblue'],
                                  ['Baseline 1', 'TooT-Triplet-offline']):
        counts, _ = np.histogram(data, bins=bins)
        log_counts = np.log1p(counts)  # Apply log transformation
        plt.bar(bins[:-1], log_counts, width=np.diff(bins), align='edge', alpha=0.5,
                color=color, edgecolor='black', label=label)
    lightblue_patch = mpatches.Patch(color='lightcoral', label='Baseline 1')
    darkblue_patch = mpatches.Patch(color='lightblue', label='TooT-Triplet-offline')
    plt.legend(handles=[lightblue_patch, darkblue_patch])
    plt.xlabel('Distance')
    plt.ylabel('Log(Frequency)')
    plt.xlim(x_min, x_max)
    plt.savefig('./Results/same_distances_histograms_overlay_frozen_offline_log.svg', format='svg')

    # Plotting the histogram for distances with different labels
    plt.figure(figsize=(20, 10))
    for data, color, label in zip([set1_diff_label, set2_diff_label],
                                  ['lightcoral', 'lightblue'],
                                  ['Baseline 1', 'TooT-Triplet-offline']):
        counts, _ = np.histogram(data, bins=bins)
        log_counts = np.log1p(counts)  # Apply log transformation
        plt.bar(bins[:-1], log_counts, width=np.diff(bins), align='edge', alpha=0.5,
                color=color, edgecolor='black', label=label)
    lightcoral_patch = mpatches.Patch(color='lightcoral', label='Baseline 1')
    darkred_patch = mpatches.Patch(color='lightblue', label='TooT-Triplet-offline')
    plt.legend(handles=[lightcoral_patch, darkred_patch])
    plt.xlabel('Distance')
    plt.ylabel('Log(Frequency)')
    plt.xlim(x_min, x_max)
    plt.savefig('./Results/diff_distances_histograms_overlay_frozen_offline_log.svg', format='svg')

def combine_histograms(set1_same_label, set1_diff_label, set2_same_label, set2_diff_label):
    # Combine all data to calculate the common axis limits
    all_data = np.concatenate([set1_same_label, set1_diff_label, set2_same_label, set2_diff_label])
    x_max = max(int(np.max(set1_diff_label)), int(np.max(set2_diff_label)))
    x_min = 0

    # To set consistent y-scale, we need to plot histograms and find the max height
    _, bins = np.histogram(all_data, bins=50)
    y_max_same = max(
        np.histogram(set1_same_label, bins=bins)[0].max(),
        np.histogram(set2_same_label, bins=bins)[0].max()
    )
    y_max_diff = max(
        np.histogram(set1_diff_label, bins=bins)[0].max(),
        np.histogram(set2_diff_label, bins=bins)[0].max()
    )
    y_max = max(y_max_same, y_max_diff)+50

    # Combined histogram for distances with the same label
    plt.figure(figsize=(20, 10))
    plt.hist(set1_same_label, bins=bins, alpha=0.5, color='lightcoral', edgecolor='black', label='Baseline 1')
    plt.hist(set2_same_label, bins=bins, alpha=0.5, color='lightblue', edgecolor='black', label='TooT-Triplet-offline')
    lightblue_patch = mpatches.Patch(color='lightcoral', label='Basline 1')
    darkblue_patch = mpatches.Patch(color='lightblue', label='TooT-Triplet-offline')
    plt.legend(handles=[lightblue_patch, darkblue_patch],fontsize=34)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)
    plt.xlabel('Distance', fontsize=38)
    plt.ylabel('Frequency',fontsize=38)
  #  plt.title('Combined Histogram: Distances with the Same Label')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
    plt.savefig(f'./Results/same_distances_histograms_overlay_frozen_offline.svg', format='svg')
    # Combined histogram for distances with different labels
    plt.figure(figsize=(20, 10))
    plt.hist(set1_diff_label, bins=bins, alpha=0.5, color='lightcoral', edgecolor='black', label='Baseline 1')
    plt.hist(set2_diff_label, bins=bins, alpha=0.5, color='lightblue', edgecolor='black', label='TooT-Triplet-offline')
    lightcoral_patch = mpatches.Patch(color='lightcoral', label='Baseline 1')
    darkred_patch = mpatches.Patch(color='lightblue', label='TooT-Triplet-offline')
    plt.legend(handles=[lightcoral_patch, darkred_patch], fontsize=34)
    plt.xlabel('Distance',fontsize=38)
    plt.ylabel('Frequency', fontsize=38)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)
    #plt.title('Combined Histogram: Distances with Different Labels')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
   # plt.show()lt.savefig(f'./Results/diff_distances_histograms_overlay_frozen_.svg', format='svg')
    plt.savefig(f'./Results/diff_distances_histograms_overlay_frozen_offline.svg', format='svg')

def get_encodings(train_dataset, model, tokenizer):
    train_emb = []
    val_emb = []
    y_train =[]
    y_val =[]
    model.to(device)
    for item in train_dataset:
    #for i in range(400):
     #   item = train_dataset[i]  
        with torch.no_grad():
             tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt').to(device)
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
    label_dict = load_label_names('./Dataset/Label_name_list_transporter_uni_ident100_t3')
    class_distances_same_label = {}
    class_distances_diff_label = {}

    class_sizes = {}  # Dictionary to store class sizes

    for index, label in enumerate(y_train):
        if label not in class_sizes:
           class_sizes[label] = 0
        class_sizes[label] += 1

    for index, (item, label) in enumerate(zip(train_emb, y_train)):
        # Create a temporary list of vectors with the same label, excluding the current item
        same_label_train_emb = [x for i, x in enumerate(train_emb) if y_train[i] == label and i != index]
        diff_label_train_emb = [x for i, x in enumerate(train_emb) if y_train[i] != label]

        # Calculate distance to the nearest vector with the same label
        closest_vector_same, min_distance_same = find_closest_vector(item, same_label_train_emb)
        if label not in class_distances_same_label:
           class_distances_same_label[label] = []
        class_distances_same_label[label].append(min_distance_same)
 
        # Calculate distance to the nearest vector with a different label
        closest_vector_diff, min_distance_diff = find_closest_vector(item, diff_label_train_emb)
        if label not in class_distances_diff_label:
           class_distances_diff_label[label] = []
        class_distances_diff_label[label].append(min_distance_diff)
    all_distances_same_label = [distance for distances in class_distances_same_label.values() for distance in distances]
    all_distances_diff_label = [distance for distances in class_distances_diff_label.values() for distance in distances]

    return all_distances_same_label, all_distances_diff_label


device = torch.device("cuda")
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
best_mcc =-1


train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f2.csv')
file_name  ='spec'
model1 = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=True)
model1.to(device)
model1_name = 'frozen'
model2 = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=True)    
# Other configuration parameters as needed
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])

model2 = get_peft_model(model2, lora_config)
model2.print_trainable_parameters()
model2.to(device)
model2_name = 'offline'
dis_metric = 'manhattan'
model_path = './models/manhattan/spec_offline_manhattan_e90'
#model_name = model_path.strip('./')
    #dis_metric = model_name.split('_')[2]
    #print(dis_metric)
    #model_path = f'./models/{dis_metric}/{model_name}'

model2.load_state_dict(torch.load(model_path))
'''
train_emb = []
val_emb = []
y_train =[]
y_val =[]
   
for item in train_dataset:
#for i in range(400):
#    item = train_dataset[i]  
    with torch.no_grad():
         tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
         input_ids = tokenized_seq['input_ids']
         attention_mask = tokenized_seq['attention_mask']
         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
         embeddings = outputs.last_hidden_state[:, 0, :]
         train_emb.append(embeddings.detach().cpu().numpy()[0])
         y_train.append(item['labels'])

         del tokenized_seq, input_ids, attention_mask, outputs, embeddings

         # Manually invoke garbage collection in critical places
         if gc.isenabled():
            gc.collect()
         torch.cuda.empty_cache()

label_dict = load_label_names('./Dataset/Label_name_list_transporter_uni_ident100_t3')
class_distances_same_label = {}
class_distances_diff_label = {}

class_sizes = {}  # Dictionary to store class sizes

for index, label in enumerate(y_train):
    if label not in class_sizes:
        class_sizes[label] = 0
    class_sizes[label] += 1

for index, (item, label) in enumerate(zip(train_emb, y_train)):
    # Create a temporary list of vectors with the same label, excluding the current item
    same_label_train_emb = [x for i, x in enumerate(train_emb) if y_train[i] == label and i != index]
    diff_label_train_emb = [x for i, x in enumerate(train_emb) if y_train[i] != label]

    # Calculate distance to the nearest vector with the same label
    closest_vector_same, min_distance_same = find_closest_vector(item, same_label_train_emb)
    if label not in class_distances_same_label:
        class_distances_same_label[label] = []
    class_distances_same_label[label].append(min_distance_same)

    # Calculate distance to the nearest vector with a different label
    closest_vector_diff, min_distance_diff = find_closest_vector(item, diff_label_train_emb)
    if label not in class_distances_diff_label:
        class_distances_diff_label[label] = []
    class_distances_diff_label[label].append(min_distance_diff)


all_distances_same_label = [distance for distances in class_distances_same_label.values() for distance in distances]
all_distances_diff_label = [distance for distances in class_distances_diff_label.values() for distance in distances]
'''

set1_same_label, set1_diff_label = get_encodings(train_dataset, model1, tokenizer)
set2_same_label, set2_diff_label = get_encodings(train_dataset, model2, tokenizer)
print(set1_same_label)
print(set1_diff_label)

print(max(set1_same_label))
print(max(set1_diff_label))
combine_histograms(set1_same_label, set1_diff_label, set2_same_label, set2_diff_label)
num_classes = 96



sorted_classes = sorted(class_sizes.keys(), key=lambda x: class_sizes[x], reverse=True)

# Split classes into three groups
top_5_classes = sorted_classes[:5]
classes_6_27 = sorted_classes[5:27]
rest_of_classes = sorted_classes[27:]

#histogram 

# Extract distances for each group
colors = ['#b0c4de','#7b8f9b', '#f2faff']

# Create the histogram for the entire dataset to get bin edges
counts, bin_edges = np.histogram(all_distances_same_label, bins=50)

# Initialize arrays to hold the stacked bar segments
top_counts = np.zeros(len(bin_edges)-1)
middle_counts = np.zeros(len(bin_edges)-1)
rest_counts = np.zeros(len(bin_edges)-1)

# Populate the counts for each group
for cls in top_5_classes:
    distances = class_distances_same_label[cls]
    top_counts += np.histogram(distances, bins=bin_edges)[0]

for cls in classes_6_27:
    distances = class_distances_same_label[cls]
    middle_counts += np.histogram(distances, bins=bin_edges)[0]

for cls in rest_of_classes:
    distances = class_distances_same_label[cls]
    rest_counts += np.histogram(distances, bins=bin_edges)[0]

# Plot the stacked histogram
plt.figure(figsize=(20,15 ))

plt.bar(bin_edges[:-1], top_counts, width=np.diff(bin_edges), color=colors[0], edgecolor='black', label='Majority Classes')
plt.bar(bin_edges[:-1], middle_counts, width=np.diff(bin_edges), bottom=top_counts, color=colors[1], edgecolor='black', label='Middle Classes')
plt.bar(bin_edges[:-1], rest_counts, width=np.diff(bin_edges), bottom=top_counts + middle_counts, color=colors[2], edgecolor='black', label='Minority Classes')

plt.xlabel('Distance', fontsize=28)
plt.ylabel('Frequency', fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
#plt.title('Histogram with Stacked Colors for Different Class Groups')
plt.legend(prop={'size': 22})



plt.savefig(f'Results/stacked_histogram_same_label_{model_name}.svg', format='svg')
# Prepare the data, positions, and colors for each group




colors = ['#f08080','#a85a5a', '#ffe1e1']
counts, bin_edges = np.histogram(all_distances_diff_label, bins=50)

# Initialize arrays to hold the stacked bar segments
top_counts = np.zeros(len(bin_edges)-1)
middle_counts = np.zeros(len(bin_edges)-1)
rest_counts = np.zeros(len(bin_edges)-1)

# Populate the counts for each group
for cls in top_5_classes:
    distances = class_distances_diff_label[cls]
    top_counts += np.histogram(distances, bins=bin_edges)[0]

for cls in classes_6_27:
    distances = class_distances_diff_label[cls]
    middle_counts += np.histogram(distances, bins=bin_edges)[0]

for cls in rest_of_classes:
    distances = class_distances_diff_label[cls]
    rest_counts += np.histogram(distances, bins=bin_edges)[0]

# Plot the stacked histogram
plt.figure(figsize=(20, 10))

plt.bar(bin_edges[:-1], top_counts, width=np.diff(bin_edges), color=colors[0], edgecolor='black', label='Majority Classes')
plt.bar(bin_edges[:-1], middle_counts, width=np.diff(bin_edges), bottom=top_counts, color=colors[1], edgecolor='black', label='Middle Classes')
plt.bar(bin_edges[:-1], rest_counts, width=np.diff(bin_edges), bottom=top_counts + middle_counts, color=colors[2], edgecolor='black', label='Minority Classes')

plt.xlabel('Distance', fontsize=28)
plt.ylabel('Frequency', fontsize=28)
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)  
#plt.title('Histogram with Stacked Colors for Different Class Groups')
plt.legend(prop={'size': 22})


plt.savefig(f'Results/stacked_histogram_diff_labels_{model_name}.svg', format='svg')


# Function to create box plot for a group of classes
def create_box_plot(classes, filename):
    pos =1
    data = []
    positions = []
    labels = []
    colors = []
    print(classes)
    for label in classes:
        data.append(class_distances_same_label[label])
        positions.append(pos)
        colors.append('lightblue')  # Color for same label
        data.append(class_distances_diff_label[label])
        positions.append(pos + 1)
        colors.append('lightcoral')  # Color for different label
        labels.append(label_dict[label])
        pos += 2
    # Create the box plot
    plt.figure(figsize=(70, 55))
    box = plt.boxplot(data, positions=positions, patch_artist=True)

    # Set colors for each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set the color of the median lines to black
    for median in box['medians']:
        median.set_color('black')

    # Add vertical lines to separate each pair of boxes
    for i in range(1, len(positions), 2):
        plt.axvline(x=positions[i] + 0.5, color='gray', linestyle='--')
    # Adjust the x-axis tick labels
    #plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    tick_positions = [(positions[i] + positions[i + 1] ) / 2 for i in range(0, len(positions), 2)]
    plt.xticks(ticks=tick_positions, labels=labels, rotation=90, fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel('Classes', fontsize=42)
    plt.ylabel('Distances', fontsize=42)
    lightblue_patch = mpatches.Patch(color='lightblue', label='Distance from the nearest positive sample')
    lightcoral_patch = mpatches.Patch(color='lightcoral', label='Distance from the nearest negative sample')
    plt.legend(handles=[lightblue_patch, lightcoral_patch], prop={'size': 32}, loc='upper right')
    # Save the plot as SVG
  #  plt.legend(handles=[lightblue_patch, lightcoral_patch], loc='upper right', bbox_to_anchor=(1.2, 1), title='Legend', title_fontsize='large', shadow=True, fancybox=True, ncol=1, fontsize='large', facecolor='lightgray', edgecolor='black', labelspacing=1, handlelength=1, handleheight=1, borderpad=1, handletextpad=1, borderaxespad=1, columnspacing=1, prop={'rotation': 90})

    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()

# Create box plots for each group
#create_box_plot(top_5_classes, f'Results/top_5_classes_{model_name}.svg')
#create_box_plot(classes_6_27, f'Results/classes_6_27_{model_name}.svg')
#create_box_plot(rest_of_classes, f'Results/rest_of_classes_{model_name}.svg')
#create_box_plot(sorted_classes, f'Results/all_{model_name}.svg')



'''
# Plot histograms
plt.figure(figsize=(14, 6))

# Histogram for distances with the same label
plt.subplot(1, 2, 1)
plt.hist(all_distances_same_label, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Distances - Same Label')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig(f'Results/distances_same_label_{model_name}.svg', format='svg')
plt.close()

# Histogram for distances with different labels
plt.subplot(1, 2, 2)
plt.hist(all_distances_diff_label, bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution of Distances - Different Labels')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig(f'Results/distances_diff_label_{model_name}.svg', format='svg')
# Show the plots
plt.tight_layout()
plt.close()
'''

def create_histograms_overlap(all_distances_same_label, all_distances_diff_label):
    plt.figure(figsize=(20, 10))

    # Histogram for distances with the same label
    plt.hist(all_distances_same_label, bins=50, alpha=0.5, color='lightblue', edgecolor='black', label='Same Label')

    # Histogram for distances with different labels
    plt.hist(all_distances_diff_label, bins=50, alpha=0.5, color='lightcoral', edgecolor='black', label='Different Labels')
    lightblue_patch = mpatches.Patch(color='lightblue', label='Distance from the nearest positive sample')
    lightcoral_patch = mpatches.Patch(color='lightcoral', label='Distance from the nearest negative sample')
    plt.legend(handles=[lightblue_patch, lightcoral_patch])
    # Add title and labels
    #plt.title('Distribution of Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    #plt.legend()

    # Save the combined plot as an SVG file
    plt.savefig(f'./Results/distances_histograms_overlay_{model_name}.svg', format='svg')
    plt.close()


prams.update({
    'font.size': 18,          # Default text size
    'axes.titlesize': 22,     # Title font size
    'axes.labelsize': 20,     # X and Y label font size
    'xtick.labelsize': 16,    # X-axis tick label font size
    'ytick.labelsize': 16,    # Y-axis tick label font size
    'legend.fontsize': 18     # Legend font size
})

def combine_histograms(set1_same_label, set1_diff_label, set2_same_label, set2_diff_label):
    # Combine all data to calculate the common axis limits
    all_data = np.concatenate([set1_same_label, set1_diff_label, set2_same_label, set2_diff_label])
    x_min, x_max = np.min(all_data), np.max(all_data)
    
    # To set consistent y-scale, we need to plot histograms and find the max height
    _, bins = np.histogram(all_data, bins=50)
    y_max_same = max(
        np.histogram(set1_same_label, bins=bins)[0].max(),
        np.histogram(set2_same_label, bins=bins)[0].max()
    )
    y_max_diff = max(
        np.histogram(set1_diff_label, bins=bins)[0].max(),
        np.histogram(set2_diff_label, bins=bins)[0].max()
    )
    y_max = max(y_max_same, y_max_diff)

    # Combined histogram for distances with the same label
    plt.figure(figsize=(20, 10))
    plt.hist(set1_same_label, bins=bins, alpha=0.5, color='lightblue', edgecolor='black', label='Baseline 1')
    plt.hist(set2_same_label, bins=bins, alpha=0.5, color='darkblue', edgecolor='black', label='TooT-Triplet-offline')
    lightblue_patch = mpatches.Patch(color='lightblue', label='Basline 1')
    darkblue_patch = mpatches.Patch(color='darkblue', label='TooT-Triplet-offline')
    plt.legend(handles=[lightblue_patch, darkblue_patch])
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
  #  plt.title('Combined Histogram: Distances with the Same Label')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
    plt.savefig(f'./Results/same_distances_histograms_overlay_frozen_offline.svg', format='svg')
    # Combined histogram for distances with different labels
    plt.figure(figsize=(20, 10))
    plt.hist(set1_diff_label, bins=bins, alpha=0.5, color='lightcoral', edgecolor='black', label='Baseline 1')
    plt.hist(set2_diff_label, bins=bins, alpha=0.5, color='darkred', edgecolor='black', label='TooT-Triplet-offline')
    lightcoral_patch = mpatches.Patch(color='lightcoral', label='Baseline 1')
    darkred_patch = mpatches.Patch(color='darkred', label='TooT-Triplet-offline')
    plt.legend(handles=[lightcoral_patch, darkred_patch])
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    #plt.title('Combined Histogram: Distances with Different Labels')
    plt.xlim(x_min, x_max)
    plt.ylim(0, y_max)
   # plt.show()lt.savefig(f'./Results/diff_distances_histograms_overlay_frozen_.svg', format='svg')
    plt.savefig(f'./Results/diff_distances_histograms_overlay_frozen_offline.svg', format='svg')

#create_histograms_overlap(all_distances_same_label, all_distances_diff_label)
'''

data = []
positions = []
labels = []
colors = []

pos = 1  # Initial position
for label in class_distances_same_label.keys():
    data.append(class_distances_same_label[label])
    positions.append(pos)
    colors.append('lightblue')  # Color for same label
    data.append(class_distances_diff_label[label])
    positions.append(pos + 1)
    colors.append('lightcoral')  # Color for different label
    labels.append(label_dict[label])
    pos += 2

# Create the box plot
plt.figure(figsize=(20, 10))
box = plt.boxplot(data, positions=positions, patch_artist=True)

# Set colors for each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Set the color of the median lines to black
for median in box['medians']:
    median.set_color('black')

# Add vertical lines to separate each pair of boxes
for i in range(1, len(positions), 2):
    plt.axvline(x=positions[i] + 0.5, color='gray', linestyle='--')

# Adjust the x-axis tick labels to be slightly more to the right
tick_positions = [(positions[i] + positions[i + 1] + 0.5) / 2 for i in range(0, len(positions), 2)]
plt.xticks(ticks=tick_positions, labels=labels)

# Add legend
lightblue_patch = mpatches.Patch(color='lightblue', label='Distances of the nearest positive sample')
lightcoral_patch = mpatches.Patch(color='lightcoral', label='Distances of the nearest negative sample')
plt.legend(handles=[lightblue_patch, lightcoral_patch])




plt.xlabel('Classes')
plt.ylabel('Distances')
plt.title('Box Plot of Distances by Class')
plt.savefig('Results/distances_boxplots_side_by_side.svg', format='svg')


data = []
labels = []

for label in class_distances_same_label.keys():
    data.append(class_distances_same_label[label])
    labels.append(f'{label_dict[label]} (Same)')
    data.append(class_distances_diff_label[label])
    labels.append(f'{label_dict[label]} (Diff)')

# Create the box plot
plt.figure(figsize=(20, 10))
plt.boxplot(data, labels=labels, patch_artist=True)

# Set plot title and labels
plt.title('Distribution of Distances - Same and Different Labels')
plt.xlabel('Class')
plt.ylabel('Distance')

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Adjust layout and save the plot as an SVG file
plt.tight_layout()
plt.savefig('Results/distances_boxplots_side_by_side.svg', format='svg')
plt.close()
'''
'''
data_same = [class_distances_same_label[i] for i in range(num_classes)]
data_diff = [class_distances_diff_label[i] for i in range(num_classes)]
palette = sns.color_palette("tab20", num_classes)

# Create the box plot
plt.figure(figsize=(20, 10))
sns.boxplot(data=data_same, color=palette, flierprops=dict(marker='o', markersize=5), width=0.4)
sns.boxplot(data=data_diff, color=[sns.light_palette(color, n_colors=1, reverse=True)[0] for color in palette], flierprops=dict(marker='o', markersize=5), width=0.4)

# Set plot title and labels
plt.title('Distribution of Distances - Same and Different Labels')
plt.xlabel('Class')
plt.ylabel('Distance')

# Create legend for the plot
import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color=color, label=label_dict[i]) for i, color in enumerate(palette)]
plt.legend(handles=legend_handles, title='Classes', loc='upper right')

# Adjust layout and save the plot as an SVG file
plt.tight_layout()
plt.savefig(f'Results/distances_boxplots_shaded_{model_name}.svg', format='svg')
plt.close()
'''
'''
# Define function to generate a color palette with light and dark shades for each base color
def generate_palette(num_classes):
    base_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                   'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = []
    for color in base_colors[:num_classes]:
        base_color = plt.get_cmap('tab20')(plt.Normalize(vmin=0, vmax=19)(base_colors.index(color)))
        light_color = tuple((x + 1.0) / 2.0 for x in base_color[:-1]) + (base_color[-1],)
        colors.extend([light_color, base_color])
    return colors

# Prepare data for box plots
data = []
labels = []

# Create and save three separate box plots based on the number of samples

# Plot for top 5 classes
top5_num_classes = 5
top5_colors = generate_palette(top5_num_classes)

# Plot for top 5 classes
plt.figure(figsize=(20, 10))
plt.boxplot(data[:10], labels=labels[:10], patch_artist=True, boxprops=dict(facecolor='white'), medianprops=dict(color='black'))
plt.title('Top 5 Classes')
plt.ylabel('Distance')
plt.xlabel('Class')
plt.xticks(rotation=90)
for i, box in enumerate(plt.gca().artists):
    box.set_facecolor(top5_colors[i * 2 + 1])  # Use the dark shade for the box color
plt.tight_layout()
plt.savefig('distances_boxplots_top5.svg', format='svg')
plt.close()

# Plot for classes 6-27
classes6_27_num_classes = 22
classes6_27_colors = generate_palette(classes6_27_num_classes)

# Plot for classes 6-27
plt.figure(figsize=(20, 10))
plt.boxplot(data[10:66], labels=labels[10:66], patch_artist=True, boxprops=dict(facecolor='white'), medianprops=dict(color='black'))
plt.title('Classes 6-27')
plt.ylabel('Distance')
plt.xlabel('Class')
plt.xticks(rotation=90)
for i, box in enumerate(plt.gca().artists):
    box.set_facecolor(classes6_27_colors[i * 2 + 1])  # Use the dark shade for the box color
plt.tight_layout()
plt.savefig('distances_boxplots_classes6_27.svg', format='svg')
plt.close()

# Plot for the rest of the classes
remaining_num_classes = 69
remaining_colors = generate_palette(remaining_num_classes)

# Plot for the rest of the classes
plt.figure(figsize=(20, 10))
plt.boxplot(data[66:], labels=labels[66:], patch_artist=True, boxprops=dict(facecolor='white'), medianprops=dict(color='black'))
plt.title('Remaining Classes')
plt.ylabel('Distance')
plt.xlabel('Class')
plt.xticks(rotation=90)
for i, box in enumerate(plt.gca().artists):
    box.set_facecolor(remaining_colors[i * 2 + 1])  # Use the dark shade for the box color
plt.tight_layout()
plt.savefig('distances_boxplots_remaining.svg', format='svg')
plt.close()



# Calculate average and standard deviation for each class and scenario
data = []
for label in class_distances_same_label.keys():
    avg_same = np.mean(class_distances_same_label[label])
    std_same = np.std(class_distances_same_label[label])
    avg_diff = np.mean(class_distances_diff_label[label])
    std_diff = np.std(class_distances_diff_label[label])
    class_name = label_dict[label]
    size = class_sizes[label]
    data.append([class_name, avg_same, std_same, avg_diff, std_diff, size])

# Create a DataFrame
df = pd.DataFrame(data, columns=['class', 'avg_same_label', 'std_same_label', 'avg_diff_label', 'std_diff_label', 'size'])

# Sort DataFrame based on the size of the class
df = df.sort_values(by='size', ascending=False)



# Save DataFrame as a long table in LaTeX format
with open(f'Results/avg_distance_nn_{model_name}.tex', 'w') as f:
    f.write(df.to_latex(index=False, longtable=True))


'''
