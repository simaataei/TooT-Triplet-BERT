import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, AutoConfig, BertModel, BertForSequenceClassification
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.nn import TripletMarginLoss
from model import SubstrateRep, TokenClassifier, SimpleNNWithBERT, reps
from peft import LoraConfig, TaskType
from peft import get_peft_model
from torch.optim import Adam, AdamW
import random
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import pandas as pd
import gc
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
import time
from pycm import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier







eps = 1e-6 # an arbitrary small value to be used for numerical stability tricks 


def euclidean_distance(x1, x2, eps=1e-6):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + eps)

def cosine_distance(x1, x2, p=2, eps=1e-6):
    return 1 - F.cosine_similarity(x1, x2, eps=eps)

def eval(X_train, X_val,y_train, y_val, predictions, label_list):
    overall = []
    cm = ConfusionMatrix(y_val,predictions ,digit=3)
    overall.append(round(cm.overall_stat['ACC Macro'],3))
    overall.append(round(cm.overall_stat['PPV Micro'],3))
    overall.append(round(cm.overall_stat['TPR Macro'],3) if cm.overall_stat['TPR Macro']!= 'None' else '-')
    overall.append(round(cm.overall_stat['F1 Macro'],3))
    overall.append(round(cm.overall_stat['Overall MCC'],3))
    return overall
'''
def eval(X_train, X_val,y_train, y_val, predictions, label_list):
    overall = []
    cm = ConfusionMatrix(y_val,predictions ,digit=3)
    overall.append('Overall')
    overall.append(len(X_train))
    overall.append(len(X_val))
    overall.append(sum(list(cm.TP.values())))
    overall.append(sum(list(cm.FP.values())))
    overall.append(sum(list(cm.FN.values())))
    overall.append(sum(list(cm.TN.values())))

    overall.append(round(cm.overall_stat['ACC Macro'],3))
    overall.append(round(cm.overall_stat['PPV Micro'],3))
    overall.append(round(cm.overall_stat['TPR Macro'],3) if cm.overall_stat['TPR Macro']!= 'None' else '-')
    overall.append(round(cm.overall_stat['F1 Macro'],3))
    overall.append(round(cm.overall_stat['Overall MCC'],3))


    acc = list(cm.class_stat['ACC'].values())
    mcc = list(cm.class_stat['MCC'].values())
    f1 = list(cm.class_stat['F1'].values())
    pre = list(cm.class_stat['PPV'].values())
    rec = list(cm.class_stat['TPR'].values())

    acc = ["{:1.3f}".format(float(i)) for i in acc]
    f1 = ["{:1.3f}".format(float(i)) for i in f1]
    pre = [ "{:1.3f}".format(float(i)) if i != 'None' else '-' for i in pre ]
    rec = ["{:1.3f}".format(float(i)) if i != 'None' else '-'  for i in rec]
    mcc = ["{:1.3f}".format(float(i)) if i != 'None' else '-' for i in mcc ]

    df_train = pd.DataFrame({'Seq':X_train,'Class':y_train})
    df_train = df_train.groupby(by='Class').count()
    df_train.columns = ['Trainset']
    df_train

    df_val = pd.DataFrame({'Seq':X_val,'Class':y_val})
    df_val= df_val.groupby(by='Class').count()
    df_val.columns = ['Validation']
    df_val


    df_ind_ss = pd.DataFrame(list(zip(label_list,list(df_train.Trainset),list(df_val.Validation),acc,pre,rec,f1,mcc)))
    df_ind_ss.columns = ['Substrate','Trainset','Testset','Accuracy', 'Precision','Recall','F1-Score','MCC']

    d_TP = {int(k): int(v) for k,v in cm.TP.items()}
    df_TP = pd.DataFrame.from_dict(d_TP,orient='index')
    df_TP.sort_index(inplace=True)
    df_TP.columns = ['TP']
    d_FP = {int(k): int(v) for k,v in cm.FP.items()}
    df_FP = pd.DataFrame.from_dict(d_FP,orient='index')
    df_FP.sort_index(inplace=True)
    df_FP.columns = ['FP']
    d_FN = {int(k): int(v) for k,v in cm.FN.items()}
    df_FN = pd.DataFrame.from_dict(d_FN,orient='index')
    df_FN.sort_index(inplace=True)
    df_FN.columns = ['FN']
    d_TN = {int(k): int(v) for k,v in cm.TN.items()}
    df_TN = pd.DataFrame.from_dict(d_TN,orient='index')
    df_TN.sort_index(inplace=True)
    df_TN.columns = ['TN']

    print(df_ind_ss.Substrate)
    df_ind_ss_d = df_ind_ss.join(df_TP.join(df_FP).join(df_FN).join(df_TN))
    cols = df_ind_ss_d.columns
    df_ind_ss_d = df_ind_ss_d[['Substrate','Trainset','Testset','TP','FP','FN','TN','Accuracy', 'Precision','Recall','F1-Score','MCC']]
    df_ind_ss_d = df_ind_ss_d.sort_values('Trainset', ascending=False)

    print(df_ind_ss_d.Substrate)
    overall_reshaped = np.reshape(overall, (1, 12))
    overall_df = pd.DataFrame(overall_reshaped, columns=df_ind_ss_d.columns)
    df_ind_ss_d  = pd.concat([df_ind_ss_d,overall_df], ignore_index=True)


    with open(f'./Results/{file_name}_{model_name}_{metric_name}_e{epoch}_detailed_latex.txt', 'w') as f:
         with pd.option_context("max_colwidth", 1000):
              f.write(df_ind_ss_d.to_latex(index=False))

    return round(cm.overall_stat['Overall MCC'],3), round(cm.overall_stat['F1 Macro'],3)

'''
def euclidean_distance_matrix(x):
    """
    Args:
      x: Input tensor of shape (batch_size, embedding_dim)
      
    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """
    # Step 1 - compute the dot product
    dot_product = torch.mm(x, x.t())
    #print('dot_product')
    #print(dot_product)
    # Step 2 - extract the squared Euclidean norm from the diagonal
    squared_norm = torch.diag(dot_product)
    #print('squared_norm')
    #print(squared_norm)
    # Step 3 - compute squared Euclidean distances
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)
   # print('distance_matrix')
   # print(distance_matrix)

    # Get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)
    #print('distance_matrix')
    #print(distance_matrix)
    # Step 4 - compute the non-squared distances
    # Handle numerical stability
    # Derivative of the square root operation applied to 0 is infinite
    # We need to handle by setting any 0 to a small positive number (epsilon)
    eps = 1e-10
    mask = distance_matrix == 0.0
    safe_distance_matrix = distance_matrix + mask.float() * eps
    #print('safe_distance_matrix')
    #print(safe_distance_matrix)
    # Now it is safe to get the square root
    distance_matrix = torch.sqrt(safe_distance_matrix)
    #print('distance_matrix')
    #print(distance_matrix)
    # Undo the trick for numerical stability
    # This step effectively zeros out the previously epsilon-added distances
    distance_matrix = distance_matrix * (1.0 - mask.float())
    return distance_matrix



def get_triplet_mask(labels):
  """compute a mask for valid triplets

  Args:
    labels: Batch of integer labels. shape: (batch_size,)

  Returns:
    Mask tensor to indicate which triplets are actually valid. Shape: (batch_size, batch_size, batch_size)
    A triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j`, `k` are different.
  """
  # step 1 - get a mask for distinct indices

  # shape: (batch_size, batch_size)
  indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
  indices_not_equal = torch.logical_not(indices_equal)
  # shape: (batch_size, batch_size, 1)
  i_not_equal_j = indices_not_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_not_equal_k = indices_not_equal.unsqueeze(1)
  # shape: (1, batch_size, batch_size)
  j_not_equal_k = indices_not_equal.unsqueeze(0)
  # Shape: (batch_size, batch_size, batch_size)
  distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

  # step 2 - get a mask for valid anchor-positive-negative triplets

  # shape: (batch_size, batch_size)
  labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
  # shape: (batch_size, batch_size, 1)
  i_equal_j = labels_equal.unsqueeze(2)
  # shape: (batch_size, 1, batch_size)
  i_equal_k = labels_equal.unsqueeze(1)
  # shape: (batch_size, batch_size, batch_size)
  valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

  # step 3 - combine two masks
  mask = torch.logical_and(distinct_indices, valid_indices)

  return mask



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

def get_embeddings(train_dataset):
    train_emb= []
    y_train =[]
    #emb = torch.empty([1, 1024])    
    for item in train_dataset:
        with torch.no_grad():
             tokenized_seq = tokenizer(item['seq'],max_length=1024, padding=True, truncation=True, return_tensors='pt')
             input_ids = tokenized_seq['input_ids'].to(device)
             attention_mask = tokenized_seq['attention_mask'].to(device)
             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
             embeddings = outputs.last_hidden_state[:, 0, :]
             train_emb.append(embeddings.detach().cpu())
             y_train.append(item['labels'])

             del tokenized_seq, input_ids, attention_mask, outputs, embeddings

           # Manually invoke garbage collection in critical places
             if gc.isenabled():
                gc.collect()
        torch.cuda.empty_cache()
    return train_emb, y_train





def find_optimal_triplets_symmetric(distance_matrix, labels):
    """
    Optimized function to find triplets based on the hardest positive and easiest negative samples
    for a symmetric distance matrix.

    Args:
        distance_matrix (torch.Tensor): Symmetric 2D tensor with distances between samples.
        labels (list or torch.Tensor): Labels for each sample.

    Returns:
        List of tuples: Each tuple contains (anchor_index, positive_index, negative_index).
    """

    device = torch.device("cpu")
    distance_matrix = distance_matrice.to(device)
    
    #if not isinstance(labels, torch.Tensor):
     #   labels = torch.tensor(labels, device=device)
    #else:
     #   labels = labels.to(device)
    #if not isinstance(labels, torch.Tensor):
     #   labels = torch.tensor(labels)
    triplets = []
    n = len(distance_matrice)
    print(n)
    label_masks = {label: (labels == label) for label in np.unique(labels)}
    for anchor_index in range(n):
        anchor_label = labels[anchor_index]
        # Positive mask, but exclude self
        positive_mask = label_masks[anchor_label]
        #positive_mask[anchor_index] = False
#        print('positive_mask')

 #       print(positive_mask)
        negative_mask = ~label_masks[anchor_label]
    #    print('negative mask')
     #   print(negative_mask) 
        positive_mask[anchor_index] = False
        negative_mask[anchor_index] = False
        anchor_distances = distance_matrix[anchor_index]

        if positive_mask.sum() > 0:
            positive_distances = anchor_distances[positive_mask]
            hardest_positive_index = np.argmax(positive_distances)
            hardest_positive = np.where(positive_mask)[0][hardest_positive_index]
        else:
            continue
        # Select easiest negative sample
        if negative_mask.sum() > 0:
            negative_distances = anchor_distances[negative_mask]
            #easiest_negative_index = np.argmin(negative_distances)
            #easiest_negative = np.where(negative_mask)[0][easiest_negative_index]
            non_zero_negative_distances = negative_distances[negative_distances > 0]
            
            if len(non_zero_negative_distances) > 0:
                easiest_negative_index = np.argmin(non_zero_negative_distances)
                # Map the index back to the original indices
                actual_negative_indices = np.where(negative_mask)[0]
                easiest_negative = actual_negative_indices[negative_distances > 0][easiest_negative_index]
            else:
                continue
        else:
            continue

        # Only add triplets where both a positive and negative sample were found
        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            triplets.append((anchor_index, hardest_positive, easiest_negative))
        else:
            continue

    
    
    with open('./Results/positive_mask.txt','w') as f:
        f.write(str(positive_mask))
    with open('./Results/negative_mask.txt','w') as f:
        f.write(str(negative_mask))
    
    # Save the distance matrix and masks to text files
    np.savetxt('./Results/distance_matrix.txt', distance_matrix, fmt='%.6f')
    return triplets

device = torch.device("cuda")
# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

# Create datasets
train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f1.csv')
val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f1.csv')
subset_dataset = ProteinSeqDataset('./Dataset/small_trans.csv')
#batch_size =len(train_dataset) # This should be a multiple of 3
batch_size =len(subset_dataset) 

print(batch_size)
#train_loader = DataLoader(train_dataset, batch_sampler=triplet_sampler)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))

model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_hidden_states=False)

#model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert')
model.half()
print(model)
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.float()  # Converts all model parameters and buffers to float32.

model= nn.DataParallel(model)
model.to(device)

file_name = 'spec'
metric_name = 'manhattan'

print(f"Using {torch.cuda.device_count()} GPUs!")

# Create a new dataset instance with the subset
subset_dataset = ProteinSeqDataset('./Dataset/small_trans.csv')
#train_loader = DataLoader(subset_dataset, batch_size=1, sampler=RandomSampler(subset_dataset))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = AdamW(model.parameters(), lr=1e-6)
triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance,margin=1.0)
patience = 3  # Number of epochs to wait for improvement
best_loss = float('inf')
epochs_no_improve = 0
num_epochs = 8
pre_tri =[]
best_mcc = -1
all_mcc= []
all_f1 = []
# Training Loop
for epoch in range(num_epochs):
    model.train() 
    gpu_index = 0
    train_emb, y_train = get_embeddings(train_dataset)
   # train_emb, y_train = get_embeddings(subset_dataset)
    embeddings= torch.cat(train_emb)
    distance_matrice = euclidean_distance_matrix(embeddings)
    triplets= find_optimal_triplets_symmetric(distance_matrice, y_train)
    print('trip eq'+str((True if triplets== pre_tri else False)))
    pre_tri = triplets
    total_loss =0

    i=0

    with torch.autograd.set_detect_anomaly(True):
         for batch in triplets:

             anchor = train_dataset[int(batch[0])]
             positive = train_dataset[int(batch[1])]
             negative = train_dataset[int(batch[2])]
            # print(anchor['labels'])
             #print(positive['labels'])
             #print(negative['labels'])
             if negative['labels'] == anchor['labels']:
             #   print('breaking')
                continue
            # print('after')
             i+=1
             seqs_anchor = anchor['seq']
             #label_anchor = anchor['labels'].to(device)
             inputs = tokenizer(seqs_anchor, return_tensors='pt', padding=True, truncation=True, max_length=1024)
             input_ids = inputs['input_ids'].to(device)
             attention_mask = inputs['attention_mask'].to(device)
             model.zero_grad() 
             outputs = model(input_ids, attention_mask=attention_mask)
             embeddings_anchor = outputs.last_hidden_state[:, 0, :]  # Using the [CLS] token embeddings




             seqs_positive = positive['seq']
             #label_positive = positive['labels'].to(device)
             inputs = tokenizer(seqs_positive, return_tensors='pt', padding=True, truncation=True, max_length=1024)
             input_ids = inputs['input_ids'].to(device)
             attention_mask = inputs['attention_mask'].to(device)
             model.zero_grad()
             outputs = model(input_ids, attention_mask=attention_mask)
             embeddings_positive = outputs.last_hidden_state[:, 0, :]  # Using the [CLS] token embeddings


             seqs_negative = negative['seq']
             #label_negative = negative['labels'].to(device)
             inputs = tokenizer(seqs_negative, return_tensors='pt', padding=True, truncation=True, max_length=1024)
             input_ids = inputs['input_ids'].to(device)
             attention_mask = inputs['attention_mask'].to(device)
             model.zero_grad()
             outputs = model(input_ids, attention_mask=attention_mask)
             embeddings_negative = outputs.last_hidden_state[:, 0, :] 

             print(type(embeddings_anchor))
             print(embeddings_anchor)
             loss = triplet_loss_fn(embeddings_anchor,embeddings_positive, embeddings_negative)
             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
             #loss = loss +current_loss
             loss.backward(retain_graph=True)
             total_loss += loss.item()

             if i% 10 == 0:
                optimizer.step()
                optimizer.zero_grad()
                print('step'+str(i))
             #total_loss += loss.item()
    avg_loss = total_loss / len(triplets)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    gc.collect()
    del embeddings, outputs, attention_mask,inputs,input_ids,loss


    model_name = f'online_epoch{epoch}'
    torch.save(model.state_dict(), f'models/{metric_name}/{file_name}_{model_name}_{metric_name}_e{epoch}')

'''    

    
    train_emb_torch, y_train = get_embeddings(train_dataset)
    val_emb_torch, y_val = get_embeddings(test_dataset)
    

    train_emb = [i.numpy()[0] for i in train_emb_torch]
    val_emb = [i.numpy()[0] for i in val_emb_torch]
    #train_emb, y_train = get_embeddings(subset_dataset)
    #val_emb, y_val = get_embeddings(subset_dataset[:2])

    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    val_emb = scaler.transform(val_emb)

    knn = KNeighborsClassifier(metric='cosine', n_neighbors=1)
    knn.fit(train_emb, y_train)
    label_dict = {}
    with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
         data = f.readlines()
         for d in data:
             d = d.split(',')
             label_dict[d[2].strip('\n')] = d[1]
    label_dict = {int(k):v for k,v in label_dict.items()}
    label_list = list(label_dict.values())




    distances, indices = knn.kneighbors(val_emb)
    nei_name= [
    [label_dict[y_train[number]] for number in row] for row in indices]
    name_distance = [
    [(nei_name[i][j], distances[i][j]) for j in range(len(nei_name[i]))]
    for i in range(len(distances))]
    print("Name-Distances:\n", name_distance)
    df = pd.DataFrame({'neighbours':name_distance})
    with open(f'./Results/{file_name}_{model_name}_1nn_predictions_{metric_name}.txt', 'w') as f:
        with pd.option_context("max_colwidth", 1000):
            f.write(df.to_latex(index=True))   
    
    predictions =knn.predict(val_emb)
    overall = []
    cm = ConfusionMatrix(y_val,predictions,digit=3)
    overall.append(round(cm.overall_stat['ACC Macro'],3))
    overall.append(round(cm.overall_stat['PPV Micro'],3))
    overall.append(round(cm.overall_stat['TPR Macro'],3) if cm.overall_stat['TPR Macro']!= 'None' else '-')
    overall.append(round(cm.overall_stat['F1 Macro'],3))
    overall.append(round(cm.overall_stat['Overall MCC'],3))

    mcc = overall[4]
    print(mcc)
    all_mcc.append(overall[4])
    all_f1.append(overall[3])
    with open(f"Results/overall_{model_name}_{file_name}_{metric_name}_e{epoch}.txt", "w") as file:
         file.write('Accuracy, Precision, Recall, F1-score, MCC\n')
         for item in overall:
             file.write(str(item) + ", ")
             

    print('mcc'+ str(mcc))
    if mcc> best_mcc:
       torch.save(model.state_dict(), f'models/{file_name}_{model_name}_{metric_name}_e{epoch}')
       best_mcc = mcc


with open(f"Results/mcc_{model_name}_{file_name}_{metric_name}.txt", "w") as file:
    # Iterate through the list and write each item to the file
    for item in all_mcc:
        file.write(str(item) + "\n")


with open(f"Results/f1_{model_name}_{file_name}_{metric_name}.txt", "w") as file:
    # Iterate through the list and write each item to the file
    for item in all_f1:
        file.write(str(item) + "\n")        

    # Early stopping
    if  best_loss - avg_loss>eps:
        best_loss = avg_loss
        epochs_no_improve = 0
       # torch.save(model.state_dict(), 'online_e10')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping')
            break
         #avg_loss = total_loss / len(train_loader)'''

 

