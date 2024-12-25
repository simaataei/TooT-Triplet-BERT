import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import BertTokenizer, AutoConfig, BertModel, BertForSequenceClassification, AdamW
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
import numpy as np
from torch.nn import TripletMarginLoss
from model import SubstrateRep, TokenClassifier, SimpleNNWithBERT, reps
from peft import LoraConfig, TaskType
from peft import get_peft_model
from torch.optim import Adam
import random
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import pandas as pd
import gc
import torch.nn as nn
import torch.nn.functional as F
from pycm import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist, pdist, squareform


def manhattan_distance(x1, x2):
    return torch.cdist(x1, x2, p=1)

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
'''    overall.append('Overall')
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

    return round(cm.overall_stat['Overall MCC'],3), round(cm.overall_stat['F1 Macro'],3)'''

    #return overall
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
             train_emb.append(embeddings.detach().cpu().numpy()[0])
             y_train.append(item['labels'])

             del tokenized_seq, input_ids, attention_mask, outputs, embeddings

           # Manually invoke garbage collection in critical places
             if gc.isenabled():
                gc.collect()
        torch.cuda.empty_cache()
    return train_emb, y_train




class AnchorBatchSampler(Sampler):
    """This sampler will just produce indices for anchors. 
    The actual triplet formation will happen during training."""
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        np.random.shuffle(self.indices)  # Shuffle indices
        batches = [self.indices[i:i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)]
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size
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



class ProteinDataset(Dataset):
    def __init__(self, sequences_filepath, labels_filepath, tokenizer, max_length=1024):
        # Read sequences and labels from files
        with open(sequences_filepath, 'r') as seq_file:
            self.sequences = seq_file.read().splitlines()
            
        with open(labels_filepath, 'r') as label_file:
            self.labels = label_file.read().splitlines()
        
        assert len(self.sequences) == len(self.labels), "Sequences and labels files must have the same number of lines."
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        assert isinstance(idx, int), f"Index must be an integer, got {type(idx)}"
        
        # Retrieve sequence and label by index
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize the sequence
        encoding = self.tokenizer(sequence, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        # Convert label to tensor
        label_tensor = torch.tensor(int(label), dtype=torch.long)  # Convert label to int and then to tensor
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label_tensor
        }

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

# Create datasets
train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f3.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f3.csv')
total_dataset =  ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3.csv')
subset_dataset = ProteinSeqDataset('./Dataset/small_trans.csv')
batch_size = 3  # This should be a multiple of 3
triplet_sampler = SimpleTripletSampler(train_dataset, batch_size)

train_loader = DataLoader(train_dataset, batch_sampler=triplet_sampler)

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

#triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance,margin=1.0)

triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance,margin=1.0)
optimizer = AdamW(model.parameters(), lr=1e-5)


file_name = 'spec'
metric_name = 'cosine'
model_name = 'offline'
fold = 'f3'

best_mcc = -1
all_mcc= []
all_f1 = []
# Training Loop
num_epochs = 100  # or any number of epochs you find appropriate
for epoch in range(num_epochs+1):
    total_loss = 0
    model.train()
    i =0
    for batch in train_loader:
        i+=1
       # print('batch')
       # print(batch)
        tokenized_batch = tokenizer(batch['seq'],max_length = 1024, padding=True, truncation=True, return_tensors='pt')        
       # print('tokens')
       # print(tokenized_batch)
       
        input_ids = tokenized_batch['input_ids'].to(device)
        attention_mask = tokenized_batch['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask) 
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Define anchor, positive, and negative samples from the embeddings
        anchor = embeddings[0].unsqueeze(0)  # First sequence as anchor
        positive = embeddings[1].unsqueeze(0)  # Second sequence as positive
        negative = embeddings[2].unsqueeze(0)  # Third sequence as negative
        #print(embeddings)

        # Calculate the loss
        loss = triplet_loss_fn(anchor, positive, negative)
        print(loss)
        #loss = loss + current_loss
        print('loss: '+str(loss))
        del tokenized_batch, input_ids, attention_mask, outputs, embeddings, anchor, positive, negative
        gc.collect()
        loss.backward()
#
        # Update the model's weights
        if i%10 == 0:
           optimizer.step()
           optimizer.zero_grad()

        gc.collect()
    #if epoch%10==0:
torch.save(model.state_dict(), f'models/{metric_name}/{file_name}_{model_name}_{metric_name}_e{epoch}_{fold}')

'''        if epoch>0:
           model_name = f'offline_epoch{epoch}'
           train_emb, y_train = get_embeddings(train_dataset)
           val_emb, y_val = get_embeddings(test_dataset)
           print(y_val)
           #train_emb, y_train = get_embeddings(subset_dataset)
           #val_emb, y_val = get_embeddings(subset_dataset[:2])
        
           scaler = StandardScaler()
           train_emb = scaler.fit_transform(train_emb)
           val_emb = scaler.transform(val_emb)
           knn = KNeighborsClassifier(metric=metric_name, n_neighbors=1)
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
           print(indices)
           nei_name= [
           [label_dict[y_train[number]] for number in row] for row in indices]
           name_distance = [
           [(nei_name[i][j], distances[i][j]) for j in range(len(nei_name[i]))]
           for i in range(len(distances))]
           print("Name-Distances:\n", name_distance)
           df = pd.DataFrame({'neighbours':name_distance})
           with open(f'./Results/{file_name}_{model_name}_1nn_predictions_{metric_name}_e{epoch}.txt', 'w') as f:
                with pd.option_context("max_colwidth", 1000):
                     f.write(df.to_latex(index=True))
           

           predictions = knn.predict(val_emb)    
           overall =[]
           cm = ConfusionMatrix(y_val,predictions ,digit=3)
           overall.append(round(cm.overall_stat['ACC Macro'],3))
           overall.append(round(cm.overall_stat['PPV Micro'],3))
           overall.append(round(cm.overall_stat['TPR Macro'],3) if cm.overall_stat['TPR Macro']!= 'None' else '-')
           overall.append(round(cm.overall_stat['F1 Macro'],3))
           overall.append(round(cm.overall_stat['Overall MCC'],3))
           mcc =round(cm.overall_stat['Overall MCC'],3)
           all_mcc.append(overall[4])
           all_f1.append(overall[3])
           with open(f"Results/overall_{model_name}_{file_name}_{metric_name}_e{epoch}.txt", "w") as file:
                file.write('Accuracy, Precision, Recall, F1-score, MCC\n')
                for item in overall:
                    file.write(str(item) + ", ")

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
'''
