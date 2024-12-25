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
from torch.optim import Adam
import random
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import pandas as pd
import gc
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
eps = 1e-8 # an arbitrary small value to be used for numerical stability tricks 
torch.autograd.set_detect_anomaly(True)
def euclidean_distance_matrix2(x, y):
    # x and y are 2D tensors of the form (n_samples, n_features)
    x2 = torch.sum(x**2, dim=1, keepdim=True)
    y2 = torch.sum(y**2, dim=1, keepdim=True)
    xy = torch.matmul(x, y.T)
    dist = x2 + y2.T - 2 * xy  # No in-place operation
    return toch.sqrt(torch.relu(dist))  # Ensure non-negative before sqrt

def euclidean_distance_matrix2(x):
  """Efficient computation of Euclidean distance matrix

  Args:
    x: Input tensor of shape (batch_size, embedding_dim)
    
  Returns:
    Distance matrix of shape (batch_size, batch_size)
  """
  # step 1 - compute the dot product

  
  
  with torch.autograd.set_detect_anomaly(True):
    # shape: (batch_size, batch_size)
    dot_product = torch.mm(x, x.t())

    # step 2 - extract the squared Euclidean norm from the diagonal

    # shape: (batch_size,)
    squared_norm = torch.diag(dot_product)

    # step 3 - compute squared Euclidean distances

    # shape: (batch_size, batch_size)
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

  # get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)

  # step 4 - compute the non-squared distances
  
  # handle numerical stability
  # derivative of the square root operation applied to 0 is infinite
  # we need to handle by setting any 0 to eps
    mask = (distance_matrix == 0.0).float()

  # use this mask to set indices with a value of 0 to eps
    distance_matrix += mask * eps

  # now it is safe to get the square root
    distance_matrix = torch.sqrt(distance_matrix)

  # undo the trick for numerical stability
    distance_matrix *= (1.0 - mask)

  return distance_matrix

import torch
import torch.nn.functional as F

def euclidean_distance_matrix(x):
    """
    Args:
      x: Input tensor of shape (batch_size, embedding_dim)
      
    Returns:
      Distance matrix of shape (batch_size, batch_size)
    """
    # Step 1 - compute the dot product
    dot_product = torch.mm(x, x.t())

    # Step 2 - extract the squared Euclidean norm from the diagonal
    squared_norm = torch.diag(dot_product)

    # Step 3 - compute squared Euclidean distances
    distance_matrix = squared_norm.unsqueeze(0) - 2 * dot_product + squared_norm.unsqueeze(1)

    # Get rid of negative distances due to numerical instabilities
    distance_matrix = F.relu(distance_matrix)

    # Step 4 - compute the non-squared distances
    # Handle numerical stability
    # Derivative of the square root operation applied to 0 is infinite
    # We need to handle by setting any 0 to a small positive number (epsilon)
    eps = 1e-10
    mask = distance_matrix == 0.0
    safe_distance_matrix = distance_matrix + mask.float() * eps

    # Now it is safe to get the square root
    distance_matrix = torch.sqrt(safe_distance_matrix)

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
class BatchAllTripletLoss(nn.Module):
  """Uses all valid triplets to compute Triplet loss

  Args:
    margin: Margin value in the Triplet Loss equation
  """
  def __init__(self, margin=1.):
    super().__init__()
    self.margin = margin
    
  def forward(self, embeddings, labels):
    """computes loss value.

    Args:
      embeddings: Batch of embeddings, e.g., output of the encoder. shape: (batch_size, embedding_dim)
      labels: Batch of integer labels associated with embeddings. shape: (batch_size,)

    Returns:
      Scalar loss value.
    """
    # step 1 - get distance matrix
    # shape: (batch_size, batch_size)
    distance_matrix = euclidean_distance_matrix(embeddings)

    print(distance_matrix)
    # step 2 - compute loss values for all triplets by applying broadcasting to distance matrix

    # shape: (batch_size, batch_size, 1)
    anchor_positive_dists = distance_matrix.unsqueeze(2)
    # shape: (batch_size, 1, batch_size)
    anchor_negative_dists = distance_matrix.unsqueeze(1)
    # get loss values for all possible n^3 triplets
    # shape: (batch_size, batch_size, batch_size)
    triplet_loss = anchor_positive_dists - anchor_negative_dists + self.margin
    print(triplet_loss)
    # step 3 - filter out invalid or easy triplets by setting their loss values to 0

    # shape: (batch_size, batch_size, batch_size)
    mask = get_triplet_mask(labels)
    print(mask)
    triplet_loss *= mask
    # easy triplets have negative loss values
    triplet_loss = F.relu(triplet_loss)
    print(triplet_loss) 
    # step 4 - compute scalar loss value by averaging positive losses
    num_positive_losses = (triplet_loss > eps).float().sum()
    triplet_loss = triplet_loss.sum() / (num_positive_losses + eps)
    print(triplet_loss)
    return triplet_loss



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
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

# Create datasets
train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f1.csv')
val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f2.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f1.csv')

batch_size =3 # This should be a multiple of 3

#train_loader = DataLoader(train_dataset, batch_sampler=triplet_sampler)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))

model = BertModel.from_pretrained('Rostlab/prot_bert', output_hidden_states=True)
#model.half()
print(model)
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1,  target_modules= ["embedding", "query","key","value"])

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)



# Create a new dataset instance with the subset
#subset_dataset = ProteinSeqDataset('./Dataset/small_trans.csv')
#train_loader = DataLoader(subset_dataset, batch_size=1, sampler=RandomSampler(subset_dataset))

optimizer = Adam(model.parameters(), lr=1e-6)
triplet_loss_fn = BatchAllTripletLoss(margin=1.0)
num_epochs = 2
# Training Loop
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    with torch.autograd.set_detect_anomaly(True):
         for batch in train_loader:
             seqs = batch['seq']
             labels = batch['labels'].to(device)

        # Tokenize the batch of sequences
             inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
             print(inputs)
             input_ids = inputs['input_ids'].to(device)
             attention_mask = inputs['attention_mask'].to(device)

             outputs = model(input_ids, attention_mask=attention_mask)
             embeddings = outputs.last_hidden_state[:, 0, :]  # Using the [CLS] token embeddings
        
             loss = triplet_loss_fn(embeddings, labels)

             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

             total_loss += loss.item()

         avg_loss = total_loss / len(train_loader)
         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Validation step (optional, for monitoring overfitting)
    # You can add your validation logic here using `model.eval()` and turning off gradients

# Save the model after training
torch.save(model.state_dict(), 'finetuned_bert_model.pth')


