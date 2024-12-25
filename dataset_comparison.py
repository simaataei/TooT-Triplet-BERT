import pandas as pd
from torch.utils.data import Dataset
import csv
from Bio import SeqIO


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

    def get_all_sequences(self):
        return self.dataframe.iloc[:, 0].tolist()



train_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_train_f1.csv')
#val_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_val_f1.csv')
test_dataset = ProteinSeqDataset('./Dataset/transporter_uniprot_ident100_t3_test_f1.csv')

print(len(train_dataset))
#print(len(val_dataset))
print(len(test_dataset))

sequences = []
for record in SeqIO.parse('./Dataset/Transpoter_substrates.fasta','fasta'):
    sequences.append(str(' '.join(record.seq)))
print(sequences)
print('sdf')
com = []
for item in train_dataset:
    #print(item['seq'])
    if item['seq'] in sequences:
       print(item['seq'])
       com.append(item['seq'])
print(len(com))
# Extract sequences from the train dataset
train_sequences = set(test_dataset.get_all_sequences())

# Convert sequences list to a set for fast intersection
fasta_sequences = set(sequences)

# Find common sequences
common_sequences = train_sequences.intersection(fasta_sequences)
print(len(common_sequences))
common_seq_records = [SeqRecord(Seq(seq), id=f"Seq_{i}", description="") for i, seq in enumerate(common_sequences)]
with open("Dataset/sc_spec_common_sequences.fasta", "w") as output_handle:
    SeqIO.write(common_seq_records, output_handle, "fasta")

