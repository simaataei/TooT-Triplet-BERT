import os
import numpy as np
from transformers import BertModel, BertTokenizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from peft import LoraConfig, TaskType
from peft import get_peft_model
from torch.optim import Adam
import random
from transformers import BertTokenizer, BertModel
import seaborn as sns
from Bio import SeqIO


def visualize_attention_map(attention_maps):
    i=0
    # Average attention scores across heads
    for attention_map in attention_maps:
        avg_attention_map = torch.mean(attention_map, dim=1).squeeze(0).detach().cpu().numpy()

    # Plot the attention heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention_map, cmap='viridis', interpolation='nearest')
        plt.xlabel('Input tokens')
        plt.ylabel('Input tokens')
        plt.title('Attention Map')
        plt.colorbar()
        plt.savefig(f'./attention/{name}/attention_weights_last_layer_head_'+str(i)+'.png')

        i+=1


def visualize_attention_layers(atttentions):
    for layer in range(len(attentions)):
        for head in range(attentions[layer].shape[1]):
            attention_weights = attentions[layer][0, head].detach().cpu().numpy()  # shape: (seq_length, seq_length)
        
            # Get the positions of the top 10 values
            top_indices = np.unravel_index(np.argsort(-attention_weights, axis=None)[:10], attention_weights.shape)
            print(top_indices)    
        
            plt.figure(figsize=(10, 8), dpi=300)  # Adjust the figure size and DPI
            sns.heatmap(attention_weights, cmap='viridis', annot=False, cbar=False)  # Remove colorbar
        
            # Bold the top 10 values
            max_idx = np.unravel_index(np.argmax(attention_weights, axis=None), attention_weights.shape)
            plt.gca().set_xticks(np.arange(attention_weights.shape[1]) + 0.5, minor=False)
            plt.gca().set_yticks(np.arange(attention_weights.shape[0]) + 0.5, minor=False)
            plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
            plt.gca().xaxis.set_tick_params(labelsize=6)  # Set x-axis label size
            plt.gca().yaxis.set_tick_params(labelsize=6)  # Set y-axis label size
            plt.gca().set_xticklabels([f'{x}\n{"[MAX]" if i == max_idx[1] else ""}' for i, x in enumerate(plt.gca().get_xticks())], ha='center', fontsize=6)
            plt.gca().set_yticklabels([f'{y}\n{"[MAX]" if i == max_idx[0] else ""}' for i, y in enumerate(plt.gca().get_yticks())], va='center', fontsize=6)

            plt.title(f'Attention Weights of Layer {layer} - Head {head}', fontsize=10)
            plt.xlabel('Token Position', fontsize=8)
            plt.ylabel('Token Position', fontsize=8)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.savefig(f'{output_dir}/attention_weights_layer_{layer}_head_{head}.png', bbox_inches='tight', format='png', dpi=80)
            # plt.savefig(f'{output_dir}/attention_weights_layer_{layer}_head_{head}.svg', bbox_inches='tight', format='svg', dpi=80)
            plt.close()

def visualize_attention_mean2(attention_maps):
    i = 0
    # Average attention scores across heads
    for attention_map in attention_maps:
        avg_attention_map = torch.mean(attention_map, dim=1).squeeze(0).detach().cpu().numpy()

        # Find the indices of the highest values in each row
        max_indices = avg_attention_map.argmax(axis=1)

        # Create labels for x-axis
        x_labels = ['' for _ in range(avg_attention_map.shape[1])]
        for idx in max_indices:
            x_labels[idx] = str(idx)

        # Plot the attention heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention_map, cmap='viridis', aspect='auto')
        plt.colorbar()

        # Set x-axis ticks and labels with rotation and smaller font
        plt.xticks(ticks=range(avg_attention_map.shape[1]), labels=x_labels, rotation=90, fontsize=8)
        plt.yticks(ticks=range(avg_attention_map.shape[0]), fontsize=8)
        plt.xlabel('Input tokens')
        plt.ylabel('Input tokens')
        plt.title('Attention Map')

        plt.savefig(f'./attention/{name}/attention_weights_last_layer_head_' + str(i) + '.png')
        plt.close()  # Close the plot to avoid overlap

        i += 1




def visualize_attention_mean(attention_maps, name):
    if not os.path.exists(f'./attention/{name}'):
        os.makedirs(f'./attention/{name}')
    
    i = 0
    for attention_map in attention_maps:
        avg_attention_map = torch.mean(attention_map, dim=1).squeeze(0).detach().cpu().numpy()

        # Find the indices of the highest values in each row
        max_indices = avg_attention_map.argmax(axis=1)

        # Create labels for x-axis
        x_labels = ['' for _ in range(avg_attention_map.shape[1])]
      
     #   for idx in max_indices:
      #      x_labels[idx] = str(idx)
        if len(max_indices) == 1:
           idx = max_indices[0]
           x_labels[idx] = str(idx)
           # Add the indices of the top 3 highest values if there is only one max index
           top_indices = avg_attention_map[idx].argsort()[-4:][::-1]  # Top 4 indices including the max index
           for top_idx in top_indices:
               x_labels[top_idx] = str(top_idx)
        else:
           for idx in max_indices:
               x_labels[idx] = str(idx)




        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention_map, cmap='viridis', aspect='auto')
        plt.colorbar()

        # Set x-axis ticks and labels with rotation and smaller font
        plt.xticks(ticks=range(avg_attention_map.shape[1]), labels=x_labels, rotation=90, fontsize=8)
  #      plt.yticks(ticks=y_labels, fontsize=8)

        plt.xlabel('Input tokens', fontsize=14)
        plt.ylabel('Input tokens', fontsize=14)

        # Set the title with the head number
        plt.title(f'Head  #{i}', fontsize=16)
        plt.tight_layout()

        plt.savefig(f'./attention/{name}/attention_weights_last_layer_head_' + str(i) + '.png')
        plt.close()  # Close the plot to avoid overlap

        i += 1
device = torch.device("cuda")


# Load model and tokenizer
model = BertModel.from_pretrained('Rostlab/prot_bert_bfd', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')

# Modify model for LoRA
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, r=1, lora_alpha=1, lora_dropout=0.1, target_modules=["embedding", "query", "key", "value"]
)
model = get_peft_model(model, lora_config)
model.to(device)
model.print_trainable_parameters()
model_path = './triplet_model_97e'
model.load_state_dict(torch.load(model_path))

name = 'P06703'
fasta_file = './Dataset/P06703.fasta'
# Sample input sequence
for record in SeqIO.parse(fasta_file, "fasta"):
    x = ' '.join(record.seq)
    x = x.replace('U', 'X')
    x = x.replace('Z', 'X')
    x = x.replace('O', 'X')
    x = x.replace('B', 'X')

inputs = tokenizer(x, return_tensors='pt').to(device)

# Run through the model
outputs = model(**inputs)
output_dir = 'Results/{name}/mean'
visualize_attention_mean(outputs.attentions, name)



'''
# Get the attention weights from the last layer
attentions = outputs.attentions[-1]  # shape: (batch_size, num_heads, seq_length, seq_length)
for head in range(attentions.shape[1]):
    attention_weights = attentions[0, head].detach().cpu().numpy()  # shape: (seq_length, seq_length)
    top_indices = np.unravel_index(np.argsort(-attention_weights, axis=None)[:10], attention_weights.shape)
    print(top_indices)    
    #plt.figure(figsize=(6, 4), dpi=80)  # Adjust the figure size and DPI
    plt.figure(figsize=(10, 8),dpi=300)
    sns.heatmap(attention_weights, cmap='viridis', annot=False, cbar=False)  # Remove colorbar
    
    # Bold the top 10 values
    max_idx = np.unravel_index(np.argmax(attention_weights, axis=None), attention_weights.shape)
    plt.gca().set_xticks(np.arange(attention_weights.shape[1]) + 0.5, minor=False)
    plt.gca().set_yticks(np.arange(attention_weights.shape[0]) + 0.5, minor=False)
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top
    plt.gca().xaxis.set_tick_params(labelsize=6)  # Set x-axis label size
    plt.gca().yaxis.set_tick_params(labelsize=6)  # Set y-axis label size
    plt.gca().set_xticklabels([f'{x}\n{"[MAX]" if i == max_idx[1] else ""}' for i, x in enumerate(plt.gca().get_xticks())], ha='center', fontsize=6)
    plt.gca().set_yticklabels([f'{y}\n{"[MAX]" if i == max_idx[0] else ""}' for i, y in enumerate(plt.gca().get_yticks())], va='center', fontsize=6)

    plt.title(f'Attention Weights of Last Layer - Head {head}', fontsize=10)
    plt.xlabel('Token Position', fontsize=8)
    plt.ylabel('Token Position', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.savefig(f'./attention/{name}/attention_weights_last_layer_head_{head}.png', bbox_inches='tight', format='png', dpi=80)
    #plt.savefig(f'attention_weights_last_layer_head_{head}.svg', bbox_inches='tight', format='svg', dpi=80)
    plt.close()

'''
