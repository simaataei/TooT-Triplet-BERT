import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from embedding import train_emb, y_train, test_emb, y_test



def load_label_names(filepath):
    label_dict = {}
    with open(filepath) as f:
        data = f.readlines()
        for d in data:
            parts = d.split(',')
            label_dict[int(parts[2].strip())] = parts[1].strip()
    return label_dict

def calculate_class_distances(data, labels, label_dict):
    """Calculate maximum intra-class and minimum inter-class distances, replace numeric labels with names.
    
    Args:
    data (list of lists): Each sublist is an embedding.
    labels (list): An array of class labels corresponding to the embeddings.
    label_dict (dict): Dictionary mapping label numbers to names.
    
    Returns:
    tuple of pd.DataFrame: DataFrames with columns for intra-class distances and inter-class distances,
                           with label names instead of numbers.
    """
    max_intra_class_distances = []
    min_inter_class_distances = []

    classes = set(labels)
    data_array = np.array(data)

    for cls in classes:
        current_indices = [i for i, label in enumerate(labels) if label == cls]
        other_indices = [i for i, label in enumerate(labels) if label != cls]

        current_data = data_array[current_indices]
        other_data = data_array[other_indices]
        other_labels = [labels[i] for i in other_indices]

        num_samples = len(current_data)

        if num_samples > 1:
            intra_distances = pdist(current_data, 'euclidean')
            max_distance = np.max(intra_distances)
        else:
            max_distance = 0

        if len(other_data) > 0 and num_samples > 0:
            inter_distances = cdist(current_data, other_data, 'euclidean')
            min_distance_index = np.argmin(inter_distances)
            min_distance = np.min(inter_distances)
            closest_class_index = min_distance_index % len(other_data)
            closest_class_label = other_labels[closest_class_index]
        else:
            min_distance = float('inf')
            closest_class_label = None

        max_intra_class_distances.append((label_dict.get(cls, cls), max_distance, num_samples))
        min_inter_class_distances.append((label_dict.get(cls, cls), min_distance, label_dict.get(closest_class_label, closest_class_label), num_samples))

    df_max_intra = pd.DataFrame(max_intra_class_distances, columns=['Class', 'Max Distance', 'Number of Samples'])
    df_min_inter = pd.DataFrame(min_inter_class_distances, columns=['Class', 'Min Distance', 'Closest Class', 'Number of Samples'])

    return df_max_intra.sort_values(by='Max Distance', ascending=False), df_min_inter.sort_values(by='Min Distance')

# Load label dictionary
label_dict = load_label_names('./Dataset/Label_name_list_transporter_uni_ident100_t3')


embeddings = train_emb
class_labels =y_train
print(class_labels)
print(type(class_labels[1]))

max_distances_df, min_df = calculate_class_distances(embeddings, class_labels, label_dict)
print("DataFrame sorted by maximum Euclidean distances:", max_distances_df)
print("DataFrame sorted by minimum Euclidean distances:", min_df)

with open('Results/class_distance/min_train.tex', 'w') as tf:
     tf.write(min_df.to_latex(index = False))

with open('Results/class_distance/max_train.tex', 'w') as tf:
     tf.write(max_distances_df.to_latex(index = False))

embeddings = train_emb + test_emb
class_labels =y_train + y_test
print(class_labels)
print(type(class_labels[1]))

max_distances_test_df, min_test_df = calculate_class_distances(embeddings, class_labels, label_dict)
print("DataFrame sorted by maximum Euclidean distances:", max_distances_df)
print("DataFrame sorted by minimum Euclidean distances:", min_df)


with open('Results/class_distance/min_train_test.tex', 'w') as tf:
     tf.write(min_test_df.to_latex(index = False))

with open('Results/class_distance/max_train_test.tex', 'w') as tf:
     tf.write(max_distances_test_df.to_latex(index = False))



