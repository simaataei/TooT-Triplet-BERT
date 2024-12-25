from embedding import train_emb, y_train
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




label_dict = {}
with open('./Dataset/Label_name_list_transporter_uni_ident100_t3') as f:
    data = f.readlines()
    for d in data:
        parts = d.split(',')
        label_dict[parts[2].strip()] = parts[1]  # Assuming the file format is consistent as described

# Convert keys to integer and prepare label list
label_dict = {int(k): v for k, v in label_dict.items()}
label_list = list(label_dict.values())


test_emb = train_emb
y_test = y_train


# Filter embeddings and labels based on your condition
test_emb = [emb for emb, label in zip(test_emb, y_test) if label > 68]
y_test = [label for label in y_test if label > 68]

# Stack embeddings into a matrix
embeddings_stack = np.vstack(test_emb)

# Fit PCA
pca = PCA(n_components=2)
pca_results = pca.fit_transform(embeddings_stack)

# Plotting
plt.figure(figsize=(8, 6))

scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=y_test, cmap='tab10', alpha=0.6)
handles, labels = scatter.legend_elements()
legend_labels = [label_dict[int(lbl)] for lbl in labels]  # Convert numeric labels to names using label_dict
plt.legend(handles, legend_labels, title="Substrate")



plt.title('PCA of Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Save the plot as an SVG file
plt.savefig('Results/pca_embeddings_offline_e97_f2_bottom_68.svg', format='svg')

plt.show()



