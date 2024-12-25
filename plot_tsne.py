from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from embedding import  test_emb, y_test, train_emb, y_train

test_emb = train_emb 
y_test = y_train


print(len(y_test))
print(len(test_emb))
test_emb = [emb for emb, label in zip(test_emb, y_test) if label > 95 or label<10]
y_test = [label for label in y_test if label > 95 or label<10]
print(max(y_test))
print(len(y_test))
print(len(test_emb))
print(y_test)

embeddings_stack = np.vstack(test_emb)
# Fit t-SNE
tsne = TSNE(n_components=2, perplexity=1, n_iter=1000)
tsne_results = tsne.fit_transform(embeddings_stack)
'''
unique_classes, counts = np.unique(y_test, return_counts=True)
classes_with_three_samples = unique_classes[counts == 3]
print(classes_with_three_samples)
# Filter data for classes with exactly 3 samples
#mask = np.isin(y_test, classes_with_three_samples)

#mask = [indx for indx in range(len(y_test)) if y_test[indx]>72]
#print(mask)
print('tsns')
print(tsne_results)
print(type(tsne_results))
print(type(tsne_results[0]))
y_test = np.array(y_test)
filtered_y_test = np.array([i for i in y_test if i>85 or i<10])
filtered_tsne_results = tsne_results[y_test>72 or y_test<10]
#filtered_tsne_results = [tsne_results[i] for i in mask] 
#filtered_y_test = [y_test[i] for i in mask]
# Plotssuming test_emb and y_test are lists and y_test contains numerical values.
#filtered_test_emb = [emb for emb, label in zip(test_emb, y_test) if label <= 72]
#filtered_y_test = [label for label in y_test if label <= 72]
print('filtered')
# Now filtered_test_emb and filtered_y_test contain only the elements where y_test <= 72
print(filtered_tsne_results)
print(filtered_y_test)
plt.figure(figsize=(14, 10))
scatter = plt.scatter(filtered_tsne_results[:, 0], filtered_tsne_results[:, 1], c=filtered_y_test, cmap='Spectral', s=5)
plt.colorbar(scatter, boundaries=np.arange((96-72) + 1) - 0.5).set_ticks(np.arange((96-72)))
plt.title('t-SNE projection of classes with 3 samples', fontsize=20)
#filtered_y_test = y_test[mask]
#filtered_y_test = y_test[mask]
plt.show()
plt.savefig('tsne_rep_triplet_off_e97_mask_after_tsne_top_bottom.svg', format='svg')


'''
# Plot
plt.figure(figsize=(14, 10))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_test, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(97)-0.5).set_ticks(np.arange(96))
plt.title('t-SNE projection of the dataset', fontsize=20)
plt.show()
plt.savefig('tsne_rep_triplet_off_e97.svg', format='svg')

