from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from embedding import train_emb, test_emb
# Assume 'X' is your dataset of embeddings
import matplotlib.pyplot as plt



X = train_emb +test_emb  # Your embeddings data here


#Maximal Silhouette Scores
# Testing different numbers of clusters and recording Silhouette scores
silhouette_scores = []
K = range(2, 97)  # Testing from 2 up to 96 clusters
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X, labels))

# Finding the optimal number of clusters based on Silhouette scores
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
optimal_ss = max(silhouette_scores)

print(f"Optimal number of clusters: {optimal_k}")
print(f"Maximal Silhouette Score: {optimal_ss}")
'''


# elbow:
range_clusters = range(2, 96)  # You might adjust this range based on your specific dataset size and characteristics

# Calculating WCSS for each number of clusters
wcss = []
for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # inertia_ is the WCSS for the fitted data

# Plotting the Elbow Curve
plt.figure(figsize=(10, 8))
plt.plot(range_clusters, wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(ticks=range(1, 30))  # Ensure all potential cluster numbers are labeled
plt.grid(True)
plt.show()
'''
