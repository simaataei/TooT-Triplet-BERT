from sklearn.neighbors import KNeighborsClassifier
from embedding_frozen import train_emb, test_emb, train_labels, test_labels
from sklearn.preprocessing import StandardScaler
from evaluate import evaluate


dis_metric = 'cosine'
knn = KNeighborsClassifier(metric=dis_metric,algorithm='brute',n_neighbors=1)
scaler = StandardScaler()
train_emb = scaler.fit_transform(train_emb)
test_emb = scaler.transform(test_emb)
# Train the classifier
knn.fit(train_emb, train_labels)
predictions = knn.predict(test_emb)
model_name = f'spec_finetuned_{dis_metric}_e97_f1'
evaluate(model_name, predictions, test_labels)

