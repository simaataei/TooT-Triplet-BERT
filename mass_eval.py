import knn 
import evaluate 
import os



def list_files(directory):
    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


metric = 'euclidean'
path = f'./models/{metric}/'

models =list_files(path)
print(models)
for model in models:
    print(model)
    all_pred, all_gold = knn.cal_knn(model)
    evaluate.evaluate(model, all_pred, all_gold)
    evaluate.conf_matrix(model, all_pred, all_gold)








