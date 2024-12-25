from embedding_peft import y_train, train_emb
from embedding_frozen import frozen_emb,y_frozen
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import matplotlib.patches as mpatches
import os
def load_label_names(filepath):
    label_dict = {}
    with open(filepath) as f:
        data = f.readlines()
        for d in data:
            parts = d.split(',')
            label_dict[int(parts[2].strip())] = parts[1].strip()
    return label_dict

def silhouette_score_per_class(X, labels, output_file='Results/silhouette_scores_perclass.txt'):
    """
    Calculate the silhouette score for each class and save them to a text file.
    
    Parameters:
    X (array-like): Feature dataset.
    labels (array-like): Class labels.
    output_file (str): Path to the output text file.
    
    Returns:
    dict: Silhouette scores for each class.
    """
    # Compute pairwise distances
    distance_matrix = pairwise_distances(X)
    
    # Set the diagonal to zero to avoid self-distance issues
    np.fill_diagonal(distance_matrix, 0)
    
    # Compute silhouette scores for all samples
    silhouette_vals = silhouette_samples(distance_matrix, labels, metric='precomputed')
    
    # Dictionary to hold silhouette scores for each class
    class_silhouette_scores = {}
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Save scores to a text file
    with open(output_file, 'w') as file:
        for cls in unique_classes:
            # Get the indices for samples belonging to the current class
            class_indices = np.where(labels == cls)[0]
            
            # Extract the silhouette scores for the current class
            scores = silhouette_vals[class_indices]
            class_silhouette_scores[cls] = scores
            
            # Write scores to file
            file.write(f'Class {cls}:\n')
            file.write('\n'.join(map(str, scores)))
            file.write('\n\n')
            
    return class_silhouette_scores


def plot_bar_chart(class_silhouette_scores, label_dict,class_sizes, output_svg='Results/silhouette_scores_by_class.svg'):
    """
    Plot a bar chart of average silhouette scores for each class and save it as an SVG file.
    
    Parameters:
    class_silhouette_scores (dict): Dictionary of silhouette scores for each class.
    label_dict (dict): Dictionary mapping class labels to class names.
    output_svg (str): Path to the output SVG file.
    """
    

    # Calculate the average silhouette score for each class
    avg_scores = {cls: np.mean(scores) for cls, scores in class_silhouette_scores.items()}
    classes = list(avg_scores.keys())
    average_scores = list(avg_scores.values())
    sizes = [class_sizes[cls] for cls in classes]

    # Sort by class sizes (largest to smallest)
    sorted_indices = np.argsort(sizes)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_scores = [average_scores[i] for i in sorted_indices]
    sorted_class_names = [label_dict[cls] for cls in sorted_classes]

    # Create a bar chart
    plt.figure(figsize=(15, 10))
    bars = plt.bar(sorted_class_names, sorted_scores, color='#2c3968')  # Use specified color

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    # Adding labels and title


    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
   # plt.title('Average Silhouette Scores by Class', fontsize=14)
    
    # Rotate x-axis labels 90 degrees
    plt.xticks(rotation=90,fontsize=9)
    plt.ylim(bottom=-1)
    plt.ylim(top=1)
    # Adjust layout to ensure labels are fully visible
    # Reduce space between the bars and the plot edges
    plt.xlim(-0.5, len(sorted_class_names) - 0.5)

    # Adjust layout to ensure labels are fully visible
    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()



def plot_stacked_bar_chart(class_silhouette_scores1, class_silhouette_scores2, label_dict, class_sizes, output_svg='Results/silhouette_scores_by_class.svg'):
    """
    Plot a stacked bar chart of average silhouette scores for each class from two different sets and save it as an SVG file.

    Parameters:
    class_silhouette_scores1 (dict): Dictionary of silhouette scores for each class (first set).
    class_silhouette_scores2 (dict): Dictionary of silhouette scores for each class (second set).
    label_dict (dict): Dictionary mapping class labels to class names.
    class_sizes (dict): Dictionary mapping class labels to class sizes.
    output_svg (str): Path to the output SVG file.
    """

    # Calculate the average silhouette score for each class
    avg_scores1 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores1.items()}
    avg_scores2 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores2.items()}
    classes = list(avg_scores1.keys())
    average_scores1 = list(avg_scores1.values())
    average_scores2 = list(avg_scores2.values())
    sizes = [class_sizes[cls] for cls in classes]

    # Sort by class sizes (largest to smallest)
    sorted_indices = np.argsort(sizes)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_scores1 = [average_scores1[i] for i in sorted_indices]
    sorted_scores2 = [average_scores2[i] for i in sorted_indices]
    sorted_class_names = [label_dict[cls] for cls in sorted_classes]

    # Create a bar chart
    plt.figure(figsize=(15, 10))
    bar_width = 0.4
    r1 = np.arange(len(sorted_classes))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, sorted_scores1, color='#2c3968', width=bar_width, label='TooT-Triplet')
    plt.bar(r2, sorted_scores2, color='#ff6f61', width=bar_width, label='Frozen model')

    # Adding labels and title
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.xticks([r + bar_width / 2 for r in range(len(sorted_class_names))], sorted_class_names, rotation=90, fontsize=9)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Adjust layout to ensure labels are fully visible
    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()


def plot_stacked_bar_chart_withlines(class_silhouette_scores1, class_silhouette_scores2, label_dict, class_sizes, output_svg='Results/silhouette_scores_by_class.svg'):
    """
    Plot a stacked bar chart of average silhouette scores for each class from two different sets and save it as an SVG file.

    Parameters:
    class_silhouette_scores1 (dict): Dictionary of silhouette scores for each class (first set).
    class_silhouette_scores2 (dict): Dictionary of silhouette scores for each class (second set).
    label_dict (dict): Dictionary mapping class labels to class names.
    class_sizes (dict): Dictionary mapping class labels to class sizes.
    output_svg (str): Path to the output SVG file.
    """

    # Calculate the average silhouette score for each class
    avg_scores1 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores1.items()}
    avg_scores2 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores2.items()}
    classes = list(avg_scores1.keys())
    average_scores1 = list(avg_scores1.values())
    average_scores2 = list(avg_scores2.values())
    sizes = [class_sizes[cls] for cls in classes]

    # Sort by class sizes (largest to smallest)
    sorted_indices = np.argsort(sizes)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_scores1 = [average_scores1[i] for i in sorted_indices]
    sorted_scores2 = [average_scores2[i] for i in sorted_indices]
    sorted_class_names = [label_dict[cls] for cls in sorted_classes]

    # Define color mappings for specific labels
    green_labels = ['iron(3+)', 'maltodextrin', 'L-tryptophan zwitterion', 'glycine betaine', 'arsenate ion', 'guanine', 'xanthine', 'gluconate', 'boric acid']
    orange_labels = ['cobalamin', 'maltose', 'folate(2-)']
    red_labels = ['tungstate']


    # Create a bar chart
    plt.figure(figsize=(15, 10))
    bar_width = 0.4
    r1 = np.arange(len(sorted_classes))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, sorted_scores1, color='#2c3968', width=bar_width, label='TooT-Triplet, Overall = -0.111')
    plt.bar(r2, sorted_scores2, color='#ff6f61', width=bar_width, label='Baseline 1, Overall = -0.154')

    # Adding labels and title
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.xticks([r + bar_width / 2 for r in range(len(sorted_class_names))], sorted_class_names, rotation=90, fontsize=9)

    # Apply colors to specific labels
    for i, name in enumerate(sorted_class_names):
        arrow_color = None
        if name in green_labels:
            plt.gca().get_xticklabels()[i].set_color('green')
            arrow_color = 'green'
        elif name in orange_labels:
            plt.gca().get_xticklabels()[i].set_color('orange')
            arrow_color = 'orange'
        elif name in red_labels:
            plt.gca().get_xticklabels()[i].set_color('red')
            arrow_color = 'red'
        
        if arrow_color:
            # Position the arrow above the end of each bar
            score = sorted_scores1[i] if i < len(r1) else sorted_scores2[i - len(r1)]
            x_position = r1[i] if i < len(r1) else r2[i - len(r1)]
            plt.annotate('', xy=(x_position, score), xytext=(x_position, score + 0.05),
                         arrowprops=dict(facecolor=arrow_color, edgecolor=arrow_color, width=2, headwidth=10))

    plt.ylim(-1, 1)
    plt.legend(loc='upper left', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Add vertical lines to separate class size categories
    more_than_100 = np.sum(np.array(sizes) > 100)
    more_than_10 = np.sum(np.array(sizes) > 10)
    more_than_3 = np.sum(np.array(sizes) > 3)

    plt.axvline(more_than_100 - 0.5, color='black', linewidth=1, linestyle='--')
    plt.axvline(more_than_10 - 0.5, color='black', linewidth=1, linestyle='--')
    plt.axvline(more_than_3 - 0.5, color='black', linewidth=1, linestyle='--')

    # Adjust layout to ensure labels are fully visible
    plt.tight_layout()
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()

def plot_stacked_bar_chart3(class_silhouette_scores1, class_silhouette_scores2, class_silhouette_scores3, label_dict, class_sizes, output_svg='Results/silhouette_scores_by_class.svg'):
    """
    Plot a stacked bar chart of average silhouette scores for each class from three different sets and save it as an SVG file.

    Parameters:
    class_silhouette_scores1 (dict): Dictionary of silhouette scores for each class (first set).
    class_silhouette_scores2 (dict): Dictionary of silhouette scores for each class (second set).
    class_silhouette_scores3 (dict): Dictionary of silhouette scores for each class (third set).
    label_dict (dict): Dictionary mapping class labels to class names.
    class_sizes (dict): Dictionary mapping class labels to class sizes.
    output_svg (str): Path to the output SVG file.
    """

    # Calculate the average silhouette score for each class
    avg_scores1 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores1.items()}
    avg_scores2 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores2.items()}
    avg_scores3 = {cls: np.mean(scores) for cls, scores in class_silhouette_scores3.items()}
    classes = list(avg_scores1.keys())
    average_scores1 = list(avg_scores1.values())
    average_scores2 = list(avg_scores2.values())
    average_scores3 = list(avg_scores3.values())
    sizes = [class_sizes[cls] for cls in classes]

    # Sort by class sizes (largest to smallest)
    sorted_indices = np.argsort(sizes)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_scores1 = [average_scores1[i] for i in sorted_indices]
    sorted_scores2 = [average_scores2[i] for i in sorted_indices]
    sorted_scores3 = [average_scores3[i] for i in sorted_indices]
    sorted_class_names = [label_dict[cls] for cls in sorted_classes]

    # Create a bar chart
    plt.figure(figsize=(15, 10))
    bar_width = 0.3
    r1 = np.arange(len(sorted_classes))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, sorted_scores1, color='#1f77b4', width=bar_width, label='TooT-Triplet(offline)')
    plt.bar(r2, sorted_scores2, color='#ff7f0e', width=bar_width, label='TooT-Triplet(online)')
    plt.bar(r3, sorted_scores3, color='#2ca02c', width=bar_width, label='Frozen-KNN')

    # Adding labels and title
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.xticks([r + bar_width for r in range(len(sorted_class_names))], sorted_class_names, rotation=90, fontsize=9)
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # Adjust layout to ensure labels are fully visible
    plt.tight_layout()

    # Save the plot as an SVG file
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()

def overall_bar_chart_compare():
    # Define file names and corresponding metrics
    models = ["frozen", "offline", "online"]
    distance_metrics = ["manhattan", "euclidean", "cosine"]
    
    # Initialize empty lists to store metrics and scores
    metrics = []
    scores = []

    # Read the scores from the respective files
    for model in models:
        for metric in distance_metrics:
            if model == "online" and metric != "manhattan":
                continue  # Skip non-manhattan distance metrics for online model
            file_name = f"Results/silhoutte/{model}_{metric}"
            try:
                with open(file_name, 'r') as file:
                    score = float(file.read().strip())
                    scores.append(round(score, 3))
                    metrics.append(f"{model.capitalize()} - {metric.capitalize()}")
            except FileNotFoundError:
                print(f"File {file_name} not found. Skipping.")
    
    # Define colors for each category
    color_map = {
        'Frozen': '#2ca02c',   
        'Offline': '#1f77b4',  
        'Online': '#ff7f0e'    
    }
    
    # Assign colors based on the model
    colors = []
    for metric in metrics:
        if 'Frozen' in metric:
            colors.append(color_map['Frozen'])
        elif 'Offline' in metric:
            colors.append(color_map['Offline'])
        elif 'Online' in metric:
            colors.append(color_map['Online'])

    # Create the bar chart with narrower bars
    bar_width = 0.5
    bars = plt.bar(metrics, scores, color=colors, width=bar_width)

    # Add numbers outside the bars
    for bar, score in zip(bars, scores):
    # Place the text labels slightly above the bar
        plt.text(
        bar.get_x() + bar.get_width() / 2,  # x position at the center of the bar
        score - 0.03,  # y position just above the bar, adjust 0.01 as needed
        f'{score:.3f}',
        ha='center',  # horizontal alignment at the center
        va='bottom',  # vertical alignment below the text
        fontsize=10,
        color='black')
    # Add grid lines in the background
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Set labels and title
    plt.ylabel('Silhouette Score')
    #plt.title('Silhouette Scores for Different Distance Metrics')
    plt.ylim(-0.5, None) 
    # Create custom legend handles
    legend_handles = [
        mpatches.Patch(color=color_map['Frozen'], label='Frozen model'),
        mpatches.Patch(color=color_map['Offline'], label='TooT-Triplet (Offline)'),
        mpatches.Patch(color=color_map['Online'], label='TooT-Triplet (Online)')
    ]

    # Add legend to the plot in the bottom right corner
    plt.legend(handles=legend_handles, loc='lower left')

    # Save the plot as an SVG file
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    output_svg = 'Results/silhouette_bar_comparison.svg'
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()


#overall_bar_chart_compare()


#overall_bar_chart_compare()
# Assuming silhouette_score_per_class and load_label_names are already defined
X1 = train_emb
y1 = y_train


distance_metric = 'euclidean'
model_name = 'offline'
#score1 = silhouette_score(X1, y1, metric =distance_metric)
#print('score :'+ model_name+' '+distance_metric+' '+str(score1))
#with open(f'Results/silhoutte/{model_name}_{distance_metric}','w') as f:
#     f.write(str(score1))


X2 = frozen_emb
y2 = y_frozen

file_path_train = 'train_emb_online.txt'

# Read the embeddings
embeddings = []
with open(file_path_train, 'r') as file:
    for line in file:
        # Convert the line to a list of floats
        embedding = list(map(float, line.strip().split()))
        embeddings.append(embedding)

'''
file_path_test = 'val_emb_online.txt'
with open(file_path_test, 'r') as file:
    for line in file:
        # Convert the line to a list of floats
        embedding = list(map(float, line.strip().split()))
        embeddings.append(embedding)

# Convert list to a numpy array for easier manipulation
X3 = np.array(embeddings)
y3= y2
'''

class_sizes = {}
for index, label in enumerate(y_train):
    if label not in class_sizes:
        class_sizes[label] = 0
    class_sizes[label] += 1
label_dict_path = 'Dataset/Label_name_list_transporter_uni_ident100_t3'  
label_dict = load_label_names(label_dict_path)
class_silhouette_scores1 = silhouette_score_per_class(X1, y1)
class_silhouette_scores2 = silhouette_score_per_class(X2, y2)
#class_silhouette_scores3 = silhouette_score_per_class(X3, y3)
plot_stacked_bar_chart_withlines(class_silhouette_scores1, class_silhouette_scores2, label_dict, class_sizes)

#plot_bar_chart(class_silhouette_scores, label_dict, class_sizes)

'''

distance_metric = 'manhattan'
model_name = 'online'
score1 = silhouette_score(X3, y3, metric =distance_metric)
print('score :'+ model_name+' '+distance_metric+' '+str(score1))
with open(f'Results/silhoutte/{model_name}_{distance_metric}','w') as f:
     f.write(str(score1))


distance_metric = 'manhattan'
model_name = 'frozen'
score2 = silhouette_score(X2, y2, metric =distance_metric)
print('score :'+ model_name+' '+distance_metric+' '+str(score1))
with open(f'Results/silhoutte/{model_name}_{distance_metric}','w') as f:
     f.write(str(score2))

distance_metric = 'euclidean'
model_name = 'frozen'
score2 = silhouette_score(X2, y2, metric =distance_metric)
print('score :'+ model_name+' '+distance_metric+' '+str(score1))
with open(f'Results/silhoutte/{model_name}_{distance_metric}','w') as f:
     f.write(str(score2))

distance_metric = 'cosine'
model_name = 'frozen'
score3 = silhouette_score(X2, y2, metric =distance_metric)
print('score :'+ model_name+' '+distance_metric+' '+str(score1))
with open(f'Results/silhoutte/{model_name}_{distance_metric}','w') as f:
     f.write(str(score3))
'''
#score2 = silhouette_score(X2, y2)
#print('score frozen:'+str(score2))


#score3 = silhouette_score(X3, y3)
#print('score online:'+str(score3))




