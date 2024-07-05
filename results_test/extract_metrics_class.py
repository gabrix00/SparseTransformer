from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns

path1 = os.getcwd()+'/results_test/class_task/bert-base-uncased_inference_mnli_2024-06-29_17-54-04/results_dataset_mnli_bert4seqclass.csv' 
path2 = os.getcwd()+'/results_test/class_task/bert-base-uncased_inference_mnli_2024-06-29_18-40-49/results_dataset_mnli_custom_bert4seqclass.csv'


df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

def mapp_labels(label):
    map_dict = {
        0: 'neutral',
        1: 'entailment',
        2: 'contradiction'
    }
    mapped_label = map_dict[label]
    return mapped_label

def plot_confusion_matrix(y_true, y_pred, model_name, path):
    # Map the numerical labels to their string equivalents
    unique_labels = sorted(set(y_true) | set(y_pred))
    mapped_labels = [mapp_labels(label) for label in unique_labels]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mapped_labels, yticklabels=mapped_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig(os.path.join(path, f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_old(y_true, y_pred, model_name, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(path,f'confusion_matrix_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_metrics_per_class(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f" {model_name} Confusion Matrix:")
    print(cm)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

    print("\nMetriche per classe:")
    for i in range(3):  # assumendo che ci siano 3 classi
        print(f"Classe {i}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1-score: {f1[i]:.4f}")

    # Calcolo delle metriche medie pesate
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print("\nMetriche medie pesate:")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")

    print('\n\n')

# Per il primo modello (Bert4SeqClass)
print_metrics_per_class(df1['label'], df1['pred'], "Bert4SeqClass")

# Per il secondo modello (CustomBertForSeqClass)
print_metrics_per_class(df2['label'], df2['pred'], "CustomBert4SeqClass")

# Plot per Bert4SeqClass
plot_confusion_matrix(df1['label'], df1['pred'], 'Bert4SeqClass',path='results_test/class_task/bert-base-uncased_inference_mnli_2024-06-29_17-54-04')

# Plot per CustomBertForSeqClass
plot_confusion_matrix(df2['label'], df2['pred'], 'CustomBert4SeqClass',path='results_test/class_task/bert-base-uncased_inference_mnli_2024-06-29_18-40-49')