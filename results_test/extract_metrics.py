import pandas as pd
import os
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


path1 = os.getcwd()+'/results_test/bert-base-uncased_inference_mnli_2024-06-29_17-54-04/results_dataset_mnli_bert4seqclass.csv' 

path2 = os.getcwd()+'/results_test/bert-base-uncased_inference_mnli_2024-06-29_18-40-49/results_dataset_mnli_custom_bert4seqclass.csv'


df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

# Compute the confusion matrix
cm = confusion_matrix(df1['label'], df1['pred'])
print(" Bert4SeqClass Confusion Matrix :")
print(cm)

# Compute the F1 score (weighted to handle multi-class)
f1 = f1_score(df1['label'], df1['pred'], average='weighted')
print("\n Bert4SeqClass F1 Score (weighted):")
print(f1)
# Compute the precision (weighted to handle multi-class)
precision = precision_score(df1['label'], df1['pred'], average='weighted')
print("\nPrecision (weighted):")
print(precision)

# Compute the recall (weighted to handle multi-class)
recall = recall_score(df1['label'], df1['pred'], average='weighted')
print("\nRecall (weighted):")
print(recall)


print('\n\n\n')
#--------------------------
cm = confusion_matrix(df2['label'], df2['pred'])
print(" CustomBertForSeqClass Confusion Matrix :")
print(cm)

# Compute the F1 score (weighted to handle multi-class)
f1 = f1_score(df2['label'], df2['pred'], average='weighted')
print("\n CustomBertForSeqClass F1 Score (weighted):")
print(f1)
# Compute the precision (weighted to handle multi-class)
precision = precision_score(df2['label'], df2['pred'], average='weighted')
print("\nPrecision (weighted):")
print(precision)

# Compute the recall (weighted to handle multi-class)
recall = recall_score(df2['label'], df2['pred'], average='weighted')
print("\nRecall (weighted):")
print(recall)