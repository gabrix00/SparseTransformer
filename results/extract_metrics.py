import pandas as pd
import os
import numpy as np


path1 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-17_18-11-32/results_dataset_mnli_bert_mlm_task.csv'
path2 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-17_18-24-50/results_dataset_mnli_custom_bert_mlm_task.csv'


df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

print(f'average cross entropy loss of Bert MLM is: {np.round(df1.loss.sum()/df1.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df1[df1["pred"]==df1["label"]].shape[0]/ df1.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM is: {np.round(df2.loss.sum()/df2.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df2[df2["pred"]==df2["label"]].shape[0]/ df2.shape[0],3)}')
print('\n\n\n')