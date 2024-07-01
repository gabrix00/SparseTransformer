import pandas as pd
import os
import numpy as np

#finetuned model (30 epochs)
path1 = os.getcwd()+'/results_test/mlm_task/bert-base-uncased_inference_mnli_2024-07-01_09-59-49/results_dataset_mnli_bert_mlm_task.csv' 
# Bert without finetuning
path2 = os.getcwd()+'/results_test/mlm_task/bert-base-uncased_inference_mnli_2024-07-01_10-30-10/results_dataset_mnli_bert_mlm_task.csv'
#CustomBert without finetuning 
path3 = os.getcwd()+'/results_test/mlm_task/bert-base-uncased_inference_mnli_2024-07-01_11-05-06/results_dataset_mnli_custom_bert_mlm_task.csv'
#CustomBert finetuned (40 epochs)
path4 = os.getcwd()+'/results_test/mlm_task/bert-base-uncased_inference_mnli_2024-07-01_22-06-05/results_dataset_mnli_custom_bert_mlm_task.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)

print('\n')

print(f'average cross entropy loss of Bert MLM (30 epochs) is: {np.round(df1.loss.sum()/df1.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df1[df1["pred"]==df1["label"]].shape[0]/ df1.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM (no finetuning) is: {np.round(df2.loss.sum()/df2.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df2[df2["pred"]==df2["label"]].shape[0]/ df2.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom CustomBert MLM (no finetuning) is: {np.round(df3.loss.sum()/df3.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df3[df3["pred"]==df3["label"]].shape[0]/ df3.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert (40 epochs) MLM is: {np.round(df4.loss.sum()/df4.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df4[df4["pred"]==df4["label"]].shape[0]/ df4.shape[0],3)}')
print('\n\n\n')