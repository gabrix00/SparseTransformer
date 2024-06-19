import pandas as pd
import os
import numpy as np


path1 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-17_18-11-32/results_dataset_mnli_bert_mlm_task.csv'
path2 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-17_18-24-50/results_dataset_mnli_custom_bert_mlm_task.csv'
path3 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-19_19-01-59/results_dataset_mnli_custom_bert_mlm_task_rgm.csv'

path4 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-19_19-44-23/results_dataset_mnli_custom_bert_mlm_task.csv'
path5 = os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-19_20-15-08/results_dataset_mnli_custom_bert_mlm_task_rgm.csv'
path6= os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-19_20-29-25/results_dataset_mnli_custom_bert_mlm_task_rgm.csv'
path7= os.getcwd()+'/results/bert-base-uncased_inference_mnli_2024-06-19_21-15-58/results_dataset_mnli_custom_bert_mlm_task.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)

df4 = pd.read_csv(path4)
df5 = pd.read_csv(path5)
df6 = pd.read_csv(path6)
df7 = pd.read_csv(path7)


print(f'average cross entropy loss of Bert MLM is: {np.round(df1.loss.sum()/df1.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df1[df1["pred"]==df1["label"]].shape[0]/ df1.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM is: {np.round(df2.loss.sum()/df2.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df2[df2["pred"]==df2["label"]].shape[0]/ df2.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM using RGM is: {np.round(df3.loss.sum()/df3.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df3[df3["pred"]==df3["label"]].shape[0]/ df3.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM with change in Atte is: {np.round(df4.loss.sum()/df4.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df4[df4["pred"]==df4["label"]].shape[0]/ df4.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM with change in Atte and using RGM is: {np.round(df5.loss.sum()/df5.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df5[df5["pred"]==df5["label"]].shape[0]/ df5.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM with change in Atte and using RGM using Drop outis: {np.round(df6.loss.sum()/df6.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df6[df6["pred"]==df6["label"]].shape[0]/ df6.shape[0],3)}')
print('\n\n\n')

print(f'average cross entropy loss of Custom Bert MLM with change in Atte and NO dropout is: {np.round(df7.loss.sum()/df7.shape[0],3)}')
print(f'NUMBER OF EXACT MATCH WORDS: {np.round(df7[df4["pred"]==df7["label"]].shape[0]/ df7.shape[0],3)}')
print('\n\n\n')
