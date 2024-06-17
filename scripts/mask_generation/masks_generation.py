import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import time
from tqdm.auto import tqdm
from datasets import DatasetDict
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.getcwd(),'scripts'))
from masking_process import  rmasking, mask_multiple_sentences
from normalization import normalizzation

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

# Definizione dei percorsi delle directory
training_root_class = os.path.join(os.getcwd(), 'experiments', 'training', 'class_task')
training_root_mlm = os.path.join(os.getcwd(), 'experiments', 'training', 'mlm_task')

training_root_class_files = os.path.join(training_root_class,'files')
training_root_mlm_files = os.path.join(training_root_mlm,'files')
#--------

gm_class_dir = os.path.join(training_root_class, 'gabriel_mask')
rgm_class_dir = os.path.join(training_root_class, 'random_gabriel_mask')  

gm_mlm_dir = os.path.join(training_root_mlm, 'gabriel_mask')
rgm_mlm_dir = os.path.join(training_root_mlm, 'random_gabriel_mask')  

gm_class_img_dir = os.path.join(training_root_class_files,'img','gabriel_mask')
rgm_class_img_dir = os.path.join(training_root_class_files,'img','random_gabriel_mask')  

text_class_dir = os.path.join(training_root_class_files,'txt')

gm_mlm_img_dir = os.path.join(training_root_mlm_files,'img','gabriel_mask')
rgm_mlm_img_dir = os.path.join(training_root_mlm_files,'img','random_gabriel_mask') 

text_mlm_dir = os.path.join(training_root_mlm_files,'txt')


def compute_attention(mask:torch.Tensor):
    ncells = np.count_nonzero(mask)
    dim = mask.shape[0]
    return ncells/(dim*dim)

def consistency_check(mask_before, mask_after):
    """
    Controlla se ci sono stati errori nella trasformazione della Gabriel mask.
    Lo fa guardando al numero di 1 in ogni riga di mask_before e mask_after. 
    Il numero di 1 deve essere lo stesso in entrambe le matrici per ogni riga di mask_before.
    
    Args:
        mask_before (torch.Tensor): Matrice di input iniziale.
        mask_after (torch.Tensor): Matrice di output dopo la trasformazione (deve essere 100x100).
    """
    # Converti i tensori in array numpy per una più facile manipolazione
    mask_before_np = mask_before.numpy()
    mask_after_np = mask_after.numpy()
    
    # Controlla il numero di 1 in ogni riga delle due matrici
    for i in range(mask_before_np.shape[0]):
        count_before = np.sum(mask_before_np[i] == 1)
        count_after = np.sum(mask_after_np[i] == 1)

        #print(f"Riga {i}: numero di 1 in mask_before = {count_before}, numero di 1 in mask_after = {count_after}")
        
        if count_before != count_after:
            print(f"Incoerenza trovata nella riga {i}: numero di 1 non corrispondente")
            return False

    return True

def process_mask(mask,gabriel_mask,max_len):
    ncells = np.count_nonzero(gabriel_mask)
    dim = gabriel_mask.shape[0] 

    random_ones_matrix = np.zeros((dim,dim))
    random_indices = np.random.choice(dim * dim, ncells, replace=False)
    random_ones_matrix.flat[random_indices] = 1
    random_ones_matrix = random_ones_matrix.reshape((dim,dim))
    random_gabriel_mask = torch.from_numpy(random_ones_matrix)

    if mask.count(0):
        n = max_len
        gabriel_mask = torch.cat([gabriel_mask, torch.zeros(n-dim, dim)], dim=0) #concat along rows
        gabriel_mask = torch.cat([gabriel_mask, torch.zeros(n, n-dim)], dim=1) #concat along columns

        random_gabriel_mask = torch.cat([random_gabriel_mask, torch.zeros(n-dim, dim)], dim=0) #concat along rows
        random_gabriel_mask = torch.cat([random_gabriel_mask, torch.zeros(n, n-dim)], dim=1)

    #compatibility dimension of gabriel mask and masking( in case of truncation)
    elif mask.count(1) < gabriel_mask.shape[0]:
        n = max_len
        gabriel_mask = gabriel_mask[:n]
        gabriel_mask = gabriel_mask[:, :n]

        random_gabriel_mask = random_gabriel_mask[:n]
        random_gabriel_mask = random_gabriel_mask[:, :n]

    # Verifica che gabriel_mask e random_gabriel_mask abbiano dimensioni: 100,100
    if gabriel_mask.shape[0] != 100 or gabriel_mask.shape[1] != 100:
        print(f"gabriel_mask shape: {gabriel_mask.shape}")
        raise ValueError(f"gabriel_mask deve avere dimensioni {(max_len,max_len)}")
        
    
    if random_gabriel_mask.shape[0] != max_len or random_gabriel_mask.shape[1] != max_len:
        print(f"random_gabriel_mask shape: {random_gabriel_mask.shape}")
        raise ValueError(f"random_gabriel_mask deve avere dimensioni {(max_len,max_len)}")

    return gabriel_mask, random_gabriel_mask


class Dataset:
    def __init__(self, texts_a, texts_b, idx, tokenizer, max_len):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.idx = idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, index):
        text_a = normalizzation(self.texts_a[index])
        text_b = normalizzation(self.texts_b[index])
        id = self.idx[index]
        text_concat = f"{text_a} [SEP] {text_b}"  # Concatenate text_a and text_b with [SEP] token
        inputs = self.tokenizer.__call__(
            text_concat,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        mask = inputs["attention_mask"]

        #gabriel_mask_mlm is gabriel_mask_premise
        gabriel_mask_class,gabriel_mask_mlm = mask_multiple_sentences(text_a,text_b)

        #for viz
        rmasking(text=text_a,rand_mask=gabriel_mask_mlm,viz=True)
        rmasking(text=text_concat,rand_mask=gabriel_mask_class,viz=True)
        
        #if f'gm_{id}.npy' not in os.listdir(gm_class_img_dir): #too memory
        #    rmasking(text=text_concat,rand_mask=gabriel_mask,viz=True, path=os.path.join(gm_class_img_dir,f'gm_{id}')) # too memory
     
        attention_gm_class = compute_attention(gabriel_mask_class)
        attention_gm_mlm = compute_attention(gabriel_mask_mlm)
       
        gabriel_mask_class_processed,random_gabriel_mask_class_processed = process_mask(mask=mask,gabriel_mask=gabriel_mask_class,max_len=self.max_len)
        print(consistency_check(mask_before=gabriel_mask_class,mask_after=gabriel_mask_class_processed))

        gabriel_mask_mlm_processed,random_gabriel_mask_mlm_processed = process_mask(mask=mask,gabriel_mask=gabriel_mask_mlm,max_len=self.max_len)
        print(consistency_check(mask_before=gabriel_mask_mlm,mask_after=gabriel_mask_mlm_processed))
    

        return {
                "gabriel_mask_class": torch.tensor(gabriel_mask_class_processed, dtype=torch.long),
                "random_gabriel_mask_class": torch.tensor(random_gabriel_mask_class_processed, dtype=torch.long),
                "attention_gm_class":attention_gm_class,
                "text_concat":text_concat,

                "gabriel_mask_mlm": torch.tensor(gabriel_mask_mlm_processed, dtype=torch.long),
                "random_gabriel_mask_mlm": torch.tensor(random_gabriel_mask_mlm_processed, dtype=torch.long),
                "attention_gm_mlm":attention_gm_mlm,
                "text_a": text_a, #premise
                #"text_b": text_b, #hypotesis
                
                "id": id,
                

            }
#–----------------------------------------------------------------------------------------------

seed = 42

# https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = seed) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed()


def main():

    directories = [
        gm_class_dir,
        rgm_class_dir,
        gm_mlm_dir,
        rgm_mlm_dir,
        text_class_dir,
        text_mlm_dir,
        gm_class_img_dir,
        rgm_class_img_dir,
        gm_mlm_img_dir,
        rgm_mlm_img_dir,
        ]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"La directory: '{dir_path}' è stata creata con successo.")
        else:
            print(f"La directory: '{dir_path}' esiste già.")


    dataset = load_dataset("LysandreJik/glue-mnli-train")
    dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)
    #sliced_train_dataset = DatasetDict(dataset["train"][3952:3954])
    #sliced_train_dataset = DatasetDict(dataset["train"][3949:3954])
    sliced_train_dataset = DatasetDict(dataset["train"][:1000])

    #print(sliced_train_dataset)


    train_dataset = Dataset(texts_a=sliced_train_dataset['premise'],
                            texts_b=sliced_train_dataset['hypothesis'],
                            idx=sliced_train_dataset['idx'],
                            tokenizer=tokenizer,
                            max_len=100)


    train_data_loader = DataLoader(train_dataset, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    num_workers=0)
    
    progress_bar = tqdm(range(len(sliced_train_dataset['idx'])))

    


    for bi, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Loading:', disable=True):
        id = batch["id"][0].numpy()
        """
        print('FOR CLASS TASK')
        print(id)
        print(batch['text_concat'][0])
        print(f"attention is: {batch['attention_gm_class'][0]}")
        print('\n')
        print('FOR  MLM TASK')
        print(id)
        print(batch['text_a'][0])
        print(f"attention is: {batch['attention_gm_mlm'][0]}")
        print('---------------------------------')
        print('\n\n\n\n')
        """

        #FOR CLASSIFICATION TASK    

        if f'gm_{id}.npy' not in os.listdir(gm_class_dir):
            gabriel_mask_class = batch["gabriel_mask_class"]
            np.save(os.path.join(gm_class_dir, f'gm_{id}.npy'), gabriel_mask_class)
            with open(os.path.join(text_class_dir, f'gm_{id}.txt'), 'w') as f:
                f.write(f"text_concat: {batch['text_concat'][0]}\npercentage of attention: {batch['attention_gm_class'][0]}")

        
        if f'rgm_{id}.npy' not in os.listdir(rgm_class_dir):
            random_gabriel_mask_class =  batch["random_gabriel_mask_class"]
            np.save(os.path.join(rgm_class_dir,f'rgm_{id}.npy'), random_gabriel_mask_class)
        
        #FOR MLM TASK

        if f'gm_{id}.npy' not in os.listdir(gm_mlm_dir):
            gabriel_mask_mlm = batch["gabriel_mask_mlm"]
            np.save(os.path.join(gm_mlm_dir, f'gm_{id}.npy'), gabriel_mask_mlm)
            with open(os.path.join(text_mlm_dir, f'gm_{id}.txt'), 'w') as f:
                f.write(f"premise: {batch['text_a'][0]}\npercentage of attention: {batch['attention_gm_mlm'][0]}")

        
        if f'rgm_{id}.npy' not in os.listdir(rgm_mlm_dir):
            random_gabriel_mask_mlm =  batch["random_gabriel_mask_mlm"]
            np.save(os.path.join(rgm_mlm_dir,f'rgm_{id}.npy'), random_gabriel_mask_mlm)


        progress_bar.update(1)
        
if __name__ =='__main__':
    main()