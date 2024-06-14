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
from masking_process import masking, rmasking, mask_multiple_sentences

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

gm_dir = os.path.join(os.getcwd(),'experiments', 'class_task', 'gabriel_mask')
rgm_dir = os.path.join(os.getcwd(),'experiments','class_task','random_gabriel_mask') #seed 42

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
        text_a = self.texts_a[index]
        text_b = self.texts_b[index]
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
        gabriel_mask = mask_multiple_sentences(text_a,text_b)

        ncells = np.count_nonzero(gabriel_mask)
        dim = gabriel_mask.shape[0] #è sempre quadrata
        

        random_ones_matrix = np.zeros((dim,dim))
        random_indices = np.random.choice(dim * dim, ncells, replace=False)
        random_ones_matrix.flat[random_indices] = 1
        random_ones_matrix = random_ones_matrix.reshape((dim,dim))
        random_gabriel_mask = torch.from_numpy(random_ones_matrix)

        if mask.count(0):
            n = self.max_len
            gabriel_mask = torch.cat([gabriel_mask, torch.zeros(n-dim, dim)], dim=0) #concat along rows
            gabriel_mask = torch.cat([gabriel_mask, torch.zeros(n, n-dim)], dim=1) #concat along columns

            random_gabriel_mask = torch.cat([random_gabriel_mask, torch.zeros(n-dim, dim)], dim=0) #concat along rows
            random_gabriel_mask = torch.cat([random_gabriel_mask, torch.zeros(n, n-dim)], dim=1)

        #compatibility dimension of gabriel mask and masking( in case of truncation)
        elif mask.count(1) < gabriel_mask.shape[0]:
            n = self.max_len
            gabriel_mask = gabriel_mask[:n]
            gabriel_mask = gabriel_mask[:, :n]

            random_gabriel_mask = random_gabriel_mask[:n]
            random_gabriel_mask = random_gabriel_mask[:, :n]

        return {
                "gabriel_mask": torch.tensor(gabriel_mask, dtype=torch.long),
                "random_gabriel_mask": torch.tensor(random_gabriel_mask, dtype=torch.long),
                "text_a": text_a,
                "text_b": text_b,
                "text_concat":text_concat,
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

    if not os.path.exists(gm_dir):
        os.makedirs(gm_dir)
        print(f"La directory: '{gm_dir}' è stata creata con successo.")
    else:
        print(f"La directory: '{gm_dir}' esiste già.")

    if not os.path.exists(rgm_dir):
        os.makedirs(rgm_dir)
        print(f"La directory: '{rgm_dir}' è stata creata con successo.")
    else:
        print(f"La directory: '{rgm_dir}' esiste già.")


    dataset = load_dataset("LysandreJik/glue-mnli-train")
    dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)
    sliced_train_dataset = DatasetDict(dataset["train"][:])

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
        print(id)
        print(batch['text_concat'][0])
        print(len(batch['text_concat'][0]))
        print('\n\n')

        #if f'gm_{id}.npy' not in os.listdir(gm_dir):
        #    gabriel_mask = batch["gabriel_mask"]
        #    np.save(os.path.join(gm_dir, f'gm_{id}.npy'), gabriel_mask)
        
        #if f'rgm_{id}.npy' not in os.listdir(rgm_dir):
        #    random_gabriel_mask =  batch["random_gabriel_mask"]
        #    np.save(os.path.join(rgm_dir,f'rgm_{id}.npy'), random_gabriel_mask)

        progress_bar.update(1)
        
if __name__ =='__main__':
    main()