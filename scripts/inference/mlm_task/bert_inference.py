import torch
from transformers import AutoTokenizer,BertForMaskedLM
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np
from datasets import DatasetDict, load_dataset
import os
import datetime, pytz
import pandas as pd


import sys
sys.path.append(os.path.join(os.getcwd(),'scripts'))
from normalization import normalizzation


class Dataset:
    def __init__(self, texts_a, tokenizer, max_len):
        self.texts_a = texts_a
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, index):
        text_a = normalizzation(self.texts_a[index])
        inputs = self.tokenizer.__call__(
            text_a,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
            
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "text_a": text_a,
        } 


def main(checkpoint_path = None):
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set timezone
    timezone = pytz.timezone("Europe/Rome")
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("Europe/Rome"))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    task = 'inference'
    dataset_name = "mnli"
    pretrained_model = "bert-base-uncased"

    model_name = pretrained_model + "_" + task + "_"+ dataset_name + "_" + pst_now.strftime("%Y-%m-%d_%H-%M-%S")
    saved_model_dir = os.path.join(os.getcwd(), 'results_test','mlm_task', model_name)
    os.makedirs(saved_model_dir, exist_ok=True)
    

    # Model parameters
    batch_size = 1

    dataset = load_dataset("LysandreJik/glue-mnli-train")
    dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)

    sliced_test_dataset = pd.DataFrame(dataset["validation"]) 
    
    model = BertForMaskedLM.from_pretrained(pretrained_model)


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('model loaded succesfully')

    model.to(device)

 
    train_dataset = Dataset(texts_a= sliced_test_dataset['premise'],#filtered_dataset["train"]['premise'], #old
                            tokenizer=tokenizer,
                            max_len=100)
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle = False,
                                   pin_memory=True)  # Transfer tensors to CUDA in a more efficient way
    
    progress_bar = tqdm(range(len(sliced_test_dataset['idx'])))

    # Initialize lists for storing results
    text_analyzed_list = []
    loss_list = []
    pred_list = []
    labels_list = []

    model.eval()
    for bi, batch in enumerate(train_data_loader):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)

        input_ids_tensor = torch.tensor(tokenizer(batch['text_a'])['input_ids'][0]).to(device) #access the second list [[101,234,987,102]]


        if len(input_ids_tensor) > 3:
            mask_token_index = np.random.randint(1, len(input_ids_tensor) - 2)
        else:
            mask_token_index = 1  

        masked_word = tokenizer.convert_ids_to_tokens([input_ids_tensor[mask_token_index]])
        

        #da 1 a len-2 perchè evitiamo di macherare il token cls e sep.
        #il primo elmento della lista ha indice 0 e l'ultimo a indice len(n)-1. Quindi per togliere il cls devo andare a len(n)-2.

        temp_mask = torch.zeros(ids.size(-1)).to(device)
        temp_mask[mask_token_index] = 1
        masked_input = ids.masked_fill(temp_mask == 1, 103).to(device)
        labels = ids.masked_fill(masked_input != 103, -100).to(device)
        
        with torch.no_grad():
            output = model(input_ids=masked_input, attention_mask=mask, labels=labels)
        
        masked_token_probs = torch.softmax(output.logits[0, mask_token_index], dim=-1)
        
        # Recupera l'ID del token previsto con la probabilità massima
        predicted_token_id = torch.argmax(masked_token_probs, dim=-1)
        
         # Convert the predicted token id to the corresponding token
        predicted_token = tokenizer.decode([predicted_token_id])
        
        loss = output.loss 
        
        # Append results to lists
        text_analyzed_list.extend(batch['text_a'])
        loss_list.append(loss.item())
        pred_list.extend([predicted_token])
        labels_list.extend(masked_word)
        
        progress_bar.update(1)
        
        
    # Save results to a CSV file
    result = pd.DataFrame({
        'sentence': text_analyzed_list,
        'pred': pred_list,
        'label': labels_list,
        'loss': loss_list,
    })
    
    result.to_csv(os.path.join(saved_model_dir,'results_dataset_mnli_bert_mlm_task.csv'), index=False)

if __name__ == '__main__':
    checkpoint_path = '/Users/gabrieletuccio/Developer/GitHub/SparseTransformer/finetuned_model/bert_finetuning/checkpoint.pt'
    main(checkpoint_path=checkpoint_path)
    #main()