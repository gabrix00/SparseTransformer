import torch
from transformers import AutoTokenizer,BertForSequenceClassification
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
    def __init__(self, texts_a, texts_b, labels, idx, tokenizer, max_len):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.idx = idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, index):
        text_a = normalizzation(self.texts_a[index])
        text_b = normalizzation(self.texts_b[index])
        label = self.labels[index]
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


        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            'text_concat':text_concat, # solo in inferenza per check finale
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
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

    model_name = pretrained_model + "_" + task + "_" + dataset_name + "_" + pst_now.strftime("%Y-%m-%d_%H-%M-%S")
    saved_model_dir = os.path.join(os.getcwd(),'results_test','class_task', model_name)
    os.makedirs(saved_model_dir, exist_ok=True)

   
    batch_size = 1


    dataset = load_dataset("LysandreJik/glue-mnli-train")
    dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)

    sliced_test_dataset = pd.DataFrame(dataset["validation"])  
    
    
    num_labels = 3   
    
    model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                          num_labels=num_labels,  # MNLI has 3 labels: 'entailment', 'neutral', 'contradiction'
                                                          problem_type="multi_label_classification",
                                                          ignore_mismatched_sizes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        print('model loaded succesfully')
    

    
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

 
    test_dataset = Dataset(texts_a = sliced_test_dataset['premise'],#filtered_dataset["train"]['premise'], #old
                            texts_b = sliced_test_dataset['hypothesis'],
                            labels = sliced_test_dataset['label'],
                            idx = sliced_test_dataset['idx'],
                            tokenizer = tokenizer,
                            max_len = 100)
    
    test_data_loader = DataLoader(test_dataset,
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
    for bi, batch in enumerate(test_data_loader):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch['label'].to(device)

        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_labels).float().to(device)

        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=mask, labels=labels_one_hot)

        loss = output.loss 

        pred_class = torch.argmax(output.logits, dim =1)


        # Append results to lists
        text_analyzed_list.extend(batch['text_concat'])
        loss_list.append(loss.item())
        pred_list.append(pred_class.item())
        labels_list.append(batch['label'].item())
        
        progress_bar.update(1)
        


        # Save results to a CSV file
    result = pd.DataFrame({
    'sentence': text_analyzed_list,
    'pred': pred_list,
    'label': labels_list,
    'loss': loss_list,
    })
    
    result.to_csv(os.path.join(saved_model_dir, 'results_dataset_mnli_bert4seqclass.csv'), index=False)

if __name__ == '__main__':
    checkpoint_path = '/Users/gabrieletuccio/Developer/GitHub/SparseTransformer/finetuned_model/bert4seqclass_finetuning_results/checkpoint.pt'
    main(checkpoint_path=checkpoint_path)