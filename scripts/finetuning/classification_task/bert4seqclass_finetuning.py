import torch
from transformers import AutoTokenizer,BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm 
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
import os
import datetime, pytz
import pandas as pd
from matplotlib import pyplot as plt

#DA CLASS TASK
import sys
sys.path.append(os.path.join(os.getcwd(),'scripts'))
from normalization import normalizzation


class Dataset:
    def __init__(self, texts_a, texts_b, labels, tokenizer, max_len):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, index):
        text_a = normalizzation(self.texts_a[index])
        text_b = normalizzation(self.texts_b[index])
        label = self.labels[index]
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
            "ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }

def training_bert_model_4_mlm_task(model, optimizer, hyper_params, dataloaders, num_labels, saved_dir, start_epochs):

    optimizer = AdamW(model.parameters(), lr=hyper_params['learning_rate'])

    best_model_wts = model.state_dict() #si salva i pesi del modello al t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_loss = {"train": float('inf'), "val": float('inf')} 
    history_loss = {"train": [], "val": []}
    history_accuracy = {"train": [], "val": []}

    epochs = hyper_params['epochs']

    total_epochs = epochs+start_epochs
    for epoch in range(start_epochs,total_epochs):
        print(f'Epoch {epoch}/{total_epochs - 1}')
        print('-' * 50)

        # Initialize epoch variables
        sum_loss = {"train": 0, "val": 0}
        correct_preds = {"train": 0, "val": 0}  
        total_samples = {"train": 0, "val": 0} 

        
        # Process each split
        for split in ["train", "val"]:
            if split == "train":
                model.train()
            else:
                model.eval()

            # Process each batch
            for batch in dataloaders[split]:
                ids = batch["ids"].to(device)
                mask = batch["mask"].to(device)
                labels = batch['label'].to(device)

                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_labels).float().to(device)
                
                optimizer.zero_grad()

                output = model(input_ids=ids, attention_mask=mask, labels=labels_one_hot)

                loss = output.loss 

                sum_loss[split] += loss.item() #batch loss
 
                if split == "train":
                    # Compute gradients
                    loss.backward()
                    # Optimize
                    optimizer.step()
                
                preds = torch.argmax(output.logits, dim =1)
                
                # Compute batch accuracy
                correct_preds[split] += (preds == labels).sum().item()
                total_samples[split] += labels.size(0)  #labels è solo per contare quanti elemnti nel batch ci sono


        # Compute epoch loss/accuracy
        epoch_loss = {split: sum_loss[split] / len(dataloaders[split]) for split in ["train", "val"]} #len(dataloaders[split]) gives you the number of batches in that DataLoader.
        epoch_accuracy = {split: correct_preds[split] / total_samples[split] for split in ["train", "val"]}

        # Update history
        for split in ["train", "val"]:
            history_loss[split].append(epoch_loss[split])
            history_accuracy[split].append(epoch_accuracy[split])

        if epoch_loss['train'] < best_loss['train']:  # Compare train loss with best train loss
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

        print(f"Train Loss: {epoch_loss['train']:.4f}, Train Acc: {epoch_accuracy['train']:.4f}")
        print(f"Val Loss: {epoch_loss['val']:.4f}, Val Acc: {epoch_accuracy['val']:.4f}")


        model.load_state_dict(best_model_wts) #ad ogni epoca si salva comunque i pesi migliori


    # Plot and saving loss
    plt.figure(figsize=(10, 6))  
    plt.title("Loss")
    for split in ["train", "val"]:
        plt.plot(history_loss[split], label=split)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(),"loss_plot.png"), dpi=300, bbox_inches="tight")  #
    plt.close()  

    # Plot and saving accuracy
    plt.figure(figsize=(10, 6))
    plt.title("Accuracy")
    for split in ["train", "val"]:
        plt.plot(history_accuracy[split], label=split)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.getcwd(),"accuracy_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()  

    model_path = os.path.join(saved_dir,'checkpoint.pt')
    torch.save({
        'epoch': epoch+start_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss['train']},  model_path)

    #torch.save(model.state_dict(),model_path)       

    return 1

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

   
    dataset_name = "mnli"
    pretrained_model = "bert-base-uncased"
    task ='classification_task'

    model_name = pretrained_model + "_" + pst_now.strftime("%Y-%m-%d_%H-%M-%S")
    saved_model_dir = os.path.join(os.getcwd(),'scripts','finetuning',task,'results', model_name)
    os.makedirs(saved_model_dir, exist_ok=True)

    # Model parameters
    hyper_params = {'epochs' : 5, 'batch_size' : 64, 'learning_rate' : 5e-5}

    num_labels = 3    
    model = BertForSequenceClassification.from_pretrained(pretrained_model,
                                                          num_labels=num_labels,  # MNLI has 3 labels: 'entailment', 'neutral', 'contradiction'
                                                          problem_type="multi_label_classification",
                                                          ignore_mismatched_sizes=True) 
    

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    optimizer = AdamW(model.parameters(), lr=hyper_params['learning_rate']) #di feafult

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
        loss = checkpoint['loss']
        print(f'avg loss checkpoint: {loss}')
        start_epoch = checkpoint['epoch'] + 1
        print(f'start_epoch: {start_epoch}')
    else:
        start_epoch = 0



    dataset = load_dataset("LysandreJik/glue-mnli-train")
    dataset = dataset.filter(lambda example: len(example["premise"]) <= 120 and len(example["hypothesis"]) <= 120)
    #print(dataset)

    sliced_train_dataset = pd.DataFrame(dataset["train"][:])

    train_df, val_df = train_test_split(sliced_train_dataset, test_size=0.2, random_state=seed)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    
    train_dataset = Dataset(texts_a = train_df['premise'],#filtered_dataset["train"]['premise'], #old
                            texts_b = train_df['hypothesis'],
                            labels = train_df['label'],
                            tokenizer = tokenizer,
                            max_len = 100)
    
    validation_dataset = Dataset(texts_a = val_df['premise'],#filtered_dataset["train"]['premise'], #old
                                 texts_b = val_df['hypothesis'],
                                 labels = val_df['label'],
                                 tokenizer = tokenizer,
                                 max_len = 100)
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=hyper_params['batch_size'],
                                   shuffle = True,
                                   pin_memory=True)  # Transfer tensors to CUDA in a more efficient way
    
    train_data_loader = DataLoader(validation_dataset,
                                   batch_size=hyper_params['batch_size'],
                                   shuffle = True,
                                   pin_memory=True)
    

    dataloaders = {'train': train_data_loader,'val': train_data_loader}

    checkpoint = training_bert_model_4_mlm_task(model=model,optimizer=optimizer,
                                   hyper_params=hyper_params,
                                   dataloaders=dataloaders,
                                   num_labels=num_labels, saved_dir=saved_model_dir,start_epochs=start_epoch)

    
       




if __name__ == '__main__':
    #checkpoint_path = "..."" # Change this to the path of your checkpoint
    #checkpoint_path = os.path.join(os.getcwd(),'scripts/finetuning/mlm_task/results/bert-base-uncased_2024-06-26_16-25-02/checkpoint.pt')
    #main(checkpoint_path)
    main()