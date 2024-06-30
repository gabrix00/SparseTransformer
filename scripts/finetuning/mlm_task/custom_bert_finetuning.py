import os
import datetime
import math
from typing import Optional, Tuple, Union, List
import pytz
from matplotlib import pyplot as plt


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AdamW, BertForMaskedLM, BertModel,
from transformers.activations import  ACT2FN
from transformers.modeling_outputs import  MaskedLMOutput                             
from torch.nn import CrossEntropyLoss


import sys
sys.path.append(os.path.join(os.getcwd(),'scripts'))
from masking_process import  masking
from normalization import normalizzation


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states) 
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    
class CustomformerForMaskedLM2(BertForMaskedLM):
    def __init__(self, config):
        super(CustomformerForMaskedLM2, self).__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self = CustomSelfAttention(config)
        

        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    """
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    
    )
    """

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        gabriel_mask :  Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict



        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self.set_gabriel_mask(gabriel_mask)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            #F.cross_entropy(logits.view(-1, tokenizer.vocab_size), ids.view(-1)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class CustomSelfAttention(nn.Module):
    """
    CustomBertSelfAttention created as an extentions of the original nn.Module class of pythorch
    in which we rewrite the code as in the BertSelfAttention in BertModeling but with the addition of new parameter: gabriel_mask
    which is used to modify the meccanism of a BertModel
    """
    def __init__(self, config, position_embedding_type=None):#gabriel_mask = None): 
        super(CustomSelfAttention,self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        #self.gabriel_mask = None

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def set_gabriel_mask(self,gabriel_mask):
        self.gabriel_mask = gabriel_mask
        
        

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) #permute the dim n1 with the dim n2. dim n0 and n3 remani the same

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        #gabriel_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        gabriel_mask= self.gabriel_mask


        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            # .cat concatente our data along a certain dim, in our eg dim=2, 
            # the tensor must have the same size along other dim expect for the dim where we concatenate
            #[1,2,3]
            #[5,2,3]
            #if we concatenate along the dim = 0
            #[6,2,3]
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2) 
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        #print(f'dim of attention_scores at the beginning is {attention_scores.shape}') #debug

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print(attention_scores)

       
        #logger.info('dim of attention score is :' +str(attention_scores.shape))


        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        #print(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        #print(attention_probs)
        
        if gabriel_mask is not None:
            #logger.info('dim of gabriel_mask is :' + str(gabriel_mask.shape))
            #logger.info('dim of gabriel_mask is :' + str(len(gabriel_mask.shape)))
            if len(gabriel_mask.shape) == 3: #cioè è gia un tensore (1,100,100) come nel caso dei .npy che ho slavato
                gabriel_mask.unsqueeze_(1)  # rimuovi singleton dimension along axis 1 così da avere (batch, )
                #logger.info('new modify shape of gabriel_mask is :' + str(gabriel_mask.shape))
                #gabriel_mask.squeeze_(dim=1)
                #Squeeze the first two dimensions inplace
                #gabriel_mask.squeeze_(dim=(0, 1))
                #
            #else:
                #logger.info(f'gabriel_mask {gabriel_mask.shape}')
                #logger.info('dim of gabriel_mask is :' + str(gabriel_mask.shape))


            attention_probs = torch.mul(attention_probs, gabriel_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        #attention_probs = self.dropout(attention_probs)
        #print(attention_probs)
        
        

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

#-----------------------------------------------------------------------------------------------------------------

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
    def __init__(self, texts_a, idx, tokenizer, max_len):
        self.texts_a = texts_a
        self.idx = idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, index):
        text_a = normalizzation(self.texts_a[index])
        id = self.idx[index]
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


        #gabriel_mask_path = os.path.join('/kaggle/input/gabriel-mask/gabriel_mask', f'gm_{id}.npy') #on kaggle
        gabriel_mask_path = os.path.join('experiments/training/mlm_task/gabriel_mask', f'gm_{id}.npy') #on kaggle
        
        try:
            gabriel_mask = np.load(gabriel_mask_path)
            print(f'gm_{id}.npy founded!')
        except:
            print(f'gm_{id}.npy NOT founded!')
            gabriel_mask,_ = process_mask(mask=mask,gabriel_mask=masking(text_a,viz=False),max_len=self.max_len)
            print(f'gm_{id}.npy generated!')
            
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "gabriel_mask": torch.tensor(gabriel_mask, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
        }


def training_bert_model_4_mlm_task(model, optimizer, hyper_params, dataloaders, tokenizer, saved_dir, start_epochs):

    optimizer = AdamW(model.parameters(), lr=hyper_params['learning_rate'])

    best_model_wts = model.state_dict() #si salva i pesi del modello al t0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_loss = {"train": float('inf'), "val": float('inf')} 
    history_loss = {"train": [], "val": []}

    epochs = hyper_params['epochs']

    total_epochs = epochs+start_epochs
    for epoch in range(start_epochs,total_epochs):
        print(f'Epoch {epoch}/{total_epochs - 1}')
        print('-' * 50)

        # Initialize epoch variables
        sum_loss = {"train": 0, "val": 0}

        
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
                gabriel_mask = batch["gabriel_mask"].to(device)

                input_ids_tensor = torch.tensor(tokenizer(batch['text_a'])['input_ids'][0]).to(device) #access the second list [[101,234,987,102]]


                if len(input_ids_tensor) > 3:
                    mask_token_index = np.random.randint(1, len(input_ids_tensor) - 2)
                else:
                    mask_token_index = 1  

                #da 1 a len-2 perchè evitiamo di macherare il token cls e sep.
                #il primo elmento della lista ha indice 0 e l'ultimo a indice len(n)-1. Quindi per togliere il cls devo andare a len(n)-2.

                temp_mask = torch.zeros(ids.size(-1)).to(device)
                temp_mask[mask_token_index] = 1
                masked_input = ids.masked_fill(temp_mask == 1, 103).to(device)
                labels = ids.masked_fill(masked_input != 103, -100).to(device)
                
                optimizer.zero_grad()

                output = model(input_ids=masked_input, attention_mask=mask, gabriel_mask=gabriel_mask, labels=labels)
        
                loss = output.loss 

                sum_loss[split] += loss.item() #batch loss
 
                if split == "train":
                    # Compute gradients
                    loss.backward()
                    # Optimize
                    optimizer.step()



        epoch_loss = {split: sum_loss[split] / len(dataloaders[split]) for split in ["train", "val"]} 

        # Update history
        for split in ["train", "val"]:
                history_loss[split].append(epoch_loss[split])

        if epoch_loss['train'] < best_loss['train']:  # Compare train loss with best train loss
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

        print(f"Train Loss: {epoch_loss['train']:.4f}")
        print(f"Val Loss: {epoch_loss['val']:.4f}")


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
    plt.savefig(os.path.join(saved_dir,"loss_plot.png"), dpi=300, bbox_inches="tight")  #
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


    task = 'inference'
    dataset_name = "mnli"
    pretrained_model = "bert-base-uncased"

    model_name = pretrained_model + "_" + pst_now.strftime("%Y-%m-%d_%H-%M-%S")
    saved_model_dir = os.path.join(os.getcwd(),'scripts','finetuning',task,'results', model_name)
    os.makedirs(saved_model_dir, exist_ok=True)

    # Model parameters
    hyper_params = {'epochs' : 10, 'batch_size' : 64, 'learning_rate' : 5e-5}

    model = CustomformerForMaskedLM2.from_pretrained(pretrained_model)

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

    sliced_train_dataset = pd.DataFrame(dataset["train"])

    train_df, val_df = train_test_split(sliced_train_dataset, test_size=0.2, random_state=seed)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    
    train_dataset = Dataset(texts_a= train_df['premise'],#filtered_dataset["train"]['premise'], #old
                            idx = train_df['idx'],
                            tokenizer=tokenizer,
                            max_len=100)
    
    validation_dataset = Dataset(texts_a= val_df['premise'],#filtered_dataset["train"]['premise'], #old
                                idx = val_df['idx'],
                                tokenizer=tokenizer,
                                max_len=100)
    
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=hyper_params['batch_size'],
                                   shuffle = True,
                                   pin_memory=True,
                                   num_workers=4)  # Transfer tensors to CUDA in a more efficient way
    
    train_data_loader = DataLoader(validation_dataset,
                                   batch_size=hyper_params['batch_size'],
                                   shuffle = True,
                                   pin_memory=True,
                                   num_workers=4)
    

    dataloaders = {'train': train_data_loader,'val': train_data_loader}

    checkpoint = training_bert_model_4_mlm_task(model=model,optimizer=optimizer,
                                   hyper_params=hyper_params,
                                   dataloaders=dataloaders,
                                   tokenizer=tokenizer, saved_dir=saved_model_dir,start_epochs=start_epoch)

    
       




if __name__ == '__main__':
    #checkpoint_path = "..."" # Change this to the path of your checkpoint
    #checkpoint_path = os.path.join(os.getcwd(),'scripts/finetuning/mlm_task/results/bert-base-uncased_2024-06-26_16-25-02/checkpoint.pt')
    #main(checkpoint_path)
    main()