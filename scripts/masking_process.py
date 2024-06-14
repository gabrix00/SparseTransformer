from spacy_dependency_15_06_v2 import create_dependency_pairs
from from_parser2masking import from_parser2masking
from from_tensor2masking import from_tensor2masking


#text = "BERT tokenizer something known as subword-based tokenization. Subword-tokenization splits unknown words into smaller words or characters such that the model can derive some meaning from the tokens."
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text = "Long Climb Pays Off for Jets' Linebacker Mark Brown has gone from the practice squad to the starting lineup in a little more than a year."
#create_dependency_pairs(text)
text= "my walkman broke so i'm upset now i just have to turn the stereo up real loud"
#print(create_dependency_pairs(text))
import numpy  as np
import torch

def masking (text,path=None):
    if path:
        return from_parser2masking(text,create_dependency_pairs(text.lower()),True, path = path)
    else: 
        return from_parser2masking(text,create_dependency_pairs(text.lower()),True)

def rmasking (text,rand_mask):
    return from_tensor2masking(text,rand_mask,False)

def mask_multiple_sentences(text1,text2):
    text = text1+' [SEP] '+text2

    t1 = masking(text1)
    t2 = masking(text2)

    #drop CLS row and column from the second gabriel_mask
    new_t2 = np.delete(t2.numpy(), 0, axis=1)
    new_t2 = np.delete(new_t2, 0, axis=0)



    first = t1.shape[0]
    #second = new_t2.shape[0]
    n = t1.shape[0] + new_t2.shape[0]

    # Concatenate t1 with zeros along dimension 0
    t1 = torch.cat([t1, torch.zeros(n - t1.shape[0], t1.shape[0])], dim=0)
    # Concatenate t1 with zeros along dimension 1
    t1 = torch.cat([t1, torch.zeros(t1.shape[0], n - t1.shape[1])], dim=1)



    t1[0, first:n] = 1 #completamento token CLS 
    t1[first:n,0] = 1 #completamento token CLS 

    # riepmpimento matrice inizializata a zeri con new_t2, ossia relazione della seconda gabriel_mask
    t1[first:n,first:n] += new_t2 #somma perchè rimarrà 0 dove è in entrambi 0 e diventerà 1 quando la rel è presente


    rmasking(text=text,rand_mask=t1)


    return t1

#s = str(p)+' [SEP] '+str(h)
#mask_multiple_sentences(p,h)
#masking(s)