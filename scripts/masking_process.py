from spacy_dependency import create_dependency_pairs
from from_parser2masking import from_parser2masking
from from_tensor2masking import from_tensor2masking


#text = "BERT tokenizer something known as subword-based tokenization. Subword-tokenization splits unknown words into smaller words or characters such that the model can derive some meaning from the tokens."
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text = "Long Climb Pays Off for Jets' Linebacker Mark Brown has gone from the practice squad to the starting lineup in a little more than a year."
#create_dependency_pairs(text)
#text= "my walkman broke so i'm upset now i just have to turn the stereo up real loud"

#print(create_dependency_pairs(text))
import numpy  as np
import torch

def masking (text, viz = False, path=None):
    dependency_pairs_list = create_dependency_pairs(text)#.lower()) #text.lower() it's must otherwise creation pairs failed!
    if viz: 
        if path:
            return from_parser2masking(text,dependency_pairs_list,True, path = path)
        else: 
            return from_parser2masking(text,dependency_pairs_list,viz=True)
    else:
        return from_parser2masking(text,dependency_pairs_list)


def rmasking (text,rand_mask,viz = False, path=None): # Dont use text.lower()! it will cause error! with [SEP] token 
    if viz: 
        if path:
            return from_tensor2masking(text,rand_mask,viz = True, path = path)
        else: 
            return from_tensor2masking(text,rand_mask,viz = True)
    else:
        return from_tensor2masking(text,rand_mask)

def mask_multiple_sentences(text1,text2):
    #text = text1+' [SEP] '+text2

    t1 = masking(text1)
    t2 = masking(text2)

    #drop CLS row and column from the second gabriel_mask
    new_t2 = np.delete(t2.numpy(), 0, axis=1)
    new_t2 = np.delete(new_t2, 0, axis=0)



    first = t1.shape[0]
    #second = new_t2.shape[0]
    n = t1.shape[0] + new_t2.shape[0]

    # Concatenate t1 with zeros along dimension 0
    t1_composed = torch.cat([t1, torch.zeros(n - t1.shape[0], t1.shape[0])], dim=0)
    # Concatenate t1 with zeros along dimension 1
    t1_composed = torch.cat([t1_composed, torch.zeros(t1_composed.shape[0], n - t1_composed.shape[1])], dim=1)



    t1_composed[0, first:n] = 1 #completamento token CLS 
    t1_composed[first:n,0] = 1 #completamento token CLS 

    # riepmpimento matrice inizializata a zeri con new_t2, ossia relazione della seconda gabriel_mask
    t1_composed[first:n,first:n] += new_t2 #somma perchè rimarrà 0 dove è in entrambi 0 e diventerà 1 quando la rel è presente


    
    
    #rmasking(text=text,rand_mask=t1_composed, viz= True)

    

    return t1_composed,t1
"""

p = "my walkman broke so i'm upset now i just have to turn the stereo up real loud"
h = "I'm upset that my walkman broke and now I have to turn the stereo up really loud."
s = str(p)+' [SEP] '+str(h)
t1_composed,t1 = mask_multiple_sentences(p,h)

rmasking(text=p,rand_mask=t1,viz=True)
rmasking(text=s,rand_mask=t1_composed, viz= True)


print(t1_composed.shape)
print('\n')
print(t1.shape)
#masking(s)
"""

"""
from datasets import load_dataset
from datasets import DatasetDict
dataset = load_dataset("LysandreJik/glue-mnli-train")
dataset = dataset.filter(lambda example: len(example["premise"]) == 120 and len(example["hypothesis"]) == 120)
sliced_train_dataset = DatasetDict(dataset["train"][:])
print(sliced_train_dataset)
length = len(sliced_train_dataset['premise'])
print(f"len of the dataset is: {length}")
print('\n\n')
for index in range(length):
    p = sliced_train_dataset['premise'][index]
    h = sliced_train_dataset['hypothesis'][index]

    s = str(p)+' [SEP] '+str(h)
    print(s)

    t1_composed,t1 = mask_multiple_sentences(p,h)
    print('\n')
    print(t1_composed.shape)
    print(t1.shape)
    print('\n')

    rmasking(text=p,rand_mask=t1,viz=True)
    rmasking(text=s,rand_mask=t1_composed, viz= True)
"""