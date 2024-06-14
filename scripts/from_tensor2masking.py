import matplotlib.pyplot as plt
import numpy as np
import torch 
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def from_tensor2masking(text:str, rand_mask:torch.tensor, viz:bool = False):

    encodes = tokenizer(text,add_special_tokens=False,return_tensors='pt')
    encodes = encodes['input_ids'][0].tolist()
    decodes = [tokenizer.decode(enc) for enc in encodes]

    mask = rand_mask.numpy()
    #print(mask)

                
    if viz:
        plt.imshow(mask, cmap='Blues', interpolation='nearest')
        plt.title('Random Attention Mask')

        adjusted_decodes = []
        index_adjustment = 0 
        for index, token in enumerate(decodes):
            if token[:2]=='##':
                index_adjustment += 1 #ogni volta che si incontra un token con ## davanti si scala il suo indice e di tutti token successivi di 1
                adjusted_decodes.append(str(token)+'_'+str(index - index_adjustment))
                
            else:
                adjusted_decodes.append(str(token)+'_'+str(index - index_adjustment))
        
        adjusted_decodes = ['[CLS]'] + adjusted_decodes + ['[SEP]']
            
        plt.xticks(range(len([el for el in adjusted_decodes])), adjusted_decodes,rotation=45)
        plt.yticks(range(len([el for el in adjusted_decodes])), adjusted_decodes,rotation=45)
        plt.colorbar()
        plt.show()

    #print(mask)
    #return torch.tensor(mask)

##### PER TAMPONARE ######

#from spacy_dependency import create_dependency_pairs
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text = "BERT tokenizer uses something known as subword-based tokenization. Subword-tokenization splits unknown words into smaller words or characters such that the model can derive some meaning from the tokens."
#text = 'A black hole is a region of spacetime where gravity is so strong that nothing'
#create_dependency_pairs(text)

#print(create_dependency_pairs(text))

#from_parser2masking(text,create_dependency_pairs(text),True)

