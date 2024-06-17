import matplotlib.pyplot as plt
import numpy as np
import torch 
from transformers import AutoTokenizer
import seaborn as sns
from normalization import normalizzation

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def from_parser2masking(text:str, list_of_mapped_rel:list, viz:bool = False, path=None):
    #text = normalizzation(text) #già il testo è normalizzato nel get.item

    encodes = tokenizer(text,add_special_tokens=False,return_tensors='pt')
    encodes = encodes['input_ids'][0].tolist()
    decodes = [tokenizer.decode(enc) for enc in encodes]
    #print(decodes) #debug

    n = len(decodes)

    mask = np.zeros((n+2, n+2)) #+2 per [CLS] e [SEP] token

    # adjusted_decodes serve a mappare correttamente i decodes, evitando il mismatch tra parole splittate es token_2, ##izer_2
    adjusted_decodes = []
    index_adjustment = 0
    for index, token in enumerate(decodes):
            if token[:2]=='##':
                index_adjustment += 1
                adjusted_decodes.append(str(token)+'____'+str(index - index_adjustment))
                #print(str(token)+'____'+str(index - index_adjustment))#debug
                
            else:
                adjusted_decodes.append(str(token)+'____'+str(index - index_adjustment))
                #print(str(token)+'____'+str(index - index_adjustment)) #debug

    adjusted_decodes = ['[CLS]'] + adjusted_decodes + ['[SEP]']

    #print(adjusted_decodes) #debug
    for i, token in enumerate(adjusted_decodes):
        for j, other_token in enumerate(adjusted_decodes):
            if i != j:
                #print(token,other_token) #debug
                #if token == '[CLS]' or token == '[SEP]' : #deprecated
                if token == '[CLS]':
                    mask[i][j] = 1
                    mask[j][i] = 1 #new modica per simmetria
                    continue
                if token == '[SEP]' :
                    continue
                if (token,other_token) in list_of_mapped_rel:
                    mask[i][j] = 1
                    mask[j][i] = 1 #new modica per simmetria
                elif other_token[:2]=='##' and (token.split('____')[1] == other_token.split('____')[1]) and (token != '[CLS]' or  token != '[SEP]'): #attendo stessa parola splittata es token_2, ##izer_2
                    mask[i][j] = 1
                    mask[j][i] = 1  # attendere sia token -->##izer in quanto stessa parola, ma anche ##izer-->token per completezza
            else:   
                mask[i][i] = 1  #caso token attende con se stesso a priori


                
    if viz:

        plt.imshow(mask, cmap='Blues', interpolation='nearest')
        plt.title('Attention Mask')
    

        adjusted_decodes = []
        index_adjustment = 0 
        for index, token in enumerate(decodes):
            if token[:2]=='##':
                index_adjustment += 1 #ogni volta che si incontra un token con ## davanti si scala il suo indice e di tutti token successivi di 1
                adjusted_decodes.append(str(token)+'____'+str(index - index_adjustment))
                
            else:
                adjusted_decodes.append(str(token)+'____'+str(index - index_adjustment))
        
        adjusted_decodes = ['[CLS]'] + adjusted_decodes + ['[SEP]']
        
        plt.xticks(range(len([el for el in adjusted_decodes])), adjusted_decodes,rotation=45)
        plt.yticks(range(len([el for el in adjusted_decodes])), adjusted_decodes,rotation=45)
        plt.colorbar()

        # Aggiungere griglia per bordi delle celle
        plt.grid(which='both', color='black', linestyle='--',linewidth=0.05)
    

        # Disabilitare i tick della griglia
        plt.gca().set_xticks([x - 0.5 for x in range(1, len(adjusted_decodes))], minor=True)
        plt.gca().set_yticks([y - 0.5 for y in range(1, len(adjusted_decodes))], minor=True)
        plt.grid(which='minor', color='black', linewidth=1)

        # Rimuovere i tick della griglia principale per mantenere solo quelli minori
        plt.gca().tick_params(which='both', bottom=False, left=False)


        if path:
            plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.show()

    #print(mask)
    return torch.tensor(mask)

##### PER TAMPONARE ######

#from spacy_dependency import create_dependency_pairs
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text = "BERT tokenizer uses something known as subword-based tokenization. Subword-tokenization splits unknown words into smaller words or characters such that the model can derive some meaning from the tokens."
#text = 'A black hole is a region of spacetime where gravity is so strong that nothing'
#create_dependency_pairs(text)

#print(create_dependency_pairs(text))

#from_parser2masking(text,create_dependency_pairs(text),True)

