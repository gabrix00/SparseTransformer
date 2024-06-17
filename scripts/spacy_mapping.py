import spacy
#from transformers import BertTokenizer
from transformers import AutoTokenizer
from normalization import normalizzation

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#text= "(Read  for Slate 's take on Jackson's findings.)"
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text= "my walkman broke so i'm upset now i just have to turn the stereo up real loud"
#text="Give Microsoft a monopoly on browsers, and you'll intensify the downward pressure on the price of its operating systems. [SEP] The downward pressure on the price of its operating systems will intensify if Microsoft is given a monopoly on browsers."
#text= "Don't get too concerned if you seem to be following a very roundabout route".lower()

def mapping (tokens_list:list, dict_to_update:dict):
    maps = {}

    #processo che mi fa i songoli encoding dei token per cui vi è un miosmatch tra i due tokenizer
    for t in tokens_list:
        encodes = tokenizer(t,add_special_tokens=False,return_tensors='pt')
        encodes = encodes['input_ids'][0].tolist()
        decodes = [tokenizer.decode(enc) for enc in encodes]


        if t not in maps: # creo il dict che presenta solo i mismatch chiave tokenizer spacy, valore tokenizer bert
            maps[t] = decodes # lo riempio, questo dict non avrà gli idnci di posizone! perchè t è proviene dalla lista 'spacy_tokenizzation'! es: tokenizer:[token, ##izer]
    
    for k in dict_to_update: # dict_to_update è il dict di mappe chiave valore (chiave spacy, valore spacy), lo voglio aggiornare in modo tale che dove è presente un mismatch tra i due tokenixer, abbia chiave spacy, valore pythorch
        if str(k).split('____')[0] in maps: # str(k).split('____')[0] mi serve ad epurare il token dal suffisso numerico, perchè la mappa in pythorch non lo presenta (infatti è tipo tokenizer_3:[token,##izer])
            dict_to_update[k] = maps[str(k).split('____')[0]] #aggiorno il dict con la mappa corretta (in questo caso per tutti i token per cui c'è un mismatch tra i due tokenizer)
 
    return dict_to_update




def spacy_map(text:str):
    
    #tokenizer Spacy and Bert
    #text= normalizzation(text) #già il testo è normalizzato nel get.item
    if '[SEP]' in text:
        #print(True)
        text = text.replace(' [SEP] ','') #RIMUOVO [SPAZIO]+[SEP]+[SPAZIO] PER COME LO CREO NELLA CONCAT

    sentence = nlp(text)
    #tokens = tokenizer(text,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True) #NotImplementedError: return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast
    tokens = tokenizer(text,add_special_tokens=False,return_tensors='pt',return_offsets_mapping=True) #correct


    spacy_tokenizzation = [token.text.lower() for token in sentence]
    bert_tokenizzation  = [tokenizer.decode(token) for token in tokens['input_ids'][0].tolist()]


    tokens_mismatch = [token for token in spacy_tokenizzation if token not in bert_tokenizzation] # prendo la lista dei mismatch tra i due tokenizer

    #dict to pass to the mapping function che si chiamerà dict_to_update in 'mapping'
    #spacy_tokenizzation_dict = dict([(token.text.lower()+'____'+str(index),token.text.lower()+'____'+str(index)) for index,token in enumerate(sentence)]) #mi creo le mappa con gli indici di posizione
    spacy_tokenizzation_dict = dict([(token.text.lower(),token.text.lower()) for index,token in enumerate(sentence)]) #mi creo le mappa con gli indici di posizione

    final_map_dict = mapping(tokens_mismatch,spacy_tokenizzation_dict)

    #for k, v in final_map_dict.items():
    #    if k == " " and v == []:
    #        print('Spacy found an empty token " " mapped to [] in Bert Tokenizer. It was removed for alignment purposes.')

    #if " " in final_map_dict and final_map_dict[" "] == []:
    #    del final_map_dict[" "]

    return final_map_dict


if __name__ == '__main__':
    print(spacy_map(normalizzation(text)))


