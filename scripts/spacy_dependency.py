
### ATTENZIONE QUESTO SCRIPT è SPACY-DEPENDENCY_15_06_V2 ####
import spacy
from spacy_mapping import spacy_map
from normalization import normalizzation
import numpy as np
from transformers import AutoTokenizer
import re

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"
#text = "(Read for Slate 's take on Jackson's findings.)"
#text= "I'm upset that my walkman broke and now I have to turn the stereo up really loud."
#text="my walkman broke so i'm ups_et now i have just to turn the stereo up real loud"
#text="Even if they were in their ship, that is, rather than in this--this--_cage_.".lower()
#text= "He actually feels _bad_ that we're leaving.".lower()
#text= "Don't get too concerned if you seem to be following a very roundabout route."# [SEP] Do not be surprised if your route in not directly to your destination.".lower()


def underscore_count(text):
    # Rimuovi qualsiasi sequenza di numeri
    #cleaned_text = re.sub(r'\d+', '', text)
    
    # Trova tutte le sequenze di underscore
    underscore_sequences = re.findall(r'_+', text)
    
    # Trova la lunghezza della sequenza più lunga
    if underscore_sequences:
        longest_sequence_length = max(len(seq) for seq in underscore_sequences)
    else:
        longest_sequence_length = 0
    
    return longest_sequence_length

def spacy_dependency(text:str):
    #text = normalizzation(text)
    #print(spacy_map_dict)
    #print('\n\n')
    sentence = nlp(text)
    tokens_children_dep = []
    adjusted_index =0
    new_list_token_idx=[]
    
    filtered_sentence = [token for token in sentence if token.text != " "]
    #normal_sentence = [token for token in sentence]

    #print(f"filtered_sentence: {filtered_sentence}")#debug
    #print(f"normal_sentence: {normal_sentence}")#debug

    spacy_map_dict = spacy_map(text)
    for index,token in enumerate(filtered_sentence):
        #for child in token.children:
            
        #presence = False
        #print(f"token:{token}")#debug
        #print(type(token))
        #try:
        #print(f"mapped_token: {spacy_map_dict[str(token)]}")
        #print('\n')#debug
        new_list_token_idx.append(token.text.lower()+'____'+str(index + adjusted_index))
        if isinstance(spacy_map_dict[str(token)], list):
            for i, el in enumerate(spacy_map_dict[str(token)]):
                # Salta il primo sottotoken (ad es. "token") e i sottottoken che iniziano con "##"
                if i == 0 or el[:2] == '##':
                    continue
                else:
                    #presence = True
                    adjusted_index+=1
        #except Exception as e:
        #    print("Spacy found an empty token " " mapped to [] in Bert Tokenizer. It was removed for alignment purposes. ")
        #    print(str(e))

        #    continue
    #print(f"new_list_token_idx: {new_list_token_idx}")#debug   
    #print('\n\n')     
    original_list_token_idx = [token.text.lower()+'____'+str(index)  for index,token in enumerate(filtered_sentence)]
    #print(f"original_list_token_idx: {original_list_token_idx}")#debug   
    #print('\n\n')


    for index,token in enumerate(filtered_sentence):
        for child in token.children:
            if child.text == ' ':
                continue
            #print(token,child)#debug
 
    #print('\n')
    for index,token in enumerate(filtered_sentence):
        for child in token.children:
            if  child.text == ' ':
                continue
            #print((token.text.lower()+'____'+str(index), # mi creo tutte le coppie tenendo traccia dei relativi suffissi di posizione 
                                    #child.text.lower()+'____'+str(list(filtered_sentence).index(child)))) #debug


    
    mapped_dict = {original: new for new, original in zip(new_list_token_idx, original_list_token_idx)}

    #print(mapped_dict) #debug importante!


    for index,token in enumerate(filtered_sentence):
        for child in token.children:
            if child.text == ' ':
                continue
            #print(token.text.lower()+'____'+str(index), child.text.lower()+'____'+str(list(sentence).index(child)))#old
            #print(mapped_dict[token.text.lower()+'____'+str(index)],#debug
            #      mapped_dict[child.text.lower()+'____'+str(list(filtered_sentence).index(child))])#debug
            #print('\n') #debug
            tokens_children_dep.append((mapped_dict[token.text.lower()+'____'+str(index)],
                  mapped_dict[child.text.lower()+'____'+str(list(filtered_sentence).index(child))]))

    return tokens_children_dep


def create_dependency_pairs(text:str):
    #text = normalizzation(text)
    spacy_map_dict = spacy_map(text)
    lista_dipendenze = spacy_dependency(text)

    lista_dipendenze_mappate = [] #secondo il tokenizer di bert
    for tup in lista_dipendenze:
        #suffix_f= '_'+str(tup[0]).split('____')[1]
        #suffix_s= '_'+str(tup[1]).split('____')[1]
        #print(tup) #debug
        uc1 = underscore_count(tup[0])
        uc2 = underscore_count(tup[1])
        #print(uc1) #debug
        #print(uc2) #debug
        #print('\n\n') #debug
        #print(tup[1].split('____')[1][:uc2-4])#debug
        if uc1>4:
            print('underscore_count grather than 4 for tup1')
            spacy_mapping_first = spacy_map_dict[tup[0].split('____')[1][:uc1-4]]
            print(spacy_mapping_first)
            print(tup[0].split('____')[1])
            #print(spacy_map_dict[tup[0].split('____')[1][:uc1-4]])
        else:
            spacy_mapping_first = spacy_map_dict[tup[0].split('____')[0]]
            #print(spacy_map_dict[tup[0].split('____')[0]])
        if uc2>4:
            print('underscore_count grather than 4 for tup2')
            spacy_mapping_second = spacy_map_dict[tup[1].split('____')[1][:uc2-4]]
            print(spacy_mapping_second)
            print(tup[1].split('____')[1])
            
            #print(spacy_map_dict[tup[1].split('____')[1][:uc2-4]])
        else:
            spacy_mapping_second = spacy_map_dict[tup[1].split('____')[0]]
            #print(spacy_map_dict[tup[1].split('____')[0]])
        
        #print(spacy_map_dict[tup[0].split('____')[0]])
        #print(spacy_map_dict[tup[1].split('____')[0]])
            
        try:
            #suffix_f = int(tup[0].split('____')[1])
            #suffix_s = int(tup[1].split('____')[1])
            suffix_f = int(tup[0].split('____')[1])
            suffix_s = int(tup[1].split('____')[1])
            #print(suffix_f)
            #print(suffix_s)
        except:

            suffix_f = int(re.search(r'\d+', tup[0].split('____')[1]).group())
            suffix_s = int(re.search(r'\d+', tup[1].split('____')[1]).group())
            #print(suffix_f)
            #print(suffix_s)

        
        # Controllo se entrambi i termini della tupla sono liste
        if isinstance(spacy_mapping_first, list) and isinstance(spacy_mapping_second, list):
            #print('1 caso') #debug
            #print(spacy_map_dict[tup[0].split('____')[0]])#debug
            #print(spacy_map_dict[tup[1].split('____')[0]])#debug
            # Aggiungi i suffissi appropriati a ogni elemento delle liste
            #updated_left_list = [el + suffix_f for el in spacy_map_dict[tup[0]] if el[:2]=='##']
            updated_left_list= []
            for index, el in enumerate(spacy_mapping_first): #se prendiamo il caso di [token,##izer] 
                index_adjustment = 0
                #print(index,el) #debug
                if index == 0:
                    updated_left_list.append(el + '____'+str(suffix_f + index_adjustment)) # token + suffisso
                else:
                    if el[:2]=='##': 
                        updated_left_list.append(el +  '____'+str(suffix_f + index_adjustment))
                        # ##izer + suffisso (com'è giusto che sia dato che si tratta della stessa parola)
                    else:
                        index_adjustment += 1
                        updated_left_list.append(el +  '____'+str(suffix_f + index_adjustment))
                        #caso in cui il secondo elemento non fa parte della stessa parola es in [',m] --> m 
            
            updated_right_list = []
            for index, el in enumerate(spacy_mapping_second): #se prendiamo il caso di [token,##izer] 
                index_adjustment = 0
                #print(index,el) #debug
                if index == 0:
                    updated_right_list.append(el +  '____'+str(suffix_s + index_adjustment)) # token + suffisso
                else:
                    if el[:2]=='##': 
                        updated_right_list.append(el +  '____'+str(suffix_s + index_adjustment))
                        # ##izer + suffisso (com'è giusto che sia dato che si tratta della stessa parola)
                    else:
                        index_adjustment += 1
                        updated_right_list.append(el +  '____'+str(suffix_s + index_adjustment))
                        #caso in cui il secondo elemento non fa parte della stessa parola es in [',m] --> m 

            #print('\n')
            #updated_right_list = [el + suffix_s for el in spacy_map_dict[tup[1]]]
            # Aggiungi le coppie di liste modificate alla lista delle dipendenze mappate
            lista_dipendenze_mappate.append((updated_left_list, updated_right_list))
            continue

        # Controllo se solo il primo termine della tupla è una lista
        if isinstance(spacy_mapping_first, list):
            #print('2 caso') #debug
            updated_left_list= []
            for index, el in enumerate(spacy_mapping_first): #se prendiamo il caso di [token,##izer] 
                index_adjustment = 0
                #print(index,el) #debug
                if index == 0:
                    updated_left_list.append(el +  '____'+str(suffix_f + index_adjustment)) # token + suffisso
                else:
                    if el[:2]=='##': 
                        updated_left_list.append(el +  '____'+str(suffix_f + index_adjustment))
                        # ##izer + suffisso (com'è giusto che sia dato che si tratta della stessa parola)
                    else:
                        index_adjustment += 1
                        updated_left_list.append(el +  '____'+str(suffix_f + index_adjustment))
            #print('\n')
        # Se il valore è una lista, aggiungi il suffisso a ogni elemento
            #updated_left_list = [el + suffix_f for el in spacy_map_dict[tup[0]]]

            lista_dipendenze_mappate.append((updated_left_list,tup[1]))
            continue
        

        # Controllo se solo il secondo termine della tupla è una lista
        if isinstance(spacy_mapping_second, list):
            #print('3 caso') #debug
            updated_right_list = []
            for index, el in enumerate(spacy_mapping_second): #se prendiamo il caso di [token,##izer] 
                index_adjustment = 0
                #print(index,el) #debug
                if index == 0:
                    updated_right_list.append(el +  '____'+str(suffix_s + index_adjustment)) # token + suffisso
                else:
                    if el[:2]=='##': 
                        updated_right_list.append(el +  '____'+str(suffix_s + index_adjustment))
                        # ##izer + suffisso (com'è giusto che sia dato che si tratta della stessa parola)
                    else:
                        index_adjustment += 1
                        updated_right_list.append(el +  '____'+str(suffix_s + index_adjustment))
            #print('\n')#debug
            #updated_right_list = [el + suffix_s for el in spacy_map_dict[tup[1]]]
            lista_dipendenze_mappate.append((tup[0],updated_right_list))
            continue
  
        lista_dipendenze_mappate.append((tup[0],tup[1])) #nel caso nessuno dei due sia lista si ottengono le mappe normali
    
    #print(f"lista dipendenze mappate: {lista_dipendenze_mappate}") #debug
    #print('\n\n')

    ##### !!!CASISTICA!!! ##### 
    update_lista_dipendenze_mappate=[]
    for tup in lista_dipendenze_mappate:
        # caso prima elemento della tuple è una lista e il secondo è un stringa es: (['token', '##izer'], 'bert')
        if type(tup[0]) == list and type(tup[1]) == str:
            for el in tup[0]:
                update_lista_dipendenze_mappate.append((el,tup[1]))

        # caso prima elemento della tuple è una lista e il secondo è una lista  es: (['token', '##ization'], ['sub', '##word'])
        elif type(tup[0]) == list and type(tup[1]) == list:
            for el1 in tup[0]:
                for el2 in tup[1]:
                    update_lista_dipendenze_mappate.append((el1,el2))

        # caso primo elemento della tupla è una stringa e il secondo è una lista es: ('uses', ['token', '##izer'])
        elif type(tup[0]) == str and type(tup[1]) == list:
            for el in tup[1]:
                    update_lista_dipendenze_mappate.append((tup[0],el))

        # caso primo elemento della tupla è una stringa e il secondo è una stringa es: ('uses', 'something')
        else:
            update_lista_dipendenze_mappate.append(tup) #nessuna modifica
    #print('\n\n')
    #print(f"update_lista_dipendenze_mappate: {update_lista_dipendenze_mappate}")#debug
    return update_lista_dipendenze_mappate


'''
print('SPACY MAP:')
print(spacy_map(text))
print('\n\n')

print('SPACY DEPENDENCY:')
print(spacy_dependency(text))
print('\n\n')


print('create_dependency_pairs:')
print(create_dependency_pairs(text))
print('\n\n')
'''
