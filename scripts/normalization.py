import re

def add_space_to_time(text):
    # Trova e sostituisci le sequenze del tipo numero+'am' o numero+'pm' oppure arrivo tra '3m'
    modified_text = re.sub(r'(\d+)(am|pm|m|s|sec|min|hr|hour)\b', r'\1 \2', text)
    return modified_text

def add_space_to_weight(text):
    # Trova e sostituisci le sequenze del tipo numero+('lb' o 'kg' o 'g' o altre unità di misura)
    modified_text = re.sub(r'(\d+)(lb|kg|g|mg|oz|stone|ton|metric ton|short ton|long ton)\b', r'\1 \2', text)
    return modified_text


def relaxation(text):
    # Dizionario per mappare le diverse forme di negazione

    negation_mapping = {
        "don't": "do not",
        "don 't": "do not",
        "don' t": "do not",
        "don ' t": "do not",
        "doesn't": "does not",
        "doesn 't": "does not",
        "doesn' t": "does not",
        "doesn ' t": "does not",
        "didn't": "did not",
        "didn 't": "did not",
        "didn' t": "did not",
        "didn ' t": "did not",
        "won't": "will not",
        "won 't": "will not",
        "won' t": "will not",
        "won ' t": "will not",
        "wouldn't": "would not",
        "wouldn 't": "would not",
        "wouldn' t": "would not",
        "wouldn ' t": "would not",
        "can't": "can not",  # correggere la forma corretta
        "can 't": "can not",
        "can' t": "can not",
        "can ' t": "cannot",
        "couldn't": "could not",
        "couldn 't": "could not",
        "couldn' t": "could not",
        "couldn ' t": "could not",
        "shouldn't": "should not",
        "shouldn 't": "should not",
        "shouldn' t": "should not",
        "shouldn ' t": "should not",
        "mightn't": "might not",
        "mightn 't": "might not",
        "mightn' t": "might not",
        "mightn ' t": "might not",
        "mustn't": "must not",
        "mustn 't": "must not",
        "mustn' t": "must not",
        "mustn ' t": "must not",
        "hasn't": "has not",
        "hasn 't": "has not",
        "hasn' t": "has not",
        "hasn ' t": "has not",
        "haven't": "have not",
        "haven 't": "have not",
        "haven' t": "have not",
        "haven ' t": "have not",
        "hadn't": "had not",
        "hadn 't": "had not",
        "hadn' t": "had not",
        "hadn ' t": "had not",
        "isn't": "is not",
        "isn 't": "is not",
        "isn' t": "is not",
        "isn ' t": "is not",
        "aren't": "are not",
        "aren 't": "are not",
        "aren' t": "are not",
        "aren ' t": "are not",
        "wasn't": "was not",
        "wasn 't": "was not",
        "wasn' t": "was not",
        "wasn ' t": "was not",
        "amn't": "am not",
        "amn 't": "am not",
        "amn' t": "am not",
        "amn ' t": "am not",
        "weren't": "were not",
        "weren 't": "were not",
        "weren' t": "were not",
        "weren ' t": "were not",
        "needn't": "need not",
        "needn 't": "need not",
        "needn' t": "need not",
        "needn ' t": "need not",
        "daren't": "dare not",
        "daren 't": "dare not",
        "daren' t": "dare not",
        "daren ' t": "dare not",
        "shan't": "shall not",
        "shan 't": "shall not",
        "shan' t": "shall not",
        "shan ' t": "shall not",
        "cannot": "can not",
        "wont": "will not",
        "dont": "do not",
        "doesnt": "does not",
        "didnt": "did not",
        "wouldnt": "would not",
        "cant": "can not", 
        "couldnt": "could not",
        "shouldnt": "should not",
        "mightnt": "might not",
        "mustnt": "must not",
        "hasnt": "has not",
        "havent": "have not",
        "hadnt": "had not",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "neednt": "need not",
        "darent": "dare not",
        "shant": "shall not",
        "ain't": "ain 't",
        "ain' t": "ain 't",
        "aint": "ain 't",
        "amnt": "am not",
        "amn't": "am not",
        "amnt": "am not",
        "arenot": "are not",
        "couldnt": "could not",
        "shouldnt": "should not",
        "mightnt": "might not",
        "mustnt": "must not",
        "hasnt": "has not",
        "havent": "have not",
        "hadnt": "had not",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "neednt": "need not",
        "darent": "dare not",
        "shant": "shall not",
    }

    others_mapping = {
        'gonna': 'going to',
        "wanna": "want to",
        "gotta": "got to",
        "hafta": "have to",
        "gimme": "give me",
        "lemme": "let me",
        "kinda": "kind of",
        "sorta": "sort of",
        "oughta": "ought to",
        "outta": "out of",
        "whaddya": "what do you",
        "whatcha": "what are you",
        "y'all": "you all",
        "y' all": "you all",
        "y 'all": "you all",
        "y ' all": "you all",
        "c'm": "come",
        "c' m": "come",
        "cmon": "come on",
        "c'mon": "come on",

    }
    
    #print("Testo originale:", text)
    for pattern, replacement in negation_mapping.items():
        text = re.sub(r'\b{}\b'.format(re.escape(pattern)), replacement, text)

    for pattern, replacement in others_mapping.items():
        text = re.sub(r'\b{}\b'.format(re.escape(pattern)), replacement, text)
    #print("Testo normalizzato:", text)
    return text
  


def normalizzation(text):
    text = text.lower()
    text = add_space_to_time(text) 
    text = relaxation(text)
    return text

#text= "my walkman doesn't broke so i'm upset now i just have to turn the stereo up real loud"
#print("Testo normalizzato:", normalizzation(text))
"""
"i'm": "i am",
"i 'm": "i am",
"i' m": "i am",
"i ' m": "i am",
"you're": "you are",
"you 're": "you are",
"you' re": "you are",
"you ' re": "you are",
"he's": "he is",
"he 's": "he is",
"he' s": "he is",
"he ' s": "he is",
"she's": "she is",
"she 's": "she is",
"she' s": "she is",
"she ' s": "she is",
"it's": "it is",
"it 's": "it is",
"it' s": "it is",
"it ' s": "it is",
"we're": "we are",
"we 're": "we are",
"we' re": "we are",
"we ' re": "we are",
"they're": "they are",
"they 're": "they are",
"they' re": "they are",
"they ' re": "they are",
"""