
from transformers import AutoTokenizer

text = "I wasn't manwalker"
pretrained_model = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


tokenized_text = tokenizer(text,add_special_tokens=False,return_tensors='pt')
num_of_tokens = len(tokenized_text["input_ids"][0])
print(num_of_tokens)

encodes = tokenized_text['input_ids'][0].tolist()
decodes = [tokenizer.decode(enc) for enc in encodes]
print(decodes)
#Bert
for i in range(num_of_tokens):
    charspan = tokenized_text.token_to_chars(i)
    print(charspan.start, charspan.end-1)
    print('\n')

#Spacy
for i in range(len(doc)):
    print(doc[i:(i+1)].start_char,doc[i:(i+1)].end_char-1)