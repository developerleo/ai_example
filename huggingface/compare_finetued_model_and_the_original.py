import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, pipeline, AutoModelForMaskedLM

autoTokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased')

model = torch.load('models/original_bert_model.pth')
selfPipeLine = pipeline('fill-mask', model=model, tokenizer=autoTokenizer)
print(type(model))

unmasker = pipeline('fill-mask', model="google-bert/bert-base-cased")
print(type(unmasker.model))

modelForMaskedLM = AutoModelForMaskedLM.from_pretrained('google-bert/bert-base-cased')

masker = pipeline('fill-mask', model=modelForMaskedLM, tokenizer=autoTokenizer)
print(type(modelForMaskedLM))

output = unmasker("Hello I'm a [MASK] model.")
print(output)

output = masker("Hello I'm a [MASK] model.")
print(output)

output = selfPipeLine("Hello I'm a [MASK] model")
print(output)
