import numpy as np
from transformers import AutoTokenizer,AutoModelForSequenceClassification

labels = ['Finance','Operations','HR','Engineering ']
def classify(text,tokenizer,model):
    input = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"        
        )
    outputs = model(**input)
    logits =  outputs.logits
    pred = logits.argmax(dim=1).item()

    return labels[pred]
    
    