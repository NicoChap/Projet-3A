from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

class Classifieur:
    path = ''
    model, tokenizer = any, any
    def __init__(self, path):
        self.path = path
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
    
    def predict(self, sentence):
        encoded_sentence_dict = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens= True,
            max_length= 128,
            truncation = True,
            padding='max_length',
            return_attention_mask = True,
            return_tensors = 'pt')
        
        #encoded_sentence = torch.tensor(encoded_sentence_dict['input_ids'])

        with torch.no_grad():
        # Forward pass, calculate logit predictions.
            result = self.model(**encoded_sentence_dict)
            logits = result.logits
            prediction = torch.argmax(logits, dim=1)
        if prediction.item() == 0:
            return "chitchat"
        else:
            return "Q&A"