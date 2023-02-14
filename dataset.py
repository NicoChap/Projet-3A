import pandas as pd
import transformers as trans
import torch
from torch.utils.data import TensorDataset

class Dataset:
    file = any
    model = ''
    max_len= 0
    tokenizer = trans.BertTokenizer
    raw_inputs = ""

    def __init__(self, filename, model):
        self.file = pd.read_csv(filename)
        self.raw_inputs = self.file['Input']
        #On choisit le tokenizer adapte au BERT utilise.
        self.model = model
        self.tokenizer = trans.BertTokenizer.from_pretrained(model)

    def Encode(self, sentence, special_char = False, max_length = None):
        truncate = True
        if max_length == None:
            truncate = False
        return(self.tokenizer.encode(
            sentence, add_special_tokens= special_char, max_length= max_length, truncation=truncate))

    def EncodeAll(self):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        inputs_ids = []
        attention_masks = []
        print("Loading data...")
        # For every sentence...
        for raw_input in self.raw_inputs:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                                raw_input,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 64,           # Pad & truncate all sentences.
                                padding = 'max_length',
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                        )
            
            # Add the encoded sentence to the list.    
            inputs_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(inputs_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(self.file['Label'])
        print("Data Loaded")
        return(TensorDataset(input_ids, attention_masks, labels)) 

    def MaxLength(self):
        max_len = 0
        for i in range(self.file.shape[0]):
            sample_size= len(self.tokenizer.encode(self.raw_inputs[i], add_special_tokens= False))
            if sample_size > max_len:
                max_len = sample_size
        self.max_len= max_len
        return(max_len)

    def SaveTokenizer(output_dir):
        trans.tokenizer.save_pretrained(output_dir)