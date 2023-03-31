import pandas as pd
import transformers as trans
import torch
from torch.utils.data import TensorDataset
from numpy import random as rd

class Dataset:
    file = any
    model = ''
    max_len= 0
    tokenizer = trans.BertTokenizer
    raw_inputs = any
    dataset_size = 0

    def __init__(self, filename, model):
        self.file = pd.read_csv(filename)
        self.file['Index'] = self.file.index
        self.file = self.file.dropna()
        self.BalanceDataset()
        self.raw_inputs = self.file['Input']
        #On choisit le tokenizer adapte au BERT utilise.
        self.model = model
        self.tokenizer = trans.BertTokenizer.from_pretrained(model)

    def Encode(self, sentence, special_char = False, max_length = None, pt_tensors = None):
        return(self.tokenizer.encode_plus(
            sentence,
            add_special_tokens= special_char,
            max_length= max_length,
            padding='max_length',
            return_attention_mask = True,
            return_tensors= pt_tensors))

    def EncodeAll(self):
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        inputs_ids = []
        attention_masks = []
        print("Loading data...")
        # For every sentence...
        for i in range(len(self.raw_inputs)):
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            raw_input = self.raw_inputs[i]
            #print(raw_input)
            encoded_dict = self.tokenizer.encode_plus(
                                raw_input,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 64, 
                                truncation=True,        # Pad & truncate all sentences.
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
    
    def BalanceDataset(self):
        #Check which between chitchat and QA have the larger databases
        qa_df = self.file.query('(Label == 1)')
        cc_df = self.file.query('(Label == 0)')
        if len(qa_df) - len(cc_df) > 0:
            samples = qa_df.sample(len(cc_df))
            self.file = pd.concat([samples, cc_df])
        else:
            samples = cc_df.sample(len(qa_df))
            self.file = pd.concat([qa_df, samples])
        self.dataset_size = len(self.file)
        self.file = self.file.reset_index(drop=True)
        print("Dealing with", self.dataset_size,"sentences, with", self.dataset_size/2, "senteces for QA and CC." )

    def SelectFewExamples(self, proportion):
        num_samples = int(self.dataset_size * proportion)
        examples = {
            "input": [],
            "attention_mask": [],
            "label": []
        }
        for i in range(num_samples):
            index = int(rd.uniform(0,self.dataset_size))
            dict = self.Encode(self.file['Input'][index], special_char=True, max_length=64, pt_tensors = 'pt')
            examples["input"].append(dict['input_ids'])
            examples["attention_mask"].append(dict['attention_mask'])
            examples["label"].append(self.file['Label'][index])
        return(examples)

    def SaveTokenizer(self,output_dir):
        self.tokenizer.save_pretrained(output_dir)