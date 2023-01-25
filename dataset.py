import pandas as pd
import transformers as trans

class dataset:
    file = any
    model = ''
    max_len= 0
    tokenizer = trans.BertTokenizer
    def __init__(self, filename, model):
        self.filename = pd.read_csv(filename)
        #On choisit le tokenizer adapte au BERT utilise.
        self.tokenizer = trans.BertTokenizer.from_pretrained(model)

    def max_length(self):
        max_len = 0
        for i in range(self.file.shape[0]):
            sample_size= len(self.tokenizer.encode(self.file['Input'][i], add_special_tokens= False))
            if sample_size > max_len:
                max_len = sample_size
        self.max_len= max_len
        return(max_len)