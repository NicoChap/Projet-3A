import pandas as pd
import transformers as trans

class Dataset:
    file = any
    model = ''
    max_len= 0
    tokenizer = trans.BertTokenizer

    def __init__():
        pass

    def SetFilename(self, filename):
        self.file = pd.read_csv(filename)

    def SetModel(self, model):
        #On choisit le tokenizer adapte au BERT utilise.
        self.model = model
        self.tokenizer = trans.BertTokenizer.from_pretrained(model)

    def Encode(self, sentence, special_char = False, max_length = None):
        truncate = True
        if max_length == None:
            truncate = False
        return(self.tokenizer.encode(
            sentence, add_special_tokens= special_char, max_length= max_length, truncation=truncate))

    def max_length(self):
        max_len = 0
        for i in range(self.file.shape[0]):
            sample_size= len(self.tokenizer.encode(self.file['Input'][i], add_special_tokens= False))
            if sample_size > max_len:
                max_len = sample_size
        self.max_len= max_len
        return(max_len)