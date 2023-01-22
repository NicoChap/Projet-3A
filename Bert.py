#Pour utiliser un réseau BERT il faut d'abord tokenize une phrase.
#Pour cela, il faut tokenize la phrase en plusieurs token créer un token de départ, de fin,
# de padding (pour arriver à la len max). Puis convertir le tout en ID.
import transformers as trans

tokenizer = trans.BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "Je suis alle en Jamaïque hier soir pour faire du snowboard"

print(sentence)
print(tokenizer.encode(sentence, add_special_tokens= False))
print(tokenizer.encode(sentence, add_special_tokens= True))