#Pour utiliser un réseau BERT il faut faire du pré-traitement de donnéesd'abord tokenize une phrase.
#Pour cela, il faut tokenize la phrase en plusieurs token créer un token de départ, de fin,
# de padding (pour arriver à la len max). Puis convertir le tout en ID.
import dataset
import transformers as trans

DS = dataset('Projet-3A\Data\data.csv', 'bert-base-uncased')
tokenizer = trans.BertTokenizer.from_pretrained('bert-base-uncased')
max_length = DS.max_length()
#On tokenize les données car le BERT n'accepte que ce format en tant qu'input et/ou pour être entraine.
sentence = "Je suis alle en Jamaïque hier soir pour faire du snowboard"
print(tokenizer.encode(sentence, add_special_tokens= True, max_length= max_length))