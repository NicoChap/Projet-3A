#Pour utiliser un réseau BERT il faut faire du pré-traitement de donnéesd'abord tokenize une phrase.
#Pour cela, il faut tokenize la phrase en plusieurs token créer un token de départ, de fin,
# de padding (pour arriver à la len max). Puis convertir le tout en ID.
from dataset import Dataset
from transformers import BertForSequenceClassification
from torch.utils.data import random_split

DS = Dataset(filename='Data\data.csv', model='bert-base-uncased')
#On tokenize les données.En effet, le BERT n'accepte que ce format en tant qu'input 
# et/ou pour être entraine.
encoded_data = DS.EncodeAll()
train_size = int(len(encoded_data[0])*0.9)
validation_size = len(encoded_data) - train_size
train_dataset, validation_dataset = random_split(encoded_data, [train_size, validation_size])


#Import du modèle BERT :
BERTmodel = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
#Configuration pour le fine-tuning:
#AJOUTER TOUT LES VALIDATION TEST ETC. VOIR CETTE VIDEO:
#https://huggingface.co/docs/transformers/training onglet 'TRAIN'.


#Entrainement du modele BERT utilisé:
