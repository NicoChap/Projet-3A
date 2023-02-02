#Pour utiliser un réseau BERT il faut faire du pré-traitement de donnéesd'abord tokenize une phrase.
#Pour cela, il faut tokenize la phrase en plusieurs token créer un token de départ, de fin,
# de padding (pour arriver à la len max). Puis convertir le tout en ID.
from dataset import Dataset
from transformers import BertForSequenceClassification, Trainer
DS = Dataset
DS.SetFilename(DS, filename='Projet-3A\Data\data.csv')
DS.SetModel(DS, 'bert-base-uncased')
max_length = DS.max_length(DS)

#On tokenize les données.En effet, le BERT n'accepte que ce format en tant qu'input et/ou pour être entraine.

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
